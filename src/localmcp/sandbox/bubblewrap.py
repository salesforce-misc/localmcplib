"""Linux command execution using bubblewrap, Landlock, and seccomp.

The backend deliberately fails closed when the host cannot enforce its policy.
See the sandbox documentation for the Linux filesystem and script-tool limitations.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import resource
import signal
import stat
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from localmcp.sandbox.base import (
    DEFAULT_COMMAND_TIMEOUT_SECONDS,
    DEFAULT_MAX_OUTPUT_BYTES,
    MAX_ADDITIONAL_PROCESSES,
    MAX_OPEN_FILES,
    CommandResult,
    RootAccess,
    SandboxError,
    SandboxProfile,
)
from localmcp.sandbox.policy import (
    _check_denied_paths,
    _check_scratch,
    _path_ancestors,
    _protected_roots,
    _validated_roots,
)
from localmcp.sandbox.runtime import (
    CommandRuntime,
    _OutputLimitExceeded,
    command_environment,
    validate_command,
    validate_limits,
)

_BIN_DIRECTORIES = tuple(map(Path, ("/usr/bin", "/bin", "/usr/sbin", "/sbin")))
_RUNTIME_TREES = tuple(map(Path, ("/usr/bin", "/usr/sbin", "/usr/lib", "/usr/lib64", "/usr/libexec")))
_NETWORK_PATHS = tuple(
    map(
        Path,
        (
            "/etc/resolv.conf",
            "/etc/hosts",
            "/etc/nsswitch.conf",
            "/etc/gai.conf",
            "/etc/ssl/certs",
            "/etc/ssl/openssl.cnf",
            "/usr/share/ca-certificates",
        ),
    )
)
_CONTROL = Path("/localmcp")
# Snapshot trusted launcher code before a workload can modify an editable install.
# Reading this file again per invocation would make a writable source checkout
# capable of replacing the next command's privileged setup code.
_RUNNER_SOURCE = Path(__file__).with_name("_linux_runner.py").read_bytes()


def _require_linux() -> None:
    if sys.platform != "linux":
        raise SandboxError("Linux bubblewrap sandbox is unavailable on this platform")


def _trusted_program(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    if not any(resolved.is_relative_to(tree.resolve()) for tree in _RUNTIME_TREES if tree.is_dir()):
        raise SandboxError(f"Linux sandbox tool code is outside the system runtime: {resolved}")
    return resolved


def _program(name: str) -> Path | None:
    for directory in _BIN_DIRECTORIES:
        candidate = directory / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _multicall(program: Path) -> bool:
    providers = ("busybox", "coreutils", "toybox")
    if program.name in providers:
        return True
    return any(
        candidate.is_file() and program.samefile(candidate)
        for directory in _BIN_DIRECTORIES
        for name in providers
        for candidate in (directory / name,)
    )


def _helpers(tool: str) -> tuple[Path, ...]:
    return tuple(
        helper.resolve()
        for parent in (Path("/usr/lib"), Path("/usr/libexec"))
        for helper in (parent / tool, parent / f"{tool}-core")
        if helper.is_dir()
    )


def _interpreter(path: Path) -> Path | None:
    with path.open("rb") as stream:
        line = stream.readline(512)
    words = line[2:].split() if line.startswith(b"#!") else []
    return Path(os.fsdecode(words[0])).resolve(strict=True) if words else None


def _elf_interpreter(path: Path) -> Path | None:
    """Read PT_INTERP without executing the program or consulting its environment."""
    with path.open("rb") as stream:
        header = stream.read(64)
        if not header.startswith(b"\x7fELF"):
            return None
        if len(header) < 52 or header[4] not in (1, 2) or header[5] not in (1, 2):
            raise SandboxError(f"unsupported ELF program: {path}")
        endian = "<" if header[5] == 1 else ">"
        wide = header[4] == 2
        offset = struct.unpack_from(endian + ("Q" if wide else "I"), header, 32 if wide else 28)[0]
        size, count = struct.unpack_from(endian + "HH", header, 54 if wide else 42)
        for index in range(count):
            stream.seek(offset + index * size)
            entry = stream.read(size)
            if len(entry) < (56 if wide else 32):
                raise SandboxError(f"invalid ELF program headers: {path}")
            if struct.unpack_from(endian + "I", entry)[0] == 3:
                start = struct.unpack_from(endian + ("Q" if wide else "I"), entry, 8 if wide else 4)[0]
                length = struct.unpack_from(endian + ("Q" if wide else "I"), entry, 32 if wide else 16)[0]
                if length > 4096:
                    raise SandboxError(f"invalid ELF interpreter: {path}")
                stream.seek(start)
                return Path(os.fsdecode(stream.read(length).rstrip(b"\0"))).resolve(strict=True)
    return None


def _libraries(program: Path) -> tuple[Path, ...]:
    """Inspect trusted distro programs only; workload programs never reach ldd."""
    _trusted_program(program)
    if _elf_interpreter(program) is None:
        return ()
    try:
        result = subprocess.run(
            ["/usr/bin/ldd", os.fspath(program)],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise SandboxError("Linux sandbox requires the system ldd utility") from exc
    if result.returncode or "not found" in result.stdout:
        raise SandboxError(f"cannot resolve shared libraries for sandbox tool: {program}")
    paths: list[Path] = []
    for line in result.stdout.splitlines():
        match = re.search(r"(?:^\s*|=>\s*)(/\S+)\s+\(", line)
        if match:
            paths.append(_trusted_program(Path(match[1])))
    if not paths:
        raise SandboxError(f"cannot resolve shared libraries for sandbox tool: {program}")
    return tuple(dict.fromkeys(paths))


@dataclass(frozen=True)
class _Tools:
    names: tuple[str, ...]
    executables: tuple[Path, ...]
    helpers: tuple[Path, ...]
    maps: tuple[Path, ...]
    data: tuple[Path, ...]
    aliases: tuple[tuple[Path, str], ...]


def _tool_grants(profile: SandboxProfile) -> _Tools:
    shell = _trusted_program(Path("/bin/sh"))
    required = set(profile.tools)
    programs: dict[str, Path] = {}
    for name in (*profile.tools, *profile.optional_tools):
        program = _program(name)
        if program is None:
            if name in required:
                raise SandboxError(f"sandbox tool is not installed: {name}")
            continue
        programs[name] = program
    explicitly_allowed = {shell, *(program.resolve() for name, program in programs.items() if name in required)}
    executables = [shell]
    helpers: list[Path] = []
    maps: list[Path] = []
    data: list[Path] = []
    aliases: dict[Path, str] = {}
    granted: list[str] = []
    for name, program in programs.items():
        try:
            resolved = _trusted_program(program)
            interpreter = _interpreter(resolved)
            if interpreter is not None and interpreter not in explicitly_allowed:
                raise SandboxError(
                    f"Linux sandbox script tool {name!r} requires its interpreter to be explicitly listed in tools"
                )
            if _multicall(resolved):
                raise SandboxError(f"Linux sandbox cannot restrict individual commands of multicall tool: {name}")
            tool_helpers = _helpers(name)
            # Landlock grants execution by where a symlink's target lives, and a target
            # inside a helper tree is scanned itself. Links out of the tree, such as
            # package data under /usr/share, cannot execute and are skipped, not rejected.
            native_helpers = {
                _trusted_program(candidate)
                for tree in tool_helpers
                for candidate in tree.rglob("*")
                if not candidate.is_symlink() and candidate.is_file() and os.access(candidate, os.X_OK)
            }
            code = (interpreter or resolved, *sorted(native_helpers, key=os.fspath))
            libraries = tuple(library for program in code for library in _libraries(program))
            loaders = tuple(loader for program in code if (loader := _elf_interpreter(program)) is not None)
        except (OSError, SandboxError) as exc:
            if name in required:
                if isinstance(exc, SandboxError):
                    raise
                raise SandboxError(f"cannot inspect sandbox tool: {name}") from exc
            continue
        granted.append(name)
        executables.extend((resolved, *loaders))
        helpers.extend(tool_helpers)
        maps.extend(libraries)
        # Alternatives links are reproduced individually, without exposing all of /etc.
        link = program
        seen: set[Path] = set()
        while link.is_symlink() and link not in seen:
            seen.add(link)
            target = os.readlink(link)
            if link.is_relative_to("/etc"):
                aliases[link] = target
            link = Path(os.path.abspath(link.parent / target))
        for candidate in (Path("/usr/share") / name, Path("/usr/share") / f"{name}-core"):
            if candidate.exists():
                data.append(candidate.resolve())
        if name == "file":
            for candidate in (
                Path("/usr/share/misc/magic"),
                Path("/usr/share/misc/magic.mgc"),
                Path("/usr/share/file/magic.mgc"),
            ):
                if candidate.exists():
                    data.append(candidate.resolve())
                    if candidate.is_symlink():
                        aliases[candidate] = os.readlink(candidate)
    maps.extend(_libraries(shell))
    if interpreter := _elf_interpreter(shell):
        executables.append(interpreter)
    return _Tools(
        tuple(granted),
        tuple(dict.fromkeys(executables)),
        tuple(dict.fromkeys(helpers)),
        tuple(dict.fromkeys(maps)),
        tuple(dict.fromkeys(data)),
        tuple(aliases.items()),
    )


def _process_limit() -> int | None:
    try:
        owned = sum(entry.stat().st_uid == os.getuid() for entry in Path("/proc").iterdir() if entry.name.isdecimal())
        soft, _ = resource.getrlimit(resource.RLIMIT_NPROC)
    except OSError:
        return None
    proposed = owned + MAX_ADDITIONAL_PROCESSES
    if soft != resource.RLIM_INFINITY:
        proposed = min(proposed, soft)
    return proposed if proposed > owned else None


class LinuxSandbox(CommandRuntime):
    """Run bounded commands under mandatory Linux namespace and syscall enforcement."""

    def __init__(
        self,
        profile: SandboxProfile,
        *,
        timeout_seconds: float = DEFAULT_COMMAND_TIMEOUT_SECONDS,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        process_label: str = "localmcp-workspace-sandbox",
    ) -> None:
        validate_limits(timeout_seconds, max_output_bytes, process_label)
        _require_linux()
        self.roots = _validated_roots(profile.roots)
        self.root = self.roots[0].path
        denied = tuple(dict.fromkeys(Path(os.path.abspath(path)) for path in profile.denied_paths))
        if any(not any(path.is_relative_to(root.path) for root in self.roots) for path in denied):
            raise SandboxError("sandbox denied path is outside its declared roots")
        self.profile = SandboxProfile(
            self.roots,
            denied,
            profile.network,
            profile.ipc,
            tuple(dict.fromkeys(profile.tools)),
            tuple(dict.fromkeys(profile.optional_tools)),
        )
        self._protected_paths = _protected_roots(self.roots)
        self._validate_paths()
        self._grants = _tool_grants(self.profile)
        self.tools = self._grants.names
        self._runtime_trees = tuple(dict.fromkeys(path.resolve() for path in _RUNTIME_TREES if path.is_dir()))
        grants = (
            *self._runtime_trees,
            *self._grants.executables,
            *self._grants.helpers,
            *self._grants.maps,
            *self._grants.data,
            _CONTROL,
            Path("/proc"),
            Path("/dev"),
            Path("/var/empty"),
        )
        for path in grants:
            if any(path.is_relative_to(root.path) or root.path.is_relative_to(path) for root in self.roots):
                raise SandboxError(f"sandbox tools or runtime overlap a sandbox root: {path}")
        self._scratch_parent = Path(tempfile.gettempdir()).resolve()
        _check_scratch(self._scratch_parent, self.roots)
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes
        self.process_label = process_label
        self._bwrap = Path("/usr/bin/bwrap")
        self._python = Path("/usr/bin/python3")
        self._process_limit = _process_limit()

    def _validate_paths(self) -> None:
        _check_denied_paths(self.profile.denied_paths)
        _check_denied_paths(self._protected_paths, "read-only root")
        for path in self.profile.denied_paths:
            try:
                mode = path.lstat().st_mode
            except FileNotFoundError as exc:
                raise SandboxError(f"Linux sandbox denied path must exist: {path}") from exc
            if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                raise SandboxError(f"Linux sandbox denied path must be a regular file or directory: {path}")

    def _invocation(self, command: str, scratch: Path, control: Path) -> tuple[list[str], dict[str, object]]:
        writable = any(root.access == RootAccess.READ_WRITE for root in self.roots)
        argv = [
            str(self._bwrap),
            "--unshare-user",
            "--uid",
            "0",
            "--gid",
            "0",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--as-pid-1",
            "--die-with-parent",
            "--new-session",
            "--cap-drop",
            "ALL",
            "--cap-add",
            "CAP_SYS_ADMIN",
            "--cap-add",
            "CAP_SETPCAP",
        ]
        if not self.profile.network:
            argv.append("--unshare-net")
        readonly: list[str] = []

        def bind(source: Path, target: Path, *, read_only: bool = True) -> None:
            argv.extend(["--ro-bind" if read_only else "--bind", str(source), str(target)])
            if read_only:
                readonly.append(str(target))

        for tree in self._runtime_trees:
            bind(tree, tree)
        for name in ("bin", "sbin", "lib", "lib64"):
            path = Path("/") / name
            if path.is_symlink():
                argv.extend(["--symlink", os.readlink(path), str(path)])
            elif path.is_dir():
                bind(path, path)
        cache = Path("/etc/ld.so.cache")
        if cache.is_file():
            bind(cache, cache)
        for path, target in self._grants.aliases:
            argv.extend(["--symlink", target, str(path)])
        for path in self._grants.data:
            if not any(path.is_relative_to(tree) for tree in self._runtime_trees):
                bind(path, path)
        if self.profile.network:
            for path in _NETWORK_PATHS:
                if path.exists():
                    bind(path.resolve(), path)
        mounts = {root.path: root.access == RootAccess.READ_ONLY for root in self.roots}
        # Pin ancestors as mount points so a rename cannot relocate a protected path.
        for path in _path_ancestors((*self.profile.denied_paths, *self._protected_paths)):
            if any(path.is_relative_to(root.path) for root in self.roots):
                mounts.setdefault(
                    path,
                    not any(
                        root.access == RootAccess.READ_WRITE and path.is_relative_to(root.path) for root in self.roots
                    ),
                )
        for path in sorted(mounts, key=lambda item: (len(item.parts), str(item))):
            bind(path, path, read_only=mounts[path])
        bind(scratch, scratch, read_only=not writable)
        for path in dict.fromkeys((*self._grants.helpers, *self._grants.executables, *self._grants.maps)):
            bind(path, path)
        argv.extend(["--proc", "/proc", "--dev", "/dev", "--dir", "/var/empty", "--dir", str(_CONTROL)])
        if self.profile.ipc:
            argv.extend(["--tmpfs", "/dev/shm"])
        # Keep the device filesystem from becoming another writable temporary
        # directory. A separately mounted shared-memory surface remains writable.
        argv.extend(["--remount-ro", "/dev", "--remount-ro", "/proc"])
        for name in ("runner.py", "profile.json"):
            bind(control / name, _CONTROL / name)
        bind(control / "status", _CONTROL / "status", read_only=False)
        # Cover denials last; children of an already covered directory add no authority.
        for path in self.profile.denied_paths:
            if any(path != other and path.is_relative_to(other) for other in self.profile.denied_paths):
                continue
            bind(control / ("deny-dir" if path.is_dir() else "deny-file"), path)
        argv.extend(
            [
                "--chdir",
                str(self.root),
                "--",
                str(self._python),
                "-I",
                "-S",
                str(_CONTROL / "runner.py"),
                str(_CONTROL / "profile.json"),
            ]
        )
        write_paths = [str(root.path) for root in self.roots if root.access == RootAccess.READ_WRITE]
        if writable:
            write_paths.append(str(scratch))
        write_paths.extend(["/dev/null", "/dev/zero", "/dev/random", "/dev/urandom"])
        if self.profile.ipc:
            write_paths.append("/dev/shm")
        profile: dict[str, object] = {
            "command": command,
            "process_label": self.process_label,
            "network": self.profile.network,
            "ipc": self.profile.ipc,
            "executables": list(map(str, self._grants.executables)),
            "executable_trees": list(map(str, self._grants.helpers)),
            "executable_maps": list(map(str, self._grants.maps)),
            "readonly_mounts": readonly,
            "read_directories": [str(root.path) for root in self.roots]
            + list(map(str, self._runtime_trees))
            + [str(scratch), "/dev", "/proc"]
            + [str(path) for path in self._grants.data if path.is_dir()]
            + [str(path) for path in _NETWORK_PATHS if self.profile.network and path.is_dir()],
            "write_paths": write_paths,
            "timeout_seconds": self.timeout_seconds,
            "process_limit": self._process_limit,
            "max_open_files": MAX_OPEN_FILES,
        }
        return argv, profile

    async def run(self, command: str) -> CommandResult:
        validate_command(command)
        if not self._bwrap.is_file() or not self._python.is_file():
            raise SandboxError("Linux sandbox requires /usr/bin/bwrap and /usr/bin/python3")
        await asyncio.to_thread(self._validate_paths)
        with (
            tempfile.TemporaryDirectory(prefix="localmcp-", dir=self._scratch_parent) as scratch_name,
            tempfile.TemporaryDirectory(prefix="localmcp-control-", dir=self._scratch_parent) as control_name,
        ):
            scratch, control = Path(scratch_name), Path(control_name)
            _check_scratch(scratch, self.roots)
            _check_scratch(control, self.roots)
            (control / "runner.py").write_bytes(_RUNNER_SOURCE)
            (control / "status").touch(mode=0o600)
            (control / "deny-file").touch(mode=0o000)
            (control / "deny-dir").mkdir(mode=0o000)
            argv, profile = self._invocation(command, scratch, control)
            (control / "profile.json").write_text(json.dumps(profile))
            # bwrap's info FD reports the host PID of our namespace init, never a workload-owned FD.
            with (control / "info").open("w+b") as info:
                argv[1:1] = ["--info-fd", str(info.fileno())]
                try:
                    process = await asyncio.create_subprocess_exec(
                        *argv,
                        env=command_environment(scratch),
                        cwd=self.root,
                        stdin=asyncio.subprocess.DEVNULL,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        start_new_session=True,
                        pass_fds=(info.fileno(),),
                    )
                except OSError as exc:
                    raise SandboxError(f"cannot start Linux sandbox: {exc}") from exc
                stdout, stderr = bytearray(), bytearray()
                try:
                    await asyncio.wait_for(self._read_output(process, stdout, stderr), self.timeout_seconds)
                except (TimeoutError, _OutputLimitExceeded) as exc:
                    await self._kill(process, control / "info")
                    timed_out = isinstance(exc, TimeoutError)
                    message = "command timed out" if timed_out else "command output exceeded its bounded safety limit"
                    return CommandResult(
                        exit_code=-1,
                        stdout=stdout.decode(errors="replace"),
                        stderr=self._append_error(stderr, message),
                        timed_out=timed_out,
                        truncated=not timed_out,
                    )
                except BaseException:
                    await self._kill(process, control / "info")
                    raise
                try:
                    status = json.loads((control / "status").read_text())
                    if "error" in status:
                        raise SandboxError(f"Linux sandbox setup failed: {status['error']}")
                    exit_code = status["exit_code"]
                    if not isinstance(exit_code, int) or process.returncode != 0:
                        raise ValueError("invalid supervisor exit")
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    detail = stderr.decode(errors="replace").strip()
                    raise SandboxError(f"Linux sandbox setup or supervision failed: {detail}") from exc
                return CommandResult(
                    exit_code=exit_code, stdout=stdout.decode(errors="replace"), stderr=stderr.decode(errors="replace")
                )

    @staticmethod
    async def _kill(process: asyncio.subprocess.Process, info: Path) -> None:
        if process.returncode is None:
            try:
                child = json.loads(info.read_text()).get("child-pid")
                if isinstance(child, int) and child > 0:
                    os.kill(child, signal.SIGKILL)
            except (OSError, ValueError):
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        await process.communicate()
