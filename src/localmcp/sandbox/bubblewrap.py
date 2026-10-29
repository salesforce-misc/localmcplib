"""Linux command execution using bubblewrap and a host-built seccomp filter."""

from __future__ import annotations

import asyncio
import json
import os
import pwd
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from localmcp.sandbox._seccomp import build_filter, find_library
from localmcp.sandbox.base import (
    DEFAULT_COMMAND_TIMEOUT_SECONDS,
    DEFAULT_MAX_OUTPUT_BYTES,
    CommandResult,
    RootAccess,
    SandboxError,
    SandboxProfile,
)
from localmcp.sandbox.command_policy import check_command
from localmcp.sandbox.policy import (
    _check_denied_paths,
    _check_scratch,
    _denied_paths_in_roots,
    _path_ancestors,
    _protected_roots,
    _validated_roots,
)
from localmcp.sandbox.runtime import (
    CommandRuntime,
    _OutputLimitExceeded,
    check_host_runtime,
    command_environment,
    limited_command,
    owned_process_limit,
    validate_command,
    validate_limits,
)

_BIN_DIRECTORIES = tuple(map(Path, ("/usr/bin", "/bin", "/usr/sbin", "/sbin")))
_RUNTIME_TREES = tuple(map(Path, ("/usr", "/nix/store")))
_RUNTIME_ALIASES = tuple(map(Path, ("/bin", "/lib", "/lib64", "/sbin")))
_ETC_PATHS = tuple(
    map(
        Path,
        ("/etc/ld.so.cache", "/etc/alternatives", "/etc/passwd", "/etc/group", "/etc/localtime", "/etc/os-release"),
    )
)
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
            "/etc/pki/tls/certs",
            "/etc/pki/ca-trust",
        ),
    )
)
_NIX_STORE = Path("/nix/store")


def _require_linux() -> None:
    if sys.platform != "linux":
        raise SandboxError("Linux bubblewrap sandbox is unavailable on this platform")


def _nix_profile_links() -> tuple[Path, ...]:
    """List supported profile links, skipping account profiles without a usable account entry."""
    try:
        account = pwd.getpwuid(os.getuid())
    except KeyError:
        account = None
    per_user = (
        (Path("/nix/var/nix/profiles/per-user") / account.pw_name / "profile",)
        if account is not None and account.pw_name and "/" not in account.pw_name
        else ()
    )
    home = Path(account.pw_dir) if account is not None else None
    in_home = (
        (home / ".nix-profile", home / ".local/state/nix/profiles/profile")
        if home is not None and home.is_absolute()
        else ()
    )
    return (Path("/nix/var/nix/profiles/default"), *per_user, Path("/run/current-system/sw"), *in_home)


def _nix_profiles() -> tuple[tuple[Path, Path], ...]:
    """Snapshot supported profile links into the store without using the caller's PATH."""
    profiles: list[tuple[Path, Path]] = []
    for profile in _nix_profile_links():
        try:
            resolved = profile.resolve(strict=True)
        except OSError:
            continue
        if resolved.is_relative_to(_NIX_STORE) and (resolved / "bin").is_dir():
            profiles.append((profile, resolved))
    return tuple(profiles)


def _program(name: str, directories: tuple[Path, ...]) -> Path | None:
    for directory in directories:
        candidate = directory / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _resolved_program(program: Path) -> Path:
    """Pin an executable's real path or report a sandbox setup failure."""
    try:
        return program.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise SandboxError(f"cannot resolve sandbox executable: {program}") from exc


def _runtime_program(program: Path, runtime_trees: tuple[Path, ...]) -> Path:
    """Require the resolved executable to be present in a mounted runtime tree."""
    resolved = _resolved_program(program)
    if not any(resolved.is_relative_to(tree) for tree in runtime_trees):
        raise SandboxError(f"sandbox executable is outside the mounted runtime: {resolved}")
    return resolved


def _tools(profile: SandboxProfile, directories: tuple[Path, ...], runtime_trees: tuple[Path, ...]) -> tuple[str, ...]:
    granted: list[str] = []
    for name in (*profile.tools, *profile.optional_tools):
        program = _program(name, directories)
        try:
            if program is None:
                raise SandboxError(f"sandbox tool is not installed: {name}")
            _runtime_program(program, runtime_trees)
        except SandboxError:
            if name in profile.tools:
                raise
        else:
            granted.append(name)
    return tuple(dict.fromkeys(granted))


class LinuxSandbox(CommandRuntime):
    """Run bounded Bash commands with filesystem, namespace, and syscall restrictions."""

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
        check_host_runtime(self.roots)
        self.root = self.roots[0].path
        self.profile = SandboxProfile(
            self.roots,
            _denied_paths_in_roots(profile.denied_paths, self.roots),
            profile.network,
            profile.ipc,
            tuple(dict.fromkeys(profile.tools)),
            tuple(dict.fromkeys(profile.optional_tools)),
        )
        self._protected_paths = _protected_roots(self.roots)
        self._validate_paths()
        self._profiles = _nix_profiles()
        self._bin_directories = tuple(dict.fromkeys((store / "bin").resolve() for _, store in self._profiles))
        self._runtime_trees = tuple(path for path in _RUNTIME_TREES if path.is_dir())
        self._runtime_aliases = tuple(path for path in _RUNTIME_ALIASES if path.exists())
        mounted_runtime = tuple(
            path.resolve()
            for path in (*self._runtime_trees, *(path for path in self._runtime_aliases if not path.is_symlink()))
        )
        directories = (*self._bin_directories, *_BIN_DIRECTORIES)
        bwrap = _program("bwrap", (*_BIN_DIRECTORIES, *self._bin_directories))
        shell = _program("bash", directories)
        if bwrap is None or shell is None:
            raise SandboxError("Linux sandbox requires bubblewrap and bash in system directories or a Nix profile")
        # Bubblewrap runs on the host; only workload executables need runtime mounts.
        self._bwrap = _resolved_program(bwrap)
        if any(root.access == RootAccess.READ_WRITE and self._bwrap.is_relative_to(root.path) for root in self.roots):
            raise SandboxError(f"writable sandbox root overlaps the host bubblewrap launcher: {self._bwrap}")
        self._shell = _runtime_program(shell, mounted_runtime)
        self.tools = _tools(self.profile, directories, mounted_runtime)
        self._etc_paths = tuple(
            path for path in (*_ETC_PATHS, *(_NETWORK_PATHS if profile.network else ())) if path.exists()
        )
        grants = (
            *self._runtime_trees,
            *self._runtime_aliases,
            *(link for link, _ in self._profiles),
            Path("/etc"),
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
        library = find_library(tuple(store for _, store in self._profiles))
        self._seccomp_filter = build_filter(library, network=profile.network, ipc=profile.ipc)

    def _validate_paths(self) -> None:
        _check_denied_paths(self.profile.denied_paths)
        _check_denied_paths(self._protected_paths, "read-only root")

    def _writable(self, path: Path) -> bool:
        return any(
            root.access == RootAccess.READ_WRITE and path.is_relative_to(root.path) for root in self.roots
        ) and not any(path.is_relative_to(protected) for protected in self._protected_paths)

    def _invocation(self, command: str, scratch: Path, control: Path) -> list[str]:
        argv = [
            str(self._bwrap),
            "--unshare-user",
            "--uid",
            str(os.getuid()),
            "--gid",
            str(os.getgid()),
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--unshare-cgroup",
            "--die-with-parent",
            "--new-session",
            "--cap-drop",
            "ALL",
            "--clearenv",
        ]
        if not self.profile.network:
            argv.append("--unshare-net")
        for name, value in command_environment(scratch, bin_directories=self._bin_directories).items():
            argv.extend(["--setenv", name, value])

        def bind(source: Path, target: Path, *, read_only: bool = True) -> None:
            argv.extend(["--ro-bind" if read_only else "--bind", str(source), str(target)])

        for tree in self._runtime_trees:
            bind(tree, tree)
        for path in self._runtime_aliases:
            if path.is_symlink():
                argv.extend(["--symlink", os.readlink(path), str(path)])
            else:
                bind(path, path)
        for path in self._etc_paths:
            bind(path.resolve(), path)
        for profile, store in self._profiles:
            argv.extend(["--symlink", str(store), str(profile)])

        present = tuple(
            path
            for path in self.profile.denied_paths
            if path.exists()
            and not any(path != other and path.is_relative_to(other) for other in self.profile.denied_paths)
        )
        mounts = {root.path: root.access == RootAccess.READ_ONLY for root in self.roots}
        for path in _path_ancestors((*present, *self._protected_paths)):
            if any(path.is_relative_to(root.path) for root in self.roots):
                mounts.setdefault(path, not self._writable(path))
        for path in sorted(mounts, key=lambda item: (len(item.parts), str(item))):
            bind(path, path, read_only=mounts[path])
        # Scratch stays on the host's temporary volume rather than consuming memory as a tmpfs.
        bind(scratch, scratch, read_only=not any(root.access == RootAccess.READ_WRITE for root in self.roots))
        argv.extend(["--dev", "/dev", "--dir", "/var/empty"])
        if self.profile.ipc:
            argv.extend(["--tmpfs", "/dev/shm"])
        argv.extend(["--remount-ro", "/dev"])
        for path in present:
            bind(control / ("deny-dir" if path.is_dir() else "deny-file"), path)
        # Only explicit writable mounts offer storage; synthetic ancestors stay read-only.
        argv.extend(["--remount-ro", "/", "--chdir", str(self.root)])
        argv.extend(["--", str(self._shell), "--noprofile", "--norc", "-c", command, self.process_label])
        return argv

    async def run(self, command: str) -> CommandResult:
        validate_command(command)
        if reason := check_command(command, self.tools):
            return CommandResult(exit_code=126, stdout="", stderr=f"localmcp: {reason}")
        await asyncio.to_thread(self._validate_paths)
        with (
            tempfile.TemporaryDirectory(prefix="localmcp-", dir=self._scratch_parent) as scratch_name,
            tempfile.TemporaryDirectory(prefix="localmcp-control-", dir=self._scratch_parent) as control_name,
        ):
            scratch, control = Path(scratch_name), Path(control_name)
            _check_scratch(scratch, self.roots)
            _check_scratch(control, self.roots)
            (control / "deny-file").touch(mode=0o000)
            (control / "deny-dir").mkdir(mode=0o000)
            argv = await asyncio.to_thread(self._invocation, command, scratch, control)
            with (
                (control / "info").open("w+b") as info,
                (control / "status").open("w+b") as status,
                (control / "seccomp").open("w+b") as seccomp,
            ):
                seccomp.write(self._seccomp_filter)
                seccomp.flush()
                seccomp.seek(0)
                argv[1:1] = [
                    "--info-fd",
                    str(info.fileno()),
                    "--json-status-fd",
                    str(status.fileno()),
                    "--seccomp",
                    str(seccomp.fileno()),
                ]
                maximum_processes = await asyncio.to_thread(owned_process_limit)
                argv = limited_command(argv, self.timeout_seconds, maximum_processes)
                try:
                    process = await asyncio.create_subprocess_exec(
                        *argv,
                        env=command_environment(scratch, bin_directories=self._bin_directories),
                        cwd=self.root,
                        stdin=asyncio.subprocess.DEVNULL,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        start_new_session=True,
                        pass_fds=(info.fileno(), status.fileno(), seccomp.fileno()),
                    )
                except (OSError, ValueError, subprocess.SubprocessError) as exc:
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
                    records = [json.loads(line) for line in (control / "status").read_text().splitlines()]
                    exits = [record["exit-code"] for record in records if "exit-code" in record]
                    if len(exits) != 1 or type(exits[0]) is not int or exits[0] != process.returncode:
                        raise ValueError("missing or inconsistent bubblewrap exit status")
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    detail = stderr.decode(errors="replace").strip()
                    raise SandboxError(f"Linux sandbox setup or supervision failed: {detail}") from exc
                assert process.returncode is not None
                return CommandResult(
                    exit_code=process.returncode,
                    stdout=stdout.decode(errors="replace"),
                    stderr=stderr.decode(errors="replace"),
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
