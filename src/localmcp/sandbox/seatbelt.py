"""macOS Seatbelt execution for model-controlled commands."""

from __future__ import annotations

import asyncio
import ctypes
import json
import os
import signal
import subprocess
import tempfile
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

from langchain_core.tools import StructuredTool

from localmcp.sandbox.base import (
    DEFAULT_COMMAND_TIMEOUT_SECONDS,
    DEFAULT_MAX_OUTPUT_BYTES,
    MAX_OPEN_FILES,
    CommandResult,
    RootAccess,
    SandboxError,
    SandboxProfile,
)
from localmcp.sandbox.base import (
    INSPECTION_TOOLS as INSPECTION_TOOLS,
)
from localmcp.sandbox.base import (
    MAX_COMMAND_CHARACTERS as MAX_COMMAND_CHARACTERS,
)
from localmcp.sandbox.base import (
    SandboxRoot as SandboxRoot,
)
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
    command_environment,
    process_limit,
    validate_command,
    validate_limits,
)
from localmcp.sandbox.tools import (
    DEFAULT_EXHAUSTED_MESSAGE,
    ToolCallBudget,
    _sandbox_tools,
)
from localmcp.sandbox.tools import (
    DEFAULT_TOOL_DESCRIPTION as DEFAULT_TOOL_DESCRIPTION,
)
from localmcp.sandbox.tools import (
    BashInput as BashInput,
)
from localmcp.sandbox.tools import (
    _tool_description as _tool_description,
)

_BASE_PROFILE = """\
(version 1)
(deny default)
(import "system.sb")
(allow process-fork)
(allow file-read*
    (subpath "/bin")
    (subpath "/usr/bin")
    (subpath "/usr/lib")
    (subpath "/System")
    (subpath "/Library/Apple")
    (subpath "/private/var/select"))
"""
_NETWORK_PROFILE = "(allow network-outbound)\n"
# The system TLS library, which curl uses, needs its configuration and trust store to start.
_TLS_FILES = (Path("/private/etc/ssl/openssl.cnf"), Path("/private/etc/ssl/cert.pem"))
_SHARED_MEMORY_IPC_PROFILE = "(allow ipc-posix-shm ipc-sysv-shm)\n"


def _tls_files() -> tuple[Path, ...]:
    """Return the system TLS files and the target of each one that is a symlink.

    Seatbelt checks the real path a file is opened by, so a host that replaces its
    trust store with a symlink needs the target granted as well.
    """
    return tuple(dict.fromkeys(path for file in _TLS_FILES for path in (file, Path(os.path.realpath(file)))))


# Xcode records license acceptance here; the /usr/bin shims refuse to run without reading it.
_XCODE_LICENSE = Path("/Library/Preferences/com.apple.dt.Xcode.plist")


@lru_cache(maxsize=1)
def _developer_directory() -> Path | None:
    """Return the system-selected developer directory, or ``None`` when none is usable.

    ``xcode-select -p`` runs without the caller's environment, so ``DEVELOPER_DIR`` cannot
    point the sandbox's read grant elsewhere. Unlike ``xcrun`` or a ``/usr/bin`` shim, it
    cannot raise the Command Line Tools install prompt.
    """
    try:
        result = subprocess.run(
            ["/usr/bin/xcode-select", "-p"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            env={"PATH": "/usr/bin:/bin"},
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    if result.returncode != 0 or not value:
        return None
    try:
        path = Path(value).resolve(strict=True)
    except OSError:
        return None
    return path if (path / "usr" / "bin").is_dir() else None


def _developer_installation(developer: Path) -> Path:
    """Return the tree the toolchain reads from: the whole Xcode.app, or the directory itself.

    Inside Xcode.app the shims also read Contents/Info.plist and load Contents/SharedFrameworks,
    so granting only Contents/Developer is not enough. A Command Line Tools install is self-contained.
    """
    bundle = developer.parent.parent
    if developer.name == "Developer" and developer.parent.name == "Contents" and bundle.suffix == ".app":
        return bundle
    return developer


def _developer_bin_directories(developer: Path) -> tuple[Path, ...]:
    """Return the toolchain's own bin directories, so commands skip the slow, noisy /usr/bin shims."""
    candidates = (
        developer / "usr" / "bin",
        developer / "Toolchains" / "XcodeDefault.xctoolchain" / "usr" / "bin",
    )
    return tuple(candidate for candidate in candidates if candidate.is_dir())


# Homebrew's default prefixes on Apple silicon and on Intel.
_HOMEBREW_PREFIXES = (Path("/opt/homebrew"), Path("/usr/local"))
_SYSTEM_BIN_DIRECTORIES = (Path("/usr/bin"), Path("/bin"), Path("/usr/sbin"), Path("/sbin"))


def _homebrew_prefix() -> Path | None:
    """Return the Homebrew prefix, or ``None`` when Homebrew is not installed.

    Only the default prefixes are considered, never the caller's ``HOMEBREW_PREFIX``. Nothing
    is executed: a prefix qualifies when it holds ``bin/brew``.
    """
    for candidate in _HOMEBREW_PREFIXES:
        try:
            prefix = candidate.resolve(strict=True)
        except OSError:
            continue
        if (prefix / "bin" / "brew").is_file():
            return prefix
    return None


def _homebrew_keg(prefix: Path, link: Path) -> Path | None:
    """Return the installed formula version (``Cellar/<name>/<version>``) ``link`` resolves into."""
    try:
        cellar = (prefix / "Cellar").resolve(strict=True)
        relative = link.resolve(strict=True).relative_to(cellar)
    except (OSError, ValueError):
        return None
    if len(relative.parts) < 2:
        return None
    return cellar / relative.parts[0] / relative.parts[1]


def _homebrew_dependencies(keg: Path) -> tuple[str, ...]:
    try:
        receipt = json.loads((keg / "INSTALL_RECEIPT.json").read_text())
        dependencies = receipt.get("runtime_dependencies") or ()
        return tuple(dependency["full_name"].rsplit("/", 1)[-1] for dependency in dependencies)
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        raise SandboxError(f"cannot read the Homebrew install receipt of {keg}") from exc


def _homebrew_formula(prefix: Path, keg: Path) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Return ``keg`` and the kegs of its runtime dependencies, plus the ``opt`` links naming them.

    Formula code reaches itself and its libraries through ``opt/<name>`` links, which Homebrew
    points at the installed version, and each keg's install receipt lists its dependencies.
    """
    kegs: list[Path] = []
    links: list[Path] = []
    pending = [keg]
    while pending:
        current = pending.pop()
        if current in kegs:
            continue
        kegs.append(current)
        link = prefix / "opt" / current.parent.name
        if link.is_symlink() and link not in links:
            links.append(link)
        for name in _homebrew_dependencies(current):
            dependency = _homebrew_keg(prefix, prefix / "opt" / name)
            if dependency is None:
                raise SandboxError(f"Homebrew dependency of {current.parent.name} is not installed: {name}")
            pending.append(dependency)
    return tuple(kegs), tuple(links)


def _executable(name: str, directories: tuple[Path, ...]) -> Path | None:
    for directory in directories:
        candidate = directory / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _homebrew_tool(prefix: Path, tool: str) -> tuple[Path, Path] | None:
    """Return the prefix's link to ``tool`` and the formula keg it runs from."""
    link = _executable(tool, (prefix / "bin", prefix / "sbin"))
    keg = None if link is None else _homebrew_keg(prefix, link)
    return None if link is None or keg is None else (link, keg)


@dataclass(frozen=True)
class _ToolGrants:
    """Grants that let the selected tools run, and the PATH entries that find them.

    ``trees`` and ``files`` are read-only. Only ``executables``, the programs under
    ``executable_trees`` and, as script interpreters only, ``interpreters`` may be executed.
    ``tools`` names the tools granted.
    """

    trees: tuple[Path, ...] = ()
    files: tuple[Path, ...] = ()
    bin_directories: tuple[Path, ...] = ()
    developer: Path | None = None
    executables: tuple[Path, ...] = ()
    executable_trees: tuple[Path, ...] = ()
    interpreters: tuple[Path, ...] = ()
    tools: tuple[str, ...] = ()


# Where the shims live that run a tool from the selected Xcode or Command Line Tools install.
_SHIM_DIRECTORY = Path("/usr/bin")
# Every shim hands off to the selected install through this libxcselect function.
_SHIM_MARKER = b"_xcselect_invoke_xcrun"
# /bin/sh runs the shell selected here, bash by default, in its place.
_SHELL = Path("/bin/sh")
_SHELL_SELECTION = Path("/private/var/select/sh")


def _shell_executables() -> tuple[Path, ...]:
    try:
        selected = _SHELL_SELECTION.resolve(strict=True)
    except OSError:
        selected = Path("/bin/bash")
    return tuple(dict.fromkeys((_SHELL, selected)))


def _interpreter(program: Path) -> Path | None:
    """Return the interpreter a ``#!`` script names, or ``None`` for any other program."""
    try:
        with program.open("rb") as file:
            line = file.readline(512)
    except OSError:
        return None
    words = line[2:].split() if line.startswith(b"#!") else []
    return Path(os.fsdecode(words[0])).resolve() if words else None


def _is_shim(program: Path) -> bool:
    """Return whether ``program`` runs a tool from the selected developer install instead of itself.

    A program that cannot be read counts as one.
    """
    try:
        return _SHIM_MARKER in program.read_bytes()
    except OSError:
        return True


def _framework_version(program: Path) -> Path | None:
    for ancestor in program.parents:
        if ancestor.parent.name == "Versions" and ancestor.parent.parent.suffix == ".framework":
            return ancestor
    return None


def _runnable(grants: _ToolGrants, programs: tuple[Path, ...], helpers: tuple[Path, ...] = ()) -> _ToolGrants:
    """Add the exec grants that let ``programs``, and the helper programs under ``helpers``, run.

    Seatbelt checks each exec's resolved path. A script's interpreter may run it, and a
    framework executable, such as Python's, may run its whole bundle, which it re-launches from.
    """
    executables: list[Path] = []
    trees = [helper for helper in helpers if helper.is_dir()]
    interpreters: list[Path] = []
    for program in programs:
        resolved = program.resolve()
        interpreter = _interpreter(resolved)
        executables.append(resolved)
        if interpreter is not None:
            interpreters.append(interpreter)
        for path in (resolved, interpreter):
            version = None if path is None else _framework_version(path)
            if version is not None:
                trees.append(version)
    return replace(
        grants, executables=tuple(executables), executable_trees=tuple(trees), interpreters=tuple(interpreters)
    )


def _helpers(libexec: Path, tool: str) -> tuple[Path, ...]:
    """Return where an install's ``libexec`` keeps ``tool``'s own helper programs, such as git's ``git-core``.

    The rest of ``libexec`` belongs to the install's other tools.
    """
    return (libexec / tool, libexec / f"{tool}-core")


def _tool_grant(homebrew: Path | None, developer: Path | None, tool: str) -> _ToolGrants | None:
    """Return what ``tool`` needs to run, or ``None`` when it is not installed."""
    found = None if homebrew is None else _homebrew_tool(homebrew, tool)
    if homebrew is not None and found is not None:
        link, keg = found
        kegs, links = _homebrew_formula(homebrew, keg)
        executable = link.resolve()
        # Formulae keep helper programs that are not on PATH in libexec.
        return _runnable(_ToolGrants(kegs, links, (executable.parent,)), (executable,), _helpers(keg / "libexec", tool))
    directories = () if developer is None else _developer_bin_directories(developer)
    programs = tuple(filter(None, (_executable(tool, (directory,)) for directory in directories)))
    if developer is not None and programs:
        # The install keeps helper programs in libexec too, and the tool's shim runs it from the install.
        # An Xcode.app shim would first run xcodebuild, which is not granted, to check the license.
        shim = _executable(tool, (_SHIM_DIRECTORY,)) if _developer_installation(developer) == developer else None
        helpers = tuple(helper for directory in directories for helper in _helpers(directory.parent / "libexec", tool))
        return _runnable(_ToolGrants(developer=developer), programs if shim is None else (*programs, shim), helpers)
    system = _executable(tool, _SYSTEM_BIN_DIRECTORIES)
    # A shim for a tool the selected install lacks can only fail.
    return None if system is None or _is_shim(system) else _runnable(_ToolGrants(), (system,))


def _extend(paths: list[Path], additions: tuple[Path, ...]) -> None:
    paths.extend(path for path in dict.fromkeys(additions) if path not in paths)


def _tool_grants(tools: tuple[str, ...], optional_tools: tuple[str, ...] = ()) -> _ToolGrants:
    """Resolve each tool to the installed code it runs from, and nothing more.

    A Homebrew formula's tool brings that formula and its runtime dependencies. A tool from
    the selected Xcode or Command Line Tools install brings that install, which a Command
    Line Tools ``/usr/bin`` shim also runs. Homebrew wins, as on a default shell PATH, and
    base-system tools need nothing but their own exec grant; a shim alone is not a
    base-system tool. The caller's PATH and environment are never consulted. An optional
    tool that is missing or broken is skipped and grants nothing.
    """
    if not tools and not optional_tools:
        return _ToolGrants()
    homebrew = _homebrew_prefix()
    developer = _developer_directory()
    trees: list[Path] = []
    files: list[Path] = []
    bin_directories: list[Path] = []
    executables: list[Path] = []
    executable_trees: list[Path] = []
    interpreters: list[Path] = []
    used_developer: Path | None = None
    granted: list[str] = []
    for tool in (*tools, *optional_tools):
        try:
            grant = _tool_grant(homebrew, developer, tool)
        except SandboxError:
            if tool in tools:
                raise
            continue
        if grant is None:
            if tool in tools:
                raise SandboxError(f"sandbox tool is not installed: {tool}")
            continue
        granted.append(tool)
        _extend(trees, grant.trees)
        _extend(files, grant.files)
        _extend(bin_directories, grant.bin_directories)
        _extend(executables, grant.executables)
        _extend(executable_trees, grant.executable_trees)
        _extend(interpreters, grant.interpreters)
        used_developer = grant.developer or used_developer
    if used_developer is not None:
        installation = _developer_installation(used_developer)
        trees.append(installation)
        if installation != used_developer:
            files.append(_XCODE_LICENSE)
        bin_directories.extend(_developer_bin_directories(used_developer))
    return _ToolGrants(
        trees=tuple(trees),
        files=tuple(files),
        bin_directories=tuple(bin_directories),
        developer=used_developer,
        executables=tuple(executables),
        executable_trees=tuple(executable_trees),
        interpreters=tuple(interpreters),
        tools=tuple(granted),
    )


# Each command's TMPDIR holds this file, which only that command's policy denies; see _runs_under.
_SCRATCH_SENTINEL = ".localmcp-sentinel"
_SANDBOX_FILTER_NONE = 0
_SANDBOX_FILTER_PATH = 1
_SANDBOX_CHECK_NO_REPORT = 0x40000000
_MAX_TERMINATION_ROUNDS = 50


@lru_cache(maxsize=1)
def _libsystem() -> ctypes.CDLL:
    try:
        library = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        sandbox_check = library.sandbox_check
        list_pids = library.proc_listallpids
    except (OSError, AttributeError) as exc:
        raise SandboxError("macOS process supervision is unavailable") from exc
    # sandbox_check is variadic: only the fixed arguments are declared.
    sandbox_check.restype = ctypes.c_int
    sandbox_check.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    list_pids.restype = ctypes.c_int
    list_pids.argtypes = [ctypes.c_void_p, ctypes.c_int]
    return library


def _process_ids() -> tuple[int, ...]:
    library = _libsystem()
    capacity = max(library.proc_listallpids(None, 0), 0) + 256
    while True:
        buffer = (ctypes.c_int * capacity)()
        count = library.proc_listallpids(buffer, ctypes.sizeof(buffer))
        if count < capacity:
            break
        capacity *= 2
    if count <= 0:
        raise SandboxError("cannot list processes to supervise the sandbox")
    return tuple(pid for pid in buffer[:count] if pid > 0)


def _runs_under(pid: int, scratch: Path) -> bool:
    """Return whether ``pid`` runs under the policy of the command that owns ``scratch``.

    That policy is the only one that can both allow reading the fresh scratch directory and
    deny its sentinel: other sandboxes allow both or neither, and exited processes neither.
    """
    library = _libsystem()
    if library.sandbox_check(pid, None, _SANDBOX_FILTER_NONE) != 1:
        return False

    def check(path: Path) -> int:
        return int(
            library.sandbox_check(
                pid,
                b"file-read-data",
                _SANDBOX_FILTER_PATH | _SANDBOX_CHECK_NO_REPORT,
                ctypes.c_char_p(os.fsencode(path)),
            )
        )

    return check(scratch) == 0 and check(scratch / _SCRATCH_SENTINEL) == 1


def _terminate_command_processes(scratch: Path) -> None:
    """Kill every process still running under a finished command's policy.

    A command can leave its process group (setsid, double fork) but never its sandbox, so
    this finds detached descendants that killing the process group misses.
    """
    for _ in range(_MAX_TERMINATION_ROUNDS):
        remaining = [pid for pid in _process_ids() if pid != os.getpid() and _runs_under(pid, scratch)]
        if not remaining:
            return
        for pid in remaining:
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                continue
    raise SandboxError("sandboxed command processes could not be terminated")


def _owned_process_limit() -> int | None:
    try:
        result = subprocess.run(
            ["/bin/ps", "-axo", "uid="],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    owned = sum(line.strip() == str(os.getuid()) for line in result.stdout.splitlines())
    if owned == 0:
        return None
    return process_limit(owned)


class MacOSSandbox(CommandRuntime):
    """Run a command with no caller environment under immutable Seatbelt authority.

    ``tools`` names the profile's tools that commands can run.
    """

    def __init__(
        self,
        profile: SandboxProfile,
        *,
        timeout_seconds: float = DEFAULT_COMMAND_TIMEOUT_SECONDS,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
        process_label: str = "localmcp-workspace-sandbox",
    ):
        validate_limits(timeout_seconds, max_output_bytes, process_label)
        self.roots = _validated_roots(profile.roots)
        self.root = self.roots[0].path
        self.profile = SandboxProfile(
            self.roots,
            denied_paths=_denied_paths_in_roots(profile.denied_paths, self.roots),
            network=profile.network,
            ipc=profile.ipc,
            tools=tuple(dict.fromkeys(profile.tools)),
            optional_tools=tuple(dict.fromkeys(profile.optional_tools)),
        )
        _check_denied_paths(self.profile.denied_paths)
        self._protected_paths = _protected_roots(self.roots)
        _check_denied_paths(self._protected_paths, "read-only root")
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes
        self.process_label = process_label
        self._sandbox_exec = Path("/usr/bin/sandbox-exec")
        self._shell = _SHELL
        grants = _tool_grants(self.profile.tools, self.profile.optional_tools)
        # A program the sandbox may run, or its code, must not be replaceable from inside it.
        for path in (*grants.trees, *grants.executables, *grants.executable_trees, *grants.interpreters):
            if any(path.is_relative_to(root.path) or root.path.is_relative_to(path) for root in self.roots):
                raise SandboxError(f"sandbox tools overlap a sandbox root: {path}")
        self.tools = grants.tools
        self._developer = grants.developer
        self._toolchain_trees = grants.trees
        self._toolchain_files = grants.files
        self._toolchain_bin_directories = grants.bin_directories
        self._executables = tuple(dict.fromkeys((*_shell_executables(), *grants.executables)))
        self._executable_trees = grants.executable_trees
        self._interpreters = grants.interpreters
        self._tls_files = _tls_files() if self.profile.network else ()
        # Each command gets a fresh temporary directory in here; see _execute.
        self._scratch_parent = Path(tempfile.gettempdir()).resolve()
        _check_scratch(self._scratch_parent, self.roots)
        self._process_limit = _owned_process_limit()
        # Grant metadata reads along every granted path so deep installs (e.g. inside
        # Xcode.app) and symlinked trust stores stay traversable without opening the
        # directories above them. The scratch parent is traversed too, to reach each
        # command's TMPDIR.
        granted = (
            tuple(root.path for root in self.roots) + self._toolchain_trees + self._toolchain_files + self._tls_files
        )
        self._metadata_ancestors = tuple(
            sorted({*_path_ancestors((*granted, self._scratch_parent)), self._scratch_parent}, key=os.fspath)
        )
        # Renaming a denied or write-protected path, or any directory above it inside a
        # root, would move its contents out from under the denial.
        self._denied_ancestors = tuple(
            ancestor
            for ancestor in _path_ancestors((*self.profile.denied_paths, *self._protected_paths))
            if any(ancestor.is_relative_to(root.path) for root in self.roots)
        )

    async def run(self, command: str) -> CommandResult:
        validate_command(command)
        limits = [
            f"ulimit -t {max(1, int(self.timeout_seconds))}",
            f"ulimit -n {MAX_OPEN_FILES}",
        ]
        if self._process_limit is not None:
            limits.append(f"ulimit -u {self._process_limit}")
        return await self._execute(
            [
                os.fspath(self._shell),
                "-c",
                f'{" && ".join(limits)} && exec /bin/sh -c "$1"',
                self.process_label,
                command,
            ],
        )

    async def _execute(self, argv: list[str]) -> CommandResult:
        if not self._sandbox_exec.is_file():
            raise SandboxError("macOS sandbox-exec is unavailable")
        # The host may have linked or replaced a denied path since the sandbox was created.
        await asyncio.to_thread(_check_denied_paths, self.profile.denied_paths)
        await asyncio.to_thread(_check_denied_paths, self._protected_paths, "read-only root")
        # A private TMPDIR per command, removed afterward. libxcrun, behind the /usr/bin
        # shims, caches lookups in it.
        with tempfile.TemporaryDirectory(
            prefix="localmcp-", dir=self._scratch_parent, ignore_cleanup_errors=True
        ) as directory:
            scratch = Path(directory)
            _check_scratch(scratch, self.roots)
            (scratch / _SCRATCH_SENTINEL).touch()
            try:
                return await self._execute_with_scratch(argv, scratch)
            finally:
                # Nothing the command started may outlive the call. This runs synchronously,
                # so cancellation cannot interrupt it.
                _terminate_command_processes(scratch)

    async def _execute_with_scratch(self, argv: list[str], scratch: Path) -> CommandResult:
        runtime_profile = self._compiled_profile()
        runtime_definitions = ["-D", f"SCRATCH={scratch}", "-D", f"SCRATCH_SENTINEL={scratch / _SCRATCH_SENTINEL}"]
        for name, paths in (
            ("TOOLCHAIN_TREE", self._toolchain_trees),
            ("TOOLCHAIN_FILE", self._toolchain_files),
            ("EXECUTABLE", self._executables),
            ("EXECUTABLE_TREE", self._executable_trees),
            ("INTERPRETER", self._interpreters),
            ("TLS_FILE", self._tls_files),
            ("METADATA_ANCESTOR", self._metadata_ancestors),
        ):
            runtime_definitions.extend(
                value for index, path in enumerate(paths) for value in ("-D", f"{name}_{index}={path}")
            )
        root_definitions = [
            value for index, root in enumerate(self.roots) for value in ("-D", f"ROOT_{index}={root.path}")
        ]
        denied_path_definitions = [
            value
            for index, path in enumerate(self.profile.denied_paths)
            for value in ("-D", f"DENIED_PATH_{index}={path}")
        ]
        denied_path_definitions.extend(
            value for index, path in enumerate(self._protected_paths) for value in ("-D", f"PROTECTED_{index}={path}")
        )
        denied_path_definitions.extend(
            value
            for index, path in enumerate(self._denied_ancestors)
            for value in ("-D", f"DENIED_ANCESTOR_{index}={path}")
        )
        process = await asyncio.create_subprocess_exec(
            os.fspath(self._sandbox_exec),
            *root_definitions,
            *denied_path_definitions,
            *runtime_definitions,
            "-p",
            runtime_profile,
            *argv,
            cwd=self.root,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._environment(scratch),
            start_new_session=True,
        )
        stdout = bytearray()
        stderr = bytearray()
        try:
            await asyncio.wait_for(self._read_output(process, stdout, stderr), timeout=self.timeout_seconds)
        except TimeoutError:
            await self._kill(process, scratch)
            return CommandResult(
                exit_code=-1,
                stdout=stdout.decode(errors="replace"),
                stderr=self._append_error(stderr, "command timed out"),
                timed_out=True,
            )
        except _OutputLimitExceeded:
            await self._kill(process, scratch)
            return CommandResult(
                exit_code=-1,
                stdout=stdout.decode(errors="replace"),
                stderr=self._append_error(stderr, "command output exceeded its bounded safety limit"),
                truncated=True,
            )
        except BaseException:
            await self._kill(process, scratch)
            raise
        return CommandResult(
            exit_code=process.returncode or 0,
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
        )

    def _compiled_profile(self) -> str:
        runtime_profile = _BASE_PROFILE
        runtime_profile += "".join(
            f'(allow file-read* (subpath (param "ROOT_{index}")))\n' for index in range(len(self.roots))
        )
        runtime_profile += "".join(
            f'(allow file-write* (subpath (param "ROOT_{index}")))\n'
            for index, root in enumerate(self.roots)
            if root.access == RootAccess.READ_WRITE
        )
        # Nothing bounds what a command writes, so the scratch is writable only when a root
        # already is: a read-only profile must not gain space to fill.
        scratch_access = "file-read*"
        if any(root.access == RootAccess.READ_WRITE for root in self.roots):
            scratch_access += " file-write*"
        runtime_profile += f'(allow {scratch_access} (subpath (param "SCRATCH")))\n'
        if self.profile.network:
            runtime_profile += _NETWORK_PROFILE
            runtime_profile += "".join(
                f'(allow file-read* (literal (param "TLS_FILE_{index}")))\n' for index in range(len(self._tls_files))
            )
        if self.profile.ipc:
            runtime_profile += _SHARED_MEMORY_IPC_PROFILE
        # Toolchain trees hold the binaries and dylibs tools run and load, so they need
        # exec-mapping as well as reads. They are never writable.
        runtime_profile += "".join(
            f'(allow file-read* file-map-executable (subpath (param "TOOLCHAIN_TREE_{index}")))\n'
            for index in range(len(self._toolchain_trees))
        )
        runtime_profile += "".join(
            f'(allow file-read* (literal (param "TOOLCHAIN_FILE_{index}")))\n'
            for index in range(len(self._toolchain_files))
        )
        # Only the shell and the profile's tools run; anything else fails to exec.
        runtime_profile += "".join(
            f'(allow process-exec (literal (param "EXECUTABLE_{index}")))\n' for index in range(len(self._executables))
        )
        runtime_profile += "".join(
            f'(allow process-exec (subpath (param "EXECUTABLE_TREE_{index}")))\n'
            for index in range(len(self._executable_trees))
        )
        runtime_profile += "".join(
            f'(allow process-exec-interpreter (literal (param "INTERPRETER_{index}")))\n'
            for index in range(len(self._interpreters))
        )
        runtime_profile += "".join(
            f'(allow file-read-metadata (literal (param "METADATA_ANCESTOR_{index}")))\n'
            for index in range(len(self._metadata_ancestors))
        )
        runtime_profile += "".join(
            f'(deny file-read* (literal (param "DENIED_PATH_{index}")) (subpath (param "DENIED_PATH_{index}")))\n'
            for index in range(len(self.profile.denied_paths))
        )
        # A read-only root inside a read-write root stays readable but nothing in it can be
        # written, created, renamed, removed, or hard-linked to a new name.
        runtime_profile += "".join(
            f'(deny file-write* (subpath (param "PROTECTED_{index}")))\n' for index in range(len(self._protected_paths))
        )
        runtime_profile += "".join(
            f'(deny file-write-unlink (literal (param "DENIED_PATH_{index}")) '
            f'(subpath (param "DENIED_PATH_{index}")))\n'
            for index in range(len(self.profile.denied_paths))
        )
        runtime_profile += "".join(
            f'(deny file-write-unlink (literal (param "DENIED_ANCESTOR_{index}")))\n'
            for index in range(len(self._denied_ancestors))
        )
        # Last, so no root grant can override the denial that identifies this command's processes.
        runtime_profile += '(deny file-read* file-write* (literal (param "SCRATCH_SENTINEL")))\n'
        return runtime_profile

    def _environment(self, scratch: Path) -> dict[str, str]:
        environment: dict[str, str] = {}
        if self._developer is not None:
            # Pin the selected install and keep shim caches in this invocation's scratch.
            environment["DEVELOPER_DIR"] = os.fspath(self._developer)
            environment["xcrun_db"] = os.fspath(scratch / "xcrun_db")
        return command_environment(scratch, bin_directories=self._toolchain_bin_directories, extra=environment)

    @staticmethod
    async def _kill(process: asyncio.subprocess.Process, scratch: Path) -> None:
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        # Detached descendants can hold the output pipes open; draining would wait for them.
        _terminate_command_processes(scratch)
        await process.communicate()


def sandbox_tools(
    profile: SandboxProfile,
    *,
    budget: ToolCallBudget,
    timeout_seconds: float = DEFAULT_COMMAND_TIMEOUT_SECONDS,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    exhausted_message: str = DEFAULT_EXHAUSTED_MESSAGE,
    description: str | None = None,
    process_label: str = "localmcp-workspace-sandbox",
) -> list[StructuredTool]:
    """Expose Bash under one explicit, immutable Seatbelt profile.

    ``description`` defaults to one naming the profile's tools that commands can run.
    """
    sandbox = MacOSSandbox(
        profile,
        timeout_seconds=timeout_seconds,
        max_output_bytes=max_output_bytes,
        process_label=process_label,
    )
    return _sandbox_tools(
        sandbox,
        granted_tools=sandbox.tools,
        budget=budget,
        exhausted_message=exhausted_message,
        description=description,
    )
