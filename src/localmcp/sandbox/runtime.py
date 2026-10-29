"""Portable command validation, environment, and bounded output collection."""

from __future__ import annotations

import asyncio
import os
import resource
import sys
from pathlib import Path

from localmcp.sandbox.base import (
    MAX_ADDITIONAL_PROCESSES,
    MAX_COMMAND_CHARACTERS,
    MAX_OPEN_FILES,
    RootAccess,
    SandboxError,
    SandboxRoot,
)

# Capture the host interpreter's library locations before any command can modify
# a writable checkout. The fresh host launcher must never import from a root.
_HOST_RUNTIME_TREES = tuple(dict.fromkeys(Path(path).resolve() for path in (sys.base_prefix, sys.base_exec_prefix)))


def check_host_runtime(roots: tuple[SandboxRoot, ...]) -> None:
    """Keep the host launcher's standard library outside writable sandbox roots."""
    for root in roots:
        if root.access == RootAccess.READ_WRITE and any(
            root.path.is_relative_to(tree) or tree.is_relative_to(root.path) for tree in _HOST_RUNTIME_TREES
        ):
            raise SandboxError(f"writable sandbox root overlaps the host Python runtime: {root.path}")


def validate_limits(timeout_seconds: float, max_output_bytes: int, process_label: str) -> None:
    if timeout_seconds <= 0:
        raise SandboxError("sandbox timeout must be positive")
    if max_output_bytes <= 0:
        raise SandboxError("sandbox output limit must be positive")
    if not process_label or "\0" in process_label:
        raise SandboxError("sandbox process label must not be blank or contain NUL bytes")


def validate_command(command: str) -> None:
    if not command.strip():
        raise SandboxError("command must not be blank")
    if "\0" in command or len(command) > MAX_COMMAND_CHARACTERS:
        raise SandboxError(f"command must contain at most {MAX_COMMAND_CHARACTERS} characters and no NUL bytes")


def process_limit(existing: int) -> int | None:
    """Return an RLIMIT_NPROC allowing ``MAX_ADDITIONAL_PROCESSES`` beyond ``existing``.

    ``existing`` is what the kernel already counts against the limit. The result
    never exceeds the caller's soft limit, and is ``None`` when that leaves no headroom.
    """
    soft_limit, _ = resource.getrlimit(resource.RLIMIT_NPROC)
    proposed = existing + MAX_ADDITIONAL_PROCESSES
    if soft_limit != resource.RLIM_INFINITY:
        proposed = min(proposed, soft_limit)
    return proposed if proposed > existing else None


def owned_process_limit() -> int | None:
    """Count the caller's Linux tasks on the host before entering a user namespace.

    RLIMIT_NPROC counts threads too. The host allowance must leave room to exec
    bubblewrap before it creates the command's namespaces. This is an inherited
    per-user limit, not a quota dedicated to the sandbox.
    """
    owned = 0
    uid = str(os.getuid())
    try:
        for entry in Path("/proc").iterdir():
            if not entry.name.isdecimal():
                continue
            try:
                fields = dict(line.split(":", 1) for line in (entry / "status").read_text().splitlines() if ":" in line)
            except (FileNotFoundError, ProcessLookupError):
                continue
            if fields.get("Uid", "").split()[:1] == [uid]:
                owned += int(fields["Threads"])
    except (OSError, ValueError, KeyError):
        raise SandboxError("cannot count host tasks for the sandbox process limit") from None
    if not owned:
        raise SandboxError("cannot count host tasks for the sandbox process limit")
    return process_limit(owned)


_LIMIT_LAUNCHER = """\
import os, resource, signal, sys
for name in ("SIGPIPE", "SIGXFSZ"):
    if hasattr(signal, name):
        signal.signal(getattr(signal, name), signal.SIG_DFL)
offset = 2
for _ in range(int(sys.argv[1])):
    maximum = int(sys.argv[offset + 1])
    resource.setrlimit(int(sys.argv[offset]), (maximum, maximum))
    offset += 2
os.execv(sys.argv[offset], sys.argv[offset:])
"""


def limited_command(argv: list[str], timeout_seconds: float, maximum_processes: int | None) -> list[str]:
    """Set host child limits in a fresh interpreter, then exec the backend.

    No Python callback runs between fork and exec in the multithreaded parent.
    The isolated launcher runs entirely outside the sandbox, uses snapshotted
    source rather than a workload-writable module, and imports no site packages.
    """
    requested = [(resource.RLIMIT_CPU, max(1, int(timeout_seconds))), (resource.RLIMIT_NOFILE, MAX_OPEN_FILES)]
    if maximum_processes is not None:
        requested.append((resource.RLIMIT_NPROC, maximum_processes))
    limits: list[str] = []
    for kind, maximum in requested:
        _, hard = resource.getrlimit(kind)
        limits.extend((str(kind), str(maximum if hard == resource.RLIM_INFINITY else min(maximum, hard))))
    # Linux resolves this in the forked child to the running interpreter inode,
    # even if a workload replaced the original venv executable alias.
    interpreter = "/proc/self/exe" if sys.platform == "linux" else sys.executable
    return [interpreter, "-I", "-S", "-c", _LIMIT_LAUNCHER, str(len(requested)), *limits, *argv]


def command_exit_code(returncode: int) -> int:
    """Use portable shell statuses: explicit exit codes or 128 + signal."""
    return 128 - returncode if returncode < 0 else returncode


def command_environment(
    scratch: Path,
    *,
    bin_directories: tuple[Path, ...] = (),
    extra: dict[str, str] | None = None,
    developer_directory: Path | None = None,
) -> dict[str, str]:
    """Build a fresh environment without consulting the caller's environment."""
    paths = [*map(os.fspath, bin_directories), "/usr/bin", "/bin", "/usr/sbin", "/sbin"]
    environment = (extra or {}) | {
        "HOME": "/var/empty",
        "PATH": os.pathsep.join(paths),
        "LC_ALL": "C",
        "RIPGREP_CONFIG_PATH": "",
        "TMPDIR": os.fspath(scratch),
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        # Also suppress bundled Git configuration, attributes, locks, and prompts.
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
    }
    if developer_directory is not None:
        # Pin Apple's selected toolchain and keep shim caches in private scratch.
        environment["DEVELOPER_DIR"] = os.fspath(developer_directory)
        environment["xcrun_db"] = os.fspath(scratch / "xcrun_db")
    return environment


class _OutputLimitExceeded(Exception):
    pass


class CommandRuntime:
    """Collect command output; each backend owns process creation and termination."""

    max_output_bytes: int

    async def _read_output(
        self,
        process: asyncio.subprocess.Process,
        stdout: bytearray,
        stderr: bytearray,
    ) -> None:
        total = 0
        lock = asyncio.Lock()

        async def read(stream: asyncio.StreamReader | None, destination: bytearray) -> None:
            nonlocal total
            if stream is None:
                return
            while chunk := await stream.read(64 * 1024):
                async with lock:
                    remaining = self.max_output_bytes - total
                    if remaining <= 0:
                        raise _OutputLimitExceeded
                    destination.extend(chunk[:remaining])
                    total += min(len(chunk), remaining)
                    if len(chunk) > remaining:
                        raise _OutputLimitExceeded

        readers = [asyncio.create_task(read(process.stdout, stdout)), asyncio.create_task(read(process.stderr, stderr))]
        try:
            await asyncio.gather(*readers)
            await process.wait()
        except BaseException:
            for task in readers:
                task.cancel()
            await asyncio.gather(*readers, return_exceptions=True)
            raise

    @staticmethod
    def _append_error(stderr: bytearray, message: str) -> str:
        rendered = stderr.decode(errors="replace").rstrip()
        return f"{rendered}\n{message}".lstrip()
