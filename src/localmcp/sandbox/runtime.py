"""Portable command validation, environment, and bounded output collection."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from localmcp.sandbox.base import MAX_COMMAND_CHARACTERS, SandboxError


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


def command_environment(
    scratch: Path,
    *,
    bin_directories: tuple[Path, ...] = (),
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build a fresh environment without consulting the caller's environment."""
    paths = [*map(os.fspath, bin_directories), "/usr/bin", "/bin", "/usr/sbin", "/sbin"]
    return (extra or {}) | {
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
