"""Portable contracts for policy-bound command sandboxes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

DEFAULT_COMMAND_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_OUTPUT_BYTES = 4 * 1024 * 1024
MAX_COMMAND_CHARACTERS = 16_000
MAX_OPEN_FILES = 256
MAX_ADDITIONAL_PROCESSES = 256

# Commands for reading, searching and comparing files: the default sandbox tools.
INSPECTION_TOOLS = (
    "awk",
    "basename",
    "cat",
    "cmp",
    "comm",
    "cut",
    "diff",
    "dirname",
    "du",
    "file",
    "find",
    "grep",
    "head",
    "ls",
    "nl",
    "od",
    "paste",
    "readlink",
    "realpath",
    "sed",
    "sort",
    "stat",
    "tail",
    "tr",
    "uniq",
    "wc",
    "xargs",
)
# Commands for managing files, for sandboxes with read-write roots. Root access, not this
# list, decides what commands can write: the shell and ``sed -i`` can write without them.
WRITE_TOOLS = ("chmod", "cp", "ln", "mkdir", "mv", "patch", "rm", "rmdir", "tee", "touch")


class SandboxError(RuntimeError):
    """Raised when a sandbox cannot be constructed or invoked safely."""


class RootAccess(StrEnum):
    """Filesystem authority granted to one sandbox root."""

    READ_ONLY = "read_only"
    READ_WRITE = "read_write"


@dataclass(frozen=True)
class SandboxRoot:
    """One filesystem root exposed to a sandbox."""

    path: Path
    access: RootAccess = RootAccess.READ_ONLY


@dataclass(frozen=True)
class SandboxProfile:
    """Portable authority requested for a sandboxed command.

    The first root is the process working directory. A read-only root inside
    a read-write root is write-protected there: it stays readable, but nothing
    in it can be written, created, renamed, removed, or linked to a new name.
    ``denied_paths`` are hidden from commands; each must lie inside a root but
    need not exist; on Linux a newly created path is hidden from the next
    command onward. ``ipc`` permits
    shared-memory IPC only, not sockets or platform service protocols.
    ``tools`` lists the literal command names accepted by the shared Bash
    check, besides a small set of supported builtins, and defaults to
    ``INSPECTION_TOOLS``. Required tools must be installed. ``optional_tools``
    adds installed commands without failing for missing ones. This check
    includes nested command substitutions but does not inspect programs
    interpreted or launched by a granted tool. macOS additionally restricts
    executable access; Linux exposes the system and Nix runtime read-only
    and permits granted tools to launch other programs within the sandbox.
    """

    roots: tuple[SandboxRoot, ...]
    denied_paths: tuple[Path, ...] = ()
    network: bool = False
    ipc: bool = False
    tools: tuple[str, ...] = INSPECTION_TOOLS
    optional_tools: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.roots:
            raise SandboxError("sandbox profile must declare at least one root")
        for kind, tools in (("tools", self.tools), ("optional tools", self.optional_tools)):
            if isinstance(tools, str):
                raise SandboxError(f"sandbox {kind} must be a sequence of command names")
            for tool in tools:
                if tool in {"", ".", ".."} or "/" in tool or "\0" in tool:
                    raise SandboxError(f"sandbox tool must be a command name: {tool!r}")
        if both := sorted(set(self.tools) & set(self.optional_tools)):
            raise SandboxError(f"sandbox tools cannot be both required and optional: {', '.join(both)}")


class CommandResult(BaseModel):
    """Captured result: a shell exit status (128 + signal), or -1 when bounded.

    ``timed_out`` and ``truncated`` distinguish the two bounded-abort cases.
    """

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False
    truncated: bool = False


class Sandbox(Protocol):
    """Narrow async interface implemented by platform sandbox backends."""

    async def run(self, command: str) -> CommandResult: ...
