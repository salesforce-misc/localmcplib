"""Portable policy and supervision tests for the Linux backend."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import struct
import subprocess
from pathlib import Path
from typing import Any

import pytest

from localmcp.sandbox import (
    MAX_COMMAND_CHARACTERS,
    CommandResult,
    RootAccess,
    SandboxError,
    SandboxProfile,
    SandboxRoot,
)
from localmcp.sandbox import bubblewrap as linux
from localmcp.sandbox.bubblewrap import LinuxSandbox
from localmcp.sandbox.runtime import _OutputLimitExceeded


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> LinuxSandbox:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setattr(linux, "_require_linux", lambda: None)
    monkeypatch.setattr(linux, "_RUNTIME_TREES", (runtime,))
    monkeypatch.setattr(linux, "_process_limit", lambda: None)
    monkeypatch.setattr(linux, "_tool_grants", lambda _: linux._Tools(("cat",), (), (), (), (), ()))
    return LinuxSandbox(SandboxProfile((SandboxRoot(root, RootAccess.READ_WRITE),), tools=("cat",)))


def test_backend_preserves_first_root_and_nested_protection(backend: LinuxSandbox) -> None:
    protected = backend.root / "docs" / "source"
    protected.mkdir(parents=True)
    (protected / "inner").mkdir()
    sandbox = LinuxSandbox(SandboxProfile((SandboxRoot(backend.root, RootAccess.READ_WRITE), SandboxRoot(protected))))

    assert sandbox.root == backend.root
    assert sandbox._protected_paths == (protected,)
    with pytest.raises(SandboxError, match="write-protected"):
        LinuxSandbox(
            SandboxProfile(
                (
                    SandboxRoot(backend.root, RootAccess.READ_WRITE),
                    SandboxRoot(protected),
                    SandboxRoot(protected / "inner", RootAccess.READ_WRITE),
                )
            )
        )


def test_denied_paths_require_existing_in_root_targets(backend: LinuxSandbox, tmp_path: Path) -> None:
    missing = backend.root / "missing"
    with pytest.raises(SandboxError, match="must exist"):
        LinuxSandbox(SandboxProfile(backend.roots, denied_paths=(missing,)))
    outside = tmp_path / "outside"
    outside.touch()
    with pytest.raises(SandboxError, match="outside"):
        LinuxSandbox(SandboxProfile(backend.roots, denied_paths=(outside,)))


def test_denied_symlinks_and_hard_links_fail_closed(backend: LinuxSandbox) -> None:
    secret = backend.root / "secret"
    secret.write_text("private")
    alias = backend.root / "alias"
    alias.symlink_to(secret)
    with pytest.raises(SandboxError, match="symlink"):
        LinuxSandbox(SandboxProfile(backend.roots, denied_paths=(alias,)))
    alias.unlink()
    os.link(secret, alias)
    with pytest.raises(SandboxError, match="hard link"):
        LinuxSandbox(SandboxProfile(backend.roots, denied_paths=(secret,)))


def test_tools_and_runtime_cannot_overlap_roots(backend: LinuxSandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        linux, "_tool_grants", lambda _: linux._Tools(("cat",), (backend.root / "cat",), (), (), (), ())
    )
    with pytest.raises(SandboxError, match="overlap"):
        LinuxSandbox(SandboxProfile(backend.roots))


def test_mount_order_keeps_narrower_protection_and_denials(backend: LinuxSandbox, tmp_path: Path) -> None:
    ancestor = backend.root / "docs"
    protected = ancestor / "source"
    protected.mkdir(parents=True)
    secret = protected / "secret"
    secret.touch()
    sandbox = LinuxSandbox(
        SandboxProfile(
            (SandboxRoot(backend.root, RootAccess.READ_WRITE), SandboxRoot(protected)), denied_paths=(secret,)
        )
    )
    argv, profile = sandbox._invocation("printf ok", tmp_path / "scratch", tmp_path / "control")

    root_bind = argv.index(str(backend.root))
    ancestor_bind = argv.index(str(ancestor))
    protected_bind = argv.index(str(protected))
    denied_bind = argv.index(str(secret))
    assert root_bind < ancestor_bind < protected_bind < denied_bind
    assert str(protected) in profile["readonly_mounts"]
    assert "--unshare-net" in argv
    assert "--unshare-user" in argv
    assert "--unshare-pid" in argv
    assert "--as-pid-1" in argv
    assert profile["command"] == "printf ok"
    assert str(backend.root) in profile["write_paths"]


def test_read_only_profile_does_not_gain_writable_scratch(backend: LinuxSandbox, tmp_path: Path) -> None:
    sandbox = LinuxSandbox(SandboxProfile((SandboxRoot(backend.root),), network=True, ipc=True, tools=()))
    scratch = tmp_path / "scratch"
    argv, profile = sandbox._invocation("true", scratch, tmp_path / "control")

    assert "--unshare-net" not in argv
    assert profile["ipc"] is True
    assert str(scratch) in profile["readonly_mounts"]
    assert str(scratch) not in profile["write_paths"]
    assert "/dev/shm" in profile["write_paths"]


@pytest.mark.parametrize("command", [" \t", "bad\0command", "x" * (MAX_COMMAND_CHARACTERS + 1)])
async def test_run_rejects_invalid_commands_before_launch(backend: LinuxSandbox, command: str) -> None:
    with pytest.raises(SandboxError):
        await backend.run(command)


async def test_missing_runtime_fails_closed(backend: LinuxSandbox) -> None:
    backend._bwrap = backend.root / "missing-bwrap"
    with pytest.raises(SandboxError, match="requires"):
        await backend.run("true")


@pytest.fixture
def tool_system(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    system = tmp_path / "system"
    system.mkdir()
    monkeypatch.setattr(linux, "_BIN_DIRECTORIES", (system,))
    monkeypatch.setattr(linux, "_trusted_program", lambda path: path.resolve(strict=True))
    monkeypatch.setattr(linux, "_libraries", lambda _: ())
    monkeypatch.setattr(linux, "_elf_interpreter", lambda _: None)
    return system


def test_tool_discovery_ignores_caller_path_and_skips_missing_optional(
    tool_system: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tool_system / "cat").touch(mode=0o755)
    decoy = tmp_path / "decoy"
    decoy.mkdir()
    (decoy / "missing").touch(mode=0o755)
    monkeypatch.setenv("PATH", str(decoy))

    grants = linux._tool_grants(SandboxProfile((SandboxRoot(tmp_path),), tools=("cat",), optional_tools=("missing",)))

    assert grants.names == ("cat",)
    assert (tool_system / "cat") in grants.executables
    assert (decoy / "missing") not in grants.executables
    with pytest.raises(SandboxError, match="not installed: missing"):
        linux._tool_grants(SandboxProfile((SandboxRoot(tmp_path),), tools=("missing",)))


def test_optional_tool_helper_failure_grants_no_partial_authority(
    tool_system: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    good, broken = tool_system / "good", tool_system / "broken"
    for program in (good, broken):
        program.touch(mode=0o755)
    helpers = tmp_path / "helpers"
    helpers.mkdir()
    helper = helpers / "helper"
    helper.touch(mode=0o755)
    good_library, broken_library = tmp_path / "good.so", tmp_path / "broken.so"
    monkeypatch.setattr(linux, "_helpers", lambda name: (helpers,) if name == "broken" else ())

    def libraries(program: Path) -> tuple[Path, ...]:
        if program == helper:
            raise SandboxError("broken helper dependency")
        if program == good:
            return (good_library,)
        if program == broken:
            return (broken_library,)
        return ()

    monkeypatch.setattr(linux, "_libraries", libraries)
    grants = linux._tool_grants(SandboxProfile((SandboxRoot(tmp_path),), tools=("good",), optional_tools=("broken",)))
    assert grants.names == ("good",)
    assert grants.executables == (Path("/bin/sh").resolve(), good)
    assert grants.helpers == ()
    assert grants.maps == (good_library,)
    assert grants.data == ()
    assert grants.aliases == ()
    with pytest.raises(SandboxError, match="broken helper dependency"):
        linux._tool_grants(SandboxProfile((SandboxRoot(tmp_path),), tools=("broken",)))


def test_script_tools_need_explicit_interpreter(tool_system: Path, tmp_path: Path) -> None:
    interpreter = tool_system / "interpreter"
    interpreter.touch(mode=0o755)
    script = tool_system / "script"
    script.write_text(f"#!{interpreter}\n")
    script.chmod(0o755)
    with pytest.raises(SandboxError, match="interpreter.*explicitly listed"):
        linux._tool_grants(SandboxProfile((SandboxRoot(tmp_path),), tools=("script",)))
    grants = linux._tool_grants(SandboxProfile((SandboxRoot(tmp_path),), tools=("script", "interpreter")))
    assert grants.names == ("script", "interpreter")
    optional = linux._tool_grants(SandboxProfile((SandboxRoot(tmp_path),), tools=(), optional_tools=("script",)))
    assert optional.names == ()
    assert script not in optional.executables


def test_multicall_aliases_are_rejected(tool_system: Path, tmp_path: Path) -> None:
    (tool_system / "busybox").touch(mode=0o755)
    (tool_system / "cat").symlink_to(tool_system / "busybox")
    with pytest.raises(SandboxError, match="multicall"):
        linux._tool_grants(SandboxProfile((SandboxRoot(tmp_path),), tools=("cat",)))


def test_multicall_hard_links_are_rejected_or_skipped_when_optional(tool_system: Path, tmp_path: Path) -> None:
    provider = tool_system / "busybox"
    provider.touch(mode=0o755)
    alias = tool_system / "cat"
    os.link(provider, alias)
    assert alias.resolve().name == "cat"

    with pytest.raises(SandboxError, match="multicall"):
        linux._tool_grants(SandboxProfile((SandboxRoot(tmp_path),), tools=("cat",)))
    optional = linux._tool_grants(SandboxProfile((SandboxRoot(tmp_path),), tools=(), optional_tools=("cat",)))
    assert optional.names == ()
    assert alias not in optional.executables
    assert provider not in optional.executables


def test_tool_code_outside_runtime_is_not_trusted(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    untrusted = tmp_path / "workload"
    untrusted.touch(mode=0o755)
    (runtime / "tool").symlink_to(untrusted)
    monkeypatch.setattr(linux, "_RUNTIME_TREES", (runtime,))
    with pytest.raises(SandboxError, match="outside the system runtime"):
        linux._trusted_program(runtime / "tool")


@pytest.mark.parametrize(("wide", "endian"), [(False, "<"), (False, ">"), (True, "<"), (True, ">")])
def test_elf_interpreter_is_read_from_program_headers(tmp_path: Path, wide: bool, endian: str) -> None:
    interpreter = tmp_path / "loader"
    interpreter.touch()
    value = os.fsencode(interpreter) + b"\0"
    header = bytearray(64)
    header[:6] = b"\x7fELF" + bytes((2 if wide else 1, 1 if endian == "<" else 2))
    struct.pack_into(endian + ("Q" if wide else "I"), header, 32 if wide else 28, 64)
    size = 56 if wide else 32
    struct.pack_into(endian + "HH", header, 54 if wide else 42, size, 1)
    entry = bytearray(size)
    struct.pack_into(endian + "I", entry, 0, 3)
    struct.pack_into(endian + ("Q" if wide else "I"), entry, 8 if wide else 4, 64 + size)
    struct.pack_into(endian + ("Q" if wide else "I"), entry, 32 if wide else 16, len(value))
    program = tmp_path / "program"
    program.write_bytes(header + entry + value)

    assert linux._elf_interpreter(program) == interpreter


@pytest.mark.parametrize("content", [b"\x7fELF", b"\x7fELF\x03\x01" + bytes(58)])
def test_invalid_elf_headers_fail_closed(tmp_path: Path, content: bytes) -> None:
    program = tmp_path / "program"
    program.write_bytes(content)
    with pytest.raises(SandboxError, match="unsupported ELF"):
        linux._elf_interpreter(program)


def test_library_discovery_uses_clean_environment_and_resolves_dependencies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    library = tmp_path / "libc.so"
    loader = tmp_path / "loader.so"
    for path in (library, loader):
        path.touch()
    monkeypatch.setattr(linux, "_trusted_program", lambda path: path)
    monkeypatch.setattr(linux, "_elf_interpreter", lambda _: loader)
    captured: dict[str, Any] = {}

    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured.update(kwargs)
        return subprocess.CompletedProcess(argv, 0, f"libc => {library} (0x1)\n{loader} (0x2)\n", "")

    monkeypatch.setattr(subprocess, "run", run)
    assert linux._libraries(tmp_path / "program") == (library, loader)
    assert captured["env"] == {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}
    assert captured["timeout"] == 5


@pytest.mark.parametrize(("returncode", "output"), [(1, ""), (0, "libc => not found"), (0, "unexpected output")])
def test_unresolved_libraries_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, returncode: int, output: str
) -> None:
    monkeypatch.setattr(linux, "_trusted_program", lambda path: path)
    monkeypatch.setattr(linux, "_elf_interpreter", lambda path: path)
    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: subprocess.CompletedProcess(args, returncode, output)
    )
    with pytest.raises(SandboxError, match="cannot resolve"):
        linux._libraries(tmp_path / "program")


class _Process:
    pid = 12345
    returncode: int | None = 0

    def __init__(self) -> None:
        self.communicated = False

    async def communicate(self) -> tuple[bytes, bytes]:
        self.communicated = True
        return b"", b""


def _fake_launch(
    backend: LinuxSandbox, monkeypatch: pytest.MonkeyPatch, *, status: dict[str, object] | None
) -> tuple[_Process, dict[str, Any]]:
    binary = backend.root / "fake-runtime"
    binary.touch()
    backend._bwrap = binary
    backend._python = binary
    process = _Process()
    captured: dict[str, Any] = {}

    async def create(*argv: str, **kwargs: Any) -> _Process:
        captured["argv"] = argv
        captured.update(kwargs)
        status_path = Path(argv[argv.index("/localmcp/status") - 1])
        if status is not None:
            status_path.write_text(json.dumps(status))
        return process

    async def read(_process: Any, stdout: bytearray, stderr: bytearray) -> None:
        stdout.extend(b"out\xff")
        stderr.extend(b"warning")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    monkeypatch.setattr(backend, "_read_output", read)
    return process, captured


async def test_success_uses_supervised_exit_and_removes_scratch(
    backend: LinuxSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, captured = _fake_launch(backend, monkeypatch, status={"exit_code": 7})
    result = await backend.run("exit 7")

    assert result == CommandResult(exit_code=7, stdout="out\ufffd", stderr="warning")
    assert captured["env"]["HOME"] == "/var/empty"
    assert captured["stdin"] == asyncio.subprocess.DEVNULL
    assert not Path(captured["env"]["TMPDIR"]).exists()
    assert not Path(captured["argv"][captured["argv"].index("/localmcp/status") - 1]).exists()


async def test_invocation_uses_trusted_launcher_snapshot_after_source_changes(
    backend: LinuxSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_launch(backend, monkeypatch, status={"exit_code": 0})
    launch = asyncio.create_subprocess_exec
    read_bytes = Path.read_bytes
    source_reads: list[Path] = []
    launched_sources: list[bytes] = []

    def changed_source(path: Path) -> bytes:
        if path.name == "_linux_runner.py":
            source_reads.append(path)
            raise AssertionError("workload replaced the editable launcher source")
        return read_bytes(path)

    async def create(*argv: str, **kwargs: Any) -> Any:
        runner_path = Path(argv[argv.index("/localmcp/runner.py") - 1])
        launched_sources.append(runner_path.read_bytes())
        return await launch(*argv, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", changed_source)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)

    result = await backend.run("true")

    assert result.exit_code == 0
    assert launched_sources == [linux._RUNNER_SOURCE]
    assert source_reads == []


@pytest.mark.parametrize("status", [None, {"error": "Landlock unavailable"}, {"exit_code": "wrong-type"}])
async def test_setup_failures_are_not_workload_exit_codes(
    backend: LinuxSandbox, monkeypatch: pytest.MonkeyPatch, status: dict[str, object] | None
) -> None:
    _fake_launch(backend, monkeypatch, status=status)
    with pytest.raises(SandboxError, match="setup"):
        await backend.run("true")


@pytest.mark.parametrize("failure", [TimeoutError(), _OutputLimitExceeded(), asyncio.CancelledError()])
async def test_limits_and_cancellation_terminate_before_cleanup(
    backend: LinuxSandbox, monkeypatch: pytest.MonkeyPatch, failure: BaseException
) -> None:
    process, captured = _fake_launch(backend, monkeypatch, status=None)
    killed: list[_Process] = []

    async def read(_process: Any, stdout: bytearray, stderr: bytearray) -> None:
        stdout.extend(b"partial")
        raise failure

    async def kill(target: _Process, info: Path) -> None:
        assert info.parent.is_dir()
        killed.append(target)

    monkeypatch.setattr(backend, "_read_output", read)
    monkeypatch.setattr(backend, "_kill", kill)
    if isinstance(failure, asyncio.CancelledError):
        with pytest.raises(asyncio.CancelledError):
            await backend.run("true")
    else:
        result = await backend.run("true")
        assert result.exit_code == -1
        assert result.stdout == "partial"
        assert result.timed_out is isinstance(failure, TimeoutError)
        assert result.truncated is isinstance(failure, _OutputLimitExceeded)
    assert killed == [process]
    assert not Path(captured["env"]["TMPDIR"]).exists()


async def test_kill_targets_namespace_init_and_waits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    process = _Process()
    process.returncode = None
    info = tmp_path / "info"
    info.write_text('{"child-pid": 56789}')
    signals: list[tuple[str, int, int]] = []
    monkeypatch.setattr(os, "kill", lambda pid, value: signals.append(("pid", pid, value)))
    monkeypatch.setattr(os, "killpg", lambda pid, value: signals.append(("group", pid, value)))

    await LinuxSandbox._kill(process, info)  # type: ignore[arg-type]

    assert signals == [("pid", 56789, signal.SIGKILL), ("group", process.pid, signal.SIGKILL)]
    assert process.communicated
