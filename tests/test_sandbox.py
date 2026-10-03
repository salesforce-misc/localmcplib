import asyncio
import ctypes
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pytest

from localmcp.sandbox import (
    INSPECTION_TOOLS,
    MAX_COMMAND_CHARACTERS,
    WRITE_TOOLS,
    CommandResult,
    RootAccess,
    SandboxError,
    SandboxProfile,
    SandboxRoot,
    seatbelt,
)
from localmcp.sandbox.seatbelt import MacOSSandbox, _developer_directory, _homebrew_prefix


@pytest.fixture(autouse=True)
def system_bin(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Serve the default tools from a fake base system, outside every test's roots.

    No toolchain is installed and no /usr/bin shim exists, so no test depends on the host.
    """
    system = tmp_path_factory.mktemp("system-bin").resolve()
    for tool in (*INSPECTION_TOOLS, *WRITE_TOOLS):
        (system / tool).touch(mode=0o755)
    monkeypatch.setattr(seatbelt, "_SYSTEM_BIN_DIRECTORIES", (system,))
    monkeypatch.setattr(seatbelt, "_SHIM_DIRECTORY", tmp_path_factory.mktemp("shims").resolve())
    monkeypatch.setattr(seatbelt, "_SHELL_SELECTION", tmp_path_factory.mktemp("select") / "sh")
    monkeypatch.setattr(seatbelt, "_homebrew_prefix", lambda: None)
    monkeypatch.setattr(seatbelt, "_developer_directory", lambda: None)
    return system


_SHELL = (Path("/bin/sh"), Path("/bin/bash"))
# Root reads and lists files regardless of their permissions.
_unprivileged = pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")


class _Process:
    def __init__(self, *, returncode: int | None = None) -> None:
        self.returncode = returncode
        self.pid = 1234
        self.stdout: asyncio.StreamReader | None = None
        self.stderr: asyncio.StreamReader | None = None
        self.communicated = False
        self.waited = False

    async def communicate(self) -> tuple[bytes, bytes]:
        self.communicated = True
        return b"", b""

    async def wait(self) -> int:
        self.waited = True
        return self.returncode or 0


class _Stream:
    def __init__(self, *chunks: bytes) -> None:
        self._chunks = iter((*chunks, b""))

    async def read(self, _size: int) -> bytes:
        return next(self._chunks)


class _Budget:
    def __init__(self, *claims: tuple[bool, str | None]) -> None:
        self._claims = iter(claims)

    async def claim(self) -> tuple[bool, str | None]:
        return next(self._claims)

    def metadata(self, exhausted_scope: str | None = None) -> dict[str, object]:
        return {"scope": exhausted_scope or "available"}


def test_profile_rejects_read_only_root_nested_in_writable_root(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()

    with pytest.raises(SandboxError, match="cannot be enforced"):
        MacOSSandbox(
            SandboxProfile(
                (
                    SandboxRoot(tmp_path, RootAccess.READ_WRITE),
                    SandboxRoot(nested, RootAccess.READ_ONLY),
                )
            )
        )


def test_profile_normalizes_and_deduplicates_nested_roots(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    root_alias = tmp_path / "root-alias"
    root_alias.symlink_to(tmp_path, target_is_directory=True)

    sandbox = MacOSSandbox(
        SandboxProfile(
            (
                SandboxRoot(root_alias),
                SandboxRoot(tmp_path),
                SandboxRoot(nested),
            )
        )
    )

    assert sandbox.roots == (SandboxRoot(tmp_path.resolve()),)
    assert sandbox.root == tmp_path.resolve()


def test_denied_path_must_be_inside_a_root(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside"

    with pytest.raises(SandboxError, match="outside"):
        MacOSSandbox(SandboxProfile((SandboxRoot(tmp_path),), denied_paths=(outside,)))


def test_compiled_profile_emits_denies_after_every_allow(tmp_path: Path) -> None:
    first_denied = tmp_path / "credentials"
    second_denied = tmp_path / "private" / "token"
    sandbox = MacOSSandbox(
        SandboxProfile(
            (SandboxRoot(tmp_path, RootAccess.READ_WRITE),),
            denied_paths=(first_denied, second_denied),
            network=True,
            ipc=True,
        )
    )

    profile_lines = sandbox._compiled_profile().splitlines()
    allow_positions = [index for index, line in enumerate(profile_lines) if line.startswith("(allow ")]
    denied_positions = [index for index, line in enumerate(profile_lines) if line.startswith("(deny file-read*")]

    assert allow_positions
    assert len(denied_positions) == 3
    assert min(denied_positions) > max(allow_positions)
    # The sentinel deny identifies this command's processes, so no later rule may re-allow it.
    assert profile_lines[-1] == '(deny file-read* file-write* (literal (param "SCRATCH_SENTINEL")))'


def test_scratch_is_writable_only_when_a_root_is(tmp_path: Path) -> None:
    read_only = MacOSSandbox(SandboxProfile((SandboxRoot(tmp_path),)))._compiled_profile()
    read_write = MacOSSandbox(SandboxProfile((SandboxRoot(tmp_path, RootAccess.READ_WRITE),)))._compiled_profile()

    assert '(allow file-read* (subpath (param "SCRATCH")))' in read_only.splitlines()
    assert '(allow file-read* file-write* (subpath (param "SCRATCH")))' in read_write.splitlines()
    assert not any(line.startswith("(allow ") and "file-write" in line for line in read_only.splitlines())


def test_profile_keeps_credentials_out_of_environment(tmp_path: Path) -> None:
    sandbox = MacOSSandbox(SandboxProfile((SandboxRoot(tmp_path),)))

    environment = sandbox._environment(tmp_path)
    assert environment["HOME"] == "/var/empty"
    assert "SSH_AUTH_SOCK" not in environment
    assert "ANTHROPIC_API_KEY" not in environment


def test_profile_requires_a_root() -> None:
    with pytest.raises(SandboxError, match="at least one root"):
        SandboxProfile(())


def test_profile_rejects_missing_file_and_conflicting_roots(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(SandboxError, match="does not exist"):
        MacOSSandbox(SandboxProfile((SandboxRoot(missing),)))

    regular_file = tmp_path / "file"
    regular_file.write_text("data")
    with pytest.raises(SandboxError, match="not a directory"):
        MacOSSandbox(SandboxProfile((SandboxRoot(regular_file),)))

    with pytest.raises(SandboxError, match="conflicting access grants"):
        MacOSSandbox(
            SandboxProfile(
                (
                    SandboxRoot(tmp_path, RootAccess.READ_ONLY),
                    SandboxRoot(tmp_path, RootAccess.READ_WRITE),
                )
            )
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"timeout_seconds": 0}, "timeout must be positive"),
        ({"max_output_bytes": 0}, "output limit must be positive"),
        ({"process_label": ""}, "process label"),
        ({"process_label": "bad\0label"}, "process label"),
    ],
)
def test_constructor_rejects_unsafe_limits_and_labels(tmp_path: Path, kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(SandboxError, match=message):
        MacOSSandbox(SandboxProfile((SandboxRoot(tmp_path),)), **kwargs)


@pytest.mark.parametrize("denied", ["credentials", "private"])
def test_denied_files_with_other_hard_links_fail_closed(tmp_path: Path, denied: str) -> None:
    (tmp_path / "private" / "nested").mkdir(parents=True)
    (tmp_path / "credentials").write_text("secret")
    (tmp_path / "private" / "nested" / "key").write_text("secret")
    (tmp_path / "private" / "link").symlink_to(tmp_path / "credentials")
    profile = SandboxProfile((SandboxRoot(tmp_path),), denied_paths=(tmp_path / denied, tmp_path / "missing"))
    MacOSSandbox(profile)

    linked = tmp_path / "credentials" if denied == "credentials" else tmp_path / "private" / "nested" / "key"
    os.link(linked, tmp_path / "alias")

    with pytest.raises(SandboxError, match=re.escape(f"another hard link and cannot be enforced: {linked}")):
        MacOSSandbox(profile)


@_unprivileged
@pytest.mark.parametrize(
    ("mode", "unchecked"), [(0o000, "locked"), (0o600, "locked/key")], ids=["unlistable", "unsearchable"]
)
def test_denied_paths_that_cannot_be_checked_fail_closed(tmp_path: Path, mode: int, unchecked: str) -> None:
    locked = tmp_path / "private" / "locked"
    locked.mkdir(parents=True)
    (locked / "key").write_text("secret")
    locked.chmod(mode)
    try:
        with pytest.raises(
            SandboxError, match=re.escape(f"cannot be checked for hard links: {locked.parent / unchecked}")
        ):
            MacOSSandbox(SandboxProfile((SandboxRoot(tmp_path),), denied_paths=(tmp_path / "private",)))
    finally:
        locked.chmod(0o755)


@pytest.mark.parametrize("denied", ["link/key", "alias"])
def test_denied_paths_through_symlinks_fail_closed(tmp_path: Path, denied: str) -> None:
    # Seatbelt matches the real path a file is opened by, so these would deny nothing.
    (tmp_path / "real").mkdir()
    (tmp_path / "real" / "key").write_text("secret")
    (tmp_path / "link").symlink_to(tmp_path / "real")
    (tmp_path / "alias").symlink_to(tmp_path / "real" / "key")

    with pytest.raises(
        SandboxError, match=re.escape(f"goes through a symlink and cannot be enforced: {tmp_path / denied}")
    ):
        MacOSSandbox(SandboxProfile((SandboxRoot(tmp_path),), denied_paths=(tmp_path / denied,)))


@pytest.mark.asyncio
@pytest.mark.parametrize(("change", "message"), [("link", "another hard link"), ("replace", "goes through a symlink")])
async def test_commands_fail_closed_on_denied_paths_changed_after_the_sandbox(
    tmp_path: Path, change: str, message: str
) -> None:
    denied = tmp_path / "private" / "credentials"
    denied.parent.mkdir()
    denied.write_text("secret")
    sandbox = MacOSSandbox(SandboxProfile((SandboxRoot(tmp_path),), denied_paths=(denied,)))
    sandbox._sandbox_exec = tmp_path / "sandbox-exec"
    sandbox._sandbox_exec.touch()
    if change == "link":
        os.link(denied, tmp_path / "alias")
    else:
        denied.parent.rename(tmp_path / "moved")
        denied.parent.symlink_to(tmp_path / "moved")

    with pytest.raises(SandboxError, match=message):
        await sandbox._execute(["/bin/sh", "-c", "true"])


@pytest.mark.parametrize("location", ["", "nested", "private"], ids=["root", "inside-root", "inside-denied-path"])
def test_scratch_must_be_outside_every_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, location: str) -> None:
    # Cleanup finds a command's processes by their access to its scratch, which a root or denial would change.
    (tmp_path / "nested").mkdir()
    (tmp_path / "private").mkdir()
    monkeypatch.setattr(tempfile, "tempdir", os.fspath(tmp_path / location))

    with pytest.raises(SandboxError, match="temporary directory must be a real path outside its roots"):
        MacOSSandbox(SandboxProfile((SandboxRoot(tmp_path),), denied_paths=(tmp_path / "private",)))


@pytest.mark.asyncio
async def test_commands_fail_closed_on_scratch_through_a_symlink(tmp_path: Path) -> None:
    # Seatbelt would see the scratch only under its real path, so the command could not read it.
    root = tmp_path / "root"
    real = tmp_path / "real"
    root.mkdir()
    real.mkdir()
    sandbox = MacOSSandbox(SandboxProfile((SandboxRoot(root),)))
    sandbox._sandbox_exec = tmp_path / "sandbox-exec"
    sandbox._sandbox_exec.touch()
    sandbox._scratch_parent = tmp_path / "link"
    sandbox._scratch_parent.symlink_to(real)

    with pytest.raises(SandboxError, match="temporary directory must be a real path"):
        await sandbox._execute(["/bin/sh", "-c", "true"])
    assert not any(real.iterdir())


def test_denied_paths_are_normalized_and_deduplicated(tmp_path: Path) -> None:
    denied = tmp_path / "private" / ".." / "secret"
    sandbox = MacOSSandbox(SandboxProfile((SandboxRoot(tmp_path),), denied_paths=(denied, denied)))

    assert sandbox.profile.denied_paths == (tmp_path / "secret",)


def test_nested_parent_root_replaces_redundant_descendant(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    sandbox = MacOSSandbox(SandboxProfile((SandboxRoot(nested), SandboxRoot(tmp_path))))

    # The first root is always retained because it defines cwd; the later parent is
    # also retained so its broader authority is represented explicitly.
    assert sandbox.roots == (SandboxRoot(nested), SandboxRoot(tmp_path))


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("error", None),
        ("nonzero", None),
        ("none-owned", None),
        ("finite", 12),
        ("exhausted", None),
        ("infinite", 258),
    ],
)
def test_owned_process_limit_handles_platform_results(
    monkeypatch: pytest.MonkeyPatch, mode: str, expected: int | None
) -> None:
    def run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        if mode == "error":
            raise subprocess.SubprocessError
        return subprocess.CompletedProcess(
            args[0],
            1 if mode == "nonzero" else 0,
            stdout="99999\n" if mode == "none-owned" else f"{seatbelt.os.getuid()}\n{seatbelt.os.getuid()}\n",
        )

    monkeypatch.setattr(seatbelt.subprocess, "run", run)
    if mode == "finite":
        monkeypatch.setattr(seatbelt.resource, "getrlimit", lambda _resource: (12, 12))
    elif mode == "exhausted":
        monkeypatch.setattr(seatbelt.resource, "getrlimit", lambda _resource: (2, 2))
    else:
        monkeypatch.setattr(seatbelt.resource, "getrlimit", lambda _resource: (seatbelt.resource.RLIM_INFINITY,) * 2)

    assert seatbelt._owned_process_limit() == expected


@pytest.mark.asyncio
async def test_run_validates_command_and_builds_bounded_shell_invocation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    sandbox = MacOSSandbox(SandboxProfile((SandboxRoot(tmp_path),)), timeout_seconds=0.5)
    sandbox._process_limit = 42

    with pytest.raises(SandboxError, match="blank"):
        await sandbox.run(" \t")
    with pytest.raises(SandboxError, match="at most"):
        await sandbox.run("bad\0command")
    with pytest.raises(SandboxError, match="at most"):
        await sandbox.run("x" * (MAX_COMMAND_CHARACTERS + 1))

    captured: list[str] = []

    async def execute(argv: list[str]) -> CommandResult:
        captured.extend(argv)
        return CommandResult(exit_code=0, stdout="ok", stderr="")

    monkeypatch.setattr(sandbox, "_execute", execute)
    result = await sandbox.run("printf ok")

    assert result.stdout == "ok"
    assert "ulimit -t 1 && ulimit -n 256 && ulimit -u 42" in captured[2]
    assert captured[-2:] == [sandbox.process_label, "printf ok"]

    sandbox._process_limit = None
    captured.clear()
    await sandbox.run("true")
    assert "ulimit -u" not in captured[2]


def test_environment_ignores_the_caller_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    planted = tmp_path / "planted"
    planted.mkdir()
    (planted / "git").touch(mode=0o755)
    monkeypatch.setenv("PATH", f"{planted}:/usr/bin:/bin")

    sandbox = MacOSSandbox(SandboxProfile((SandboxRoot(tmp_path),)))

    assert sandbox._environment(tmp_path)["PATH"] == "/usr/bin:/bin:/usr/sbin:/sbin"
    assert str(planted) not in sandbox._compiled_profile()
    assert planted not in sandbox._metadata_ancestors


@pytest.mark.asyncio
async def test_execute_fails_closed_when_seatbelt_is_unavailable(tmp_path: Path) -> None:
    sandbox = MacOSSandbox(SandboxProfile((SandboxRoot(tmp_path),)))
    sandbox._sandbox_exec = tmp_path / "missing-sandbox-exec"

    with pytest.raises(SandboxError, match="unavailable"):
        await sandbox._execute(["command"])


async def _prepare_execute(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, read_output: Any, *, returncode: int | None = 0
) -> tuple[MacOSSandbox, _Process, dict[str, Any]]:
    sandbox = MacOSSandbox(
        SandboxProfile(
            (SandboxRoot(tmp_path, RootAccess.READ_WRITE),),
            denied_paths=(tmp_path / "denied",),
        ),
        timeout_seconds=0.01,
    )
    sandbox_exec = tmp_path / "sandbox-exec"
    sandbox_exec.touch()
    sandbox._sandbox_exec = sandbox_exec
    sandbox._toolchain_trees = (tmp_path / "toolchain",)
    sandbox._toolchain_files = (tmp_path / "license",)
    sandbox._executables = (tmp_path / "executable",)
    sandbox._executable_trees = (tmp_path / "libexec",)
    sandbox._interpreters = (tmp_path / "interpreter",)
    sandbox._metadata_ancestors = (tmp_path.parent,)
    # Outside the root, where the sandbox requires its scratch.
    sandbox._scratch_parent = tmp_path.with_name(f"{tmp_path.name}-tmp")
    sandbox._scratch_parent.mkdir()
    process = _Process(returncode=returncode)
    captured: dict[str, Any] = {"swept": []}

    async def create(*argv: str, **kwargs: Any) -> _Process:
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        scratch = Path(kwargs["env"]["TMPDIR"])
        captured["scratch"] = scratch
        captured["sentinel_existed"] = (scratch / seatbelt._SCRATCH_SENTINEL).is_file()
        (scratch / "left-behind").write_text("x")
        return process

    def sweep(scratch: Path) -> None:
        assert scratch.is_dir()
        captured["swept"].append(scratch)

    monkeypatch.setattr(seatbelt, "_terminate_command_processes", sweep)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    monkeypatch.setattr(sandbox, "_read_output", read_output)
    return sandbox, process, captured


@pytest.mark.asyncio
async def test_execute_passes_complete_policy_and_returns_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def read_output(process: _Process, stdout: bytearray, stderr: bytearray) -> None:
        stdout.extend(b"output")
        stderr.extend(b"warning")

    sandbox, _process, captured = await _prepare_execute(monkeypatch, tmp_path, read_output, returncode=7)
    result = await sandbox._execute(["/bin/sh", "-c", "exit 7"])

    assert result == CommandResult(exit_code=7, stdout="output", stderr="warning")
    argv = captured["argv"]
    assert f"ROOT_0={tmp_path}" in argv
    assert f"DENIED_PATH_0={tmp_path / 'denied'}" in argv
    assert f"TOOLCHAIN_TREE_0={tmp_path / 'toolchain'}" in argv
    assert f"TOOLCHAIN_FILE_0={tmp_path / 'license'}" in argv
    assert f"EXECUTABLE_0={tmp_path / 'executable'}" in argv
    assert f"EXECUTABLE_TREE_0={tmp_path / 'libexec'}" in argv
    assert f"INTERPRETER_0={tmp_path / 'interpreter'}" in argv
    assert f"METADATA_ANCESTOR_0={tmp_path.parent}" in argv
    scratch = captured["scratch"]
    assert scratch.parent == sandbox._scratch_parent
    assert f"SCRATCH={scratch}" in argv
    assert f"SCRATCH_SENTINEL={scratch / seatbelt._SCRATCH_SENTINEL}" in argv
    assert captured["sentinel_existed"] is True
    assert captured["swept"] == [scratch]
    assert not scratch.exists()
    assert captured["kwargs"]["cwd"] == tmp_path
    assert captured["kwargs"]["start_new_session"] is True
    assert captured["kwargs"]["env"]["HOME"] == "/var/empty"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "flag", "message"),
    [
        (TimeoutError(), "timed_out", "command timed out"),
        (
            seatbelt._OutputLimitExceeded(),
            "truncated",
            "command output exceeded its bounded safety limit",
        ),
    ],
)
async def test_execute_kills_process_for_timeout_and_output_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: BaseException,
    flag: str,
    message: str,
) -> None:
    async def read_output(process: _Process, stdout: bytearray, stderr: bytearray) -> None:
        stdout.extend(b"partial")
        stderr.extend(b"existing\n")
        raise failure

    sandbox, process, captured = await _prepare_execute(monkeypatch, tmp_path, read_output, returncode=None)
    killed: list[_Process] = []

    async def kill(target: _Process, scratch: Path) -> None:
        assert scratch == captured["scratch"]
        killed.append(target)

    monkeypatch.setattr(sandbox, "_kill", kill)
    result = await sandbox._execute(["command"])

    assert result.exit_code == -1
    assert result.stdout == "partial"
    assert getattr(result, flag) is True
    assert result.stderr == f"existing\n{message}"
    assert killed == [process]
    assert captured["swept"] == [captured["scratch"]]
    assert not captured["scratch"].exists()


@pytest.mark.asyncio
async def test_execute_kills_and_propagates_unexpected_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async def read_output(process: _Process, stdout: bytearray, stderr: bytearray) -> None:
        raise RuntimeError("reader failed")

    sandbox, process, captured = await _prepare_execute(monkeypatch, tmp_path, read_output, returncode=None)
    killed: list[_Process] = []

    async def kill(target: _Process, scratch: Path) -> None:
        assert scratch == captured["scratch"]
        killed.append(target)

    monkeypatch.setattr(sandbox, "_kill", kill)
    with pytest.raises(RuntimeError, match="reader failed"):
        await sandbox._execute(["command"])
    assert killed == [process]
    assert captured["swept"] == [captured["scratch"]]
    assert not captured["scratch"].exists()


@pytest.mark.asyncio
async def test_read_output_collects_streams_and_waits(tmp_path: Path) -> None:
    sandbox = MacOSSandbox(SandboxProfile((SandboxRoot(tmp_path),)), max_output_bytes=20)
    process = _Process(returncode=0)
    process.stdout = _Stream(b"out", b"put")  # type: ignore[assignment]
    process.stderr = _Stream(b"err")  # type: ignore[assignment]
    stdout = bytearray()
    stderr = bytearray()

    await sandbox._read_output(process, stdout, stderr)  # type: ignore[arg-type]

    assert stdout == b"output"
    assert stderr == b"err"
    assert process.waited is True


@pytest.mark.asyncio
async def test_read_output_enforces_combined_limit_and_cancels_readers(tmp_path: Path) -> None:
    sandbox = MacOSSandbox(SandboxProfile((SandboxRoot(tmp_path),)), max_output_bytes=4)
    process = _Process(returncode=None)
    process.stdout = _Stream(b"12345")  # type: ignore[assignment]
    stdout = bytearray()
    stderr = bytearray()

    with pytest.raises(seatbelt._OutputLimitExceeded):
        await sandbox._read_output(process, stdout, stderr)  # type: ignore[arg-type]
    assert stdout == b"1234"

    process.stdout = _Stream(b"1234", b"5")  # type: ignore[assignment]
    stdout.clear()
    with pytest.raises(seatbelt._OutputLimitExceeded):
        await sandbox._read_output(process, stdout, stderr)  # type: ignore[arg-type]
    assert stdout == b"1234"


@pytest.mark.asyncio
async def test_kill_terminates_process_group_and_detached_descendants_before_draining(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    events: list[object] = []
    monkeypatch.setattr(seatbelt, "_terminate_command_processes", lambda scratch: events.append(("sweep", scratch)))

    class _Recorded(_Process):
        async def communicate(self) -> tuple[bytes, bytes]:
            events.append("drain")
            return await super().communicate()

    running = _Recorded(returncode=None)
    monkeypatch.setattr(seatbelt.os, "killpg", lambda pid, sig: events.append((pid, sig)))
    await MacOSSandbox._kill(running, tmp_path)  # type: ignore[arg-type]
    # Detached descendants can hold the pipes open, so they die before the output is drained.
    assert events == [(running.pid, seatbelt.signal.SIGKILL), ("sweep", tmp_path), "drain"]

    raced = _Recorded(returncode=None)

    def missing_process(_pid: int, _signal: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr(seatbelt.os, "killpg", missing_process)
    events.clear()
    await MacOSSandbox._kill(raced, tmp_path)  # type: ignore[arg-type]
    assert events == [("sweep", tmp_path), "drain"]

    finished = _Recorded(returncode=0)
    monkeypatch.setattr(seatbelt.os, "killpg", lambda _pid, _sig: pytest.fail("must not kill a completed process"))
    events.clear()
    await MacOSSandbox._kill(finished, tmp_path)  # type: ignore[arg-type]
    assert events == [("sweep", tmp_path), "drain"]


class _LibSystem:
    """Stands in for libSystem's process listing and sandbox_check."""

    def __init__(self, pids: list[int], allowed: dict[int, set[bytes]], *, sandboxed: set[int]) -> None:
        self.pids = pids
        self.allowed = allowed
        self.sandboxed = sandboxed
        self.listed = 0

    def proc_listallpids(self, buffer: Any, size: int) -> int:
        self.listed += 1
        if buffer is None:
            return 1
        capacity = size // ctypes.sizeof(ctypes.c_int)
        for index, pid in enumerate(self.pids[:capacity]):
            buffer[index] = pid
        return min(len(self.pids), capacity)

    def sandbox_check(self, pid: int, operation: bytes | None, kind: int, *args: Any) -> int:
        if operation is None:
            assert kind == seatbelt._SANDBOX_FILTER_NONE
            return 1 if pid in self.sandboxed else 0
        assert operation == b"file-read-data"
        assert kind == seatbelt._SANDBOX_FILTER_PATH | seatbelt._SANDBOX_CHECK_NO_REPORT
        return 0 if args[0].value in self.allowed.get(pid, set()) else 1


def test_process_ids_grow_the_buffer_until_every_process_fits(monkeypatch: pytest.MonkeyPatch) -> None:
    library = _LibSystem([0, *range(1, 600)], {}, sandboxed=set())
    monkeypatch.setattr(seatbelt, "_libsystem", lambda: library)

    assert seatbelt._process_ids() == tuple(range(1, 600))
    assert library.listed == 4


def test_process_ids_fail_closed_when_listing_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    library = _LibSystem([], {}, sandboxed=set())
    monkeypatch.setattr(seatbelt, "_libsystem", lambda: library)

    with pytest.raises(SandboxError, match="cannot list processes"):
        seatbelt._process_ids()


def test_runs_under_matches_only_the_policy_that_owns_the_scratch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    scratch = os.fsencode(tmp_path)
    sentinel = os.fsencode(tmp_path / seatbelt._SCRATCH_SENTINEL)
    library = _LibSystem(
        [],
        {
            1: {scratch},  # the command: scratch allowed, sentinel denied
            2: {scratch, sentinel},  # a broader sandbox that allows the whole temporary directory
            3: set(),  # an unrelated sandbox
            4: {scratch},  # unsandboxed: every check passes
        },
        sandboxed={1, 2, 3},
    )
    monkeypatch.setattr(seatbelt, "_libsystem", lambda: library)

    assert [pid for pid in (1, 2, 3, 4) if seatbelt._runs_under(pid, tmp_path)] == [1]


def test_terminate_command_processes_kills_until_none_remain(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    alive = {101, 102, 103}
    monkeypatch.setattr(seatbelt, "_process_ids", lambda: (seatbelt.os.getpid(), 1, *sorted(alive)))
    monkeypatch.setattr(seatbelt, "_runs_under", lambda pid, _scratch: pid in alive or pid == os.getpid())
    killed: list[int] = []

    def kill(pid: int, sig: int) -> None:
        assert sig == seatbelt.signal.SIGKILL
        killed.append(pid)
        alive.discard(pid)
        if pid == 102:
            raise ProcessLookupError

    monkeypatch.setattr(seatbelt.os, "kill", kill)
    seatbelt._terminate_command_processes(tmp_path)

    assert sorted(killed) == [101, 102, 103]
    assert alive == set()


def test_terminate_command_processes_reports_survivors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(seatbelt, "_process_ids", lambda: (101,))
    monkeypatch.setattr(seatbelt, "_runs_under", lambda _pid, _scratch: True)

    def kill(_pid: int, _sig: int) -> None:
        raise PermissionError

    monkeypatch.setattr(seatbelt.os, "kill", kill)
    with pytest.raises(SandboxError, match="could not be terminated"):
        seatbelt._terminate_command_processes(tmp_path)


def test_libsystem_fails_closed_without_supervision_symbols(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*_args: object, **_kwargs: object) -> None:
        raise OSError("not macOS")

    monkeypatch.setattr(seatbelt.ctypes, "CDLL", missing)
    seatbelt._libsystem.cache_clear()
    try:
        with pytest.raises(SandboxError, match="process supervision is unavailable"):
            seatbelt._libsystem()
    finally:
        seatbelt._libsystem.cache_clear()


@pytest.mark.asyncio
async def test_sandbox_tool_reports_budget_success_and_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    results = iter(
        (
            CommandResult(exit_code=0, stdout="ok", stderr=""),
            SandboxError("unavailable"),
        )
    )

    async def run(_self: MacOSSandbox, _command: str) -> CommandResult:
        result = next(results)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(MacOSSandbox, "run", run)
    budget = _Budget((False, "operation"), (True, None), (True, None))
    tool = seatbelt.sandbox_tools(
        SandboxProfile((SandboxRoot(tmp_path),)),
        budget=budget,
        exhausted_message="done",
        description="bounded bash",
    )[0]

    exhausted = await tool.ainvoke({"command": "first"})
    success = await tool.ainvoke({"command": "second"})
    failure = await tool.ainvoke({"command": "third"})

    assert tool.description == "bounded bash"
    assert exhausted == {"error": "done", "tool_budget": {"scope": "operation"}}
    assert success["result"] == CommandResult(exit_code=0, stdout="ok", stderr="").model_dump()
    assert success["tool_budget"] == {"scope": "available"}
    assert failure == {
        "error": "Bash failed with SandboxError: unavailable",
        "tool_budget": {"scope": "available"},
    }


def _fake_xcode(tmp_path: Path, *tools: str) -> Path:
    developer = tmp_path.resolve() / "Xcode.app" / "Contents" / "Developer"
    (developer / "usr" / "bin").mkdir(parents=True)
    toolchain = developer / "Toolchains" / "XcodeDefault.xctoolchain" / "usr" / "bin"
    toolchain.mkdir(parents=True)
    for tool in tools:
        (toolchain / tool).touch(mode=0o755)
    return developer


@pytest.mark.parametrize("outcome", ["selected", "unselected", "error", "missing", "not-a-toolchain"])
def test_developer_directory_comes_from_xcode_select_without_the_caller_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, outcome: str
) -> None:
    developer = _fake_xcode(tmp_path)
    selected = {"missing": tmp_path / "gone", "not-a-toolchain": tmp_path}.get(outcome, developer)
    monkeypatch.setenv("DEVELOPER_DIR", str(tmp_path / "elsewhere"))
    calls: list[tuple[list[str], dict[str, str]]] = []

    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((argv, kwargs["env"]))
        if outcome == "error":
            raise OSError("missing")
        return subprocess.CompletedProcess(argv, 2 if outcome == "unselected" else 0, f"{selected}\n", "")

    monkeypatch.setattr(seatbelt.subprocess, "run", run)
    _developer_directory.cache_clear()
    try:
        assert _developer_directory() == (developer if outcome == "selected" else None)
    finally:
        _developer_directory.cache_clear()
    # Never xcrun or a /usr/bin shim, which can raise the Command Line Tools install prompt, and
    # never the caller's DEVELOPER_DIR, which would move the read grant.
    assert calls == [(["/usr/bin/xcode-select", "-p"], {"PATH": "/usr/bin:/bin"})]


def _fake_homebrew(tmp_path: Path) -> Path:
    prefix = tmp_path.resolve() / "homebrew"
    for name in ("bin", "sbin", "opt", "Cellar", "share", "etc"):
        (prefix / name).mkdir(parents=True)
    (prefix / "bin" / "brew").touch(mode=0o755)
    return prefix


def _install_formula(
    prefix: Path, name: str, *, tools: tuple[str, ...] = (), dependencies: tuple[str, ...] = ()
) -> Path:
    keg = prefix / "Cellar" / name / "1.0"
    (keg / "bin").mkdir(parents=True)
    receipt = {"runtime_dependencies": [{"full_name": dependency, "version": "1.0"} for dependency in dependencies]}
    (keg / "INSTALL_RECEIPT.json").write_text(json.dumps(receipt))
    (prefix / "opt" / name).symlink_to(Path("..") / "Cellar" / name / "1.0", target_is_directory=True)
    for tool in tools:
        (keg / "bin" / tool).touch(mode=0o755)
        (prefix / "bin" / tool).symlink_to(Path("..") / "Cellar" / name / "1.0" / "bin" / tool)
    return keg


def _use_toolchains(monkeypatch: pytest.MonkeyPatch, *, homebrew: Path | None, developer: Path | None) -> None:
    monkeypatch.setattr(seatbelt, "_homebrew_prefix", lambda: homebrew)
    monkeypatch.setattr(seatbelt, "_developer_directory", lambda: developer)


def test_homebrew_prefix_uses_only_the_default_prefixes_without_running_anything(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prefix = _fake_homebrew(tmp_path)
    decoy = _fake_homebrew(tmp_path / "decoy")
    monkeypatch.setenv("HOMEBREW_PREFIX", str(decoy))
    monkeypatch.setattr(seatbelt.subprocess, "run", lambda *args, **kwargs: pytest.fail("must not run anything"))

    monkeypatch.setattr(seatbelt, "_HOMEBREW_PREFIXES", (tmp_path / "missing", prefix))
    assert _homebrew_prefix() == prefix
    monkeypatch.setattr(seatbelt, "_HOMEBREW_PREFIXES", (tmp_path / "missing", tmp_path))
    assert _homebrew_prefix() is None


def test_tools_grant_only_their_formulae_and_runtime_dependencies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    prefix = _fake_homebrew(tmp_path)
    cellar = prefix / "Cellar"
    _install_formula(prefix, "libunistring")
    # git reaches pcre2 directly and through gettext.
    _install_formula(prefix, "gettext", dependencies=("libunistring", "pcre2"))
    _install_formula(prefix, "pcre2")
    ripgrep = _install_formula(prefix, "ripgrep", tools=("rg",), dependencies=("pcre2",))
    # Tap formulae are recorded by full name; their kegs use the short one.
    git = _install_formula(prefix, "git", tools=("git",), dependencies=("pcre2", "example/tap/gettext"))
    _install_formula(prefix, "unrelated", tools=("unrelated",))
    _use_toolchains(monkeypatch, homebrew=prefix, developer=None)

    sandbox = MacOSSandbox(SandboxProfile((SandboxRoot(root),), tools=("rg", "git", "rg")))

    assert sandbox.profile.tools == ("rg", "git")
    kegs = ("ripgrep", "pcre2", "git", "gettext", "libunistring")
    assert sandbox._toolchain_trees == tuple(cellar / name / "1.0" for name in kegs)
    assert sandbox._toolchain_files == tuple(prefix / "opt" / name for name in kegs)
    assert sandbox._developer is None
    environment = sandbox._environment(tmp_path)
    assert environment["PATH"] == f"{ripgrep}/bin:{git}/bin:/usr/bin:/bin:/usr/sbin:/sbin"
    assert "DEVELOPER_DIR" not in environment
    # Ancestors are traversable, never listable: only the kegs and their opt links are readable.
    assert {prefix, cellar, prefix / "opt"} <= set(sandbox._metadata_ancestors)
    profile = sandbox._compiled_profile()
    assert '(allow file-read* file-map-executable (subpath (param "TOOLCHAIN_TREE_4")))' in profile
    assert '(allow file-read* (literal (param "TOOLCHAIN_FILE_4")))' in profile
    assert "TOOLCHAIN_TREE_5" not in profile
    assert not any("file-write" in line and "TOOLCHAIN_" in line for line in profile.splitlines())


def test_tools_prefer_homebrew_then_the_developer_install_then_the_base_system(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    prefix = _fake_homebrew(tmp_path)
    git = _install_formula(prefix, "git", tools=("git",))
    developer = _fake_xcode(tmp_path, "git", "clang")
    _use_toolchains(monkeypatch, homebrew=prefix, developer=developer)

    sandbox = MacOSSandbox(SandboxProfile((SandboxRoot(root),), tools=("git", "clang", "ls")))

    # The whole Xcode.app, which its shims read beyond Contents/Developer, and its license record.
    assert sandbox._toolchain_trees == (git, developer.parent.parent)
    assert sandbox._toolchain_files == (prefix / "opt" / "git", seatbelt._XCODE_LICENSE)
    environment = sandbox._environment(tmp_path)
    toolchain = developer / "Toolchains" / "XcodeDefault.xctoolchain"
    assert environment["PATH"] == (f"{git}/bin:{developer}/usr/bin:{toolchain}/usr/bin:/usr/bin:/bin:/usr/sbin:/sbin")
    # Pin the selection the grant was computed for, and keep libxcrun's cache in the scratch.
    assert environment["DEVELOPER_DIR"] == str(developer)
    assert environment["xcrun_db"] == str(tmp_path / "xcrun_db")

    base_only = MacOSSandbox(SandboxProfile((SandboxRoot(root),), tools=("ls",)))
    assert base_only._toolchain_trees == base_only._toolchain_files == ()
    assert base_only._environment(tmp_path)["PATH"] == "/usr/bin:/bin:/usr/sbin:/sbin"


def test_tools_grant_a_command_line_tools_install_as_is(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    developer = tmp_path / "CommandLineTools"
    (developer / "usr" / "bin").mkdir(parents=True)
    (developer / "usr" / "bin" / "make").touch(mode=0o755)
    _use_toolchains(monkeypatch, homebrew=None, developer=developer)

    sandbox = MacOSSandbox(SandboxProfile((SandboxRoot(root),), tools=("make",)))

    assert sandbox._toolchain_trees == (developer,)
    assert sandbox._toolchain_files == ()
    assert sandbox._environment(tmp_path)["PATH"] == f"{developer}/usr/bin:/usr/bin:/bin:/usr/sbin:/sbin"


def test_tools_from_one_unlinked_formula_share_its_grant(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    prefix = _fake_homebrew(tmp_path)
    git = _install_formula(prefix, "git", tools=("git", "scalar"))
    (prefix / "opt" / "git").unlink()
    _use_toolchains(monkeypatch, homebrew=prefix, developer=None)

    sandbox = MacOSSandbox(SandboxProfile((SandboxRoot(root),), tools=("git", "scalar")))

    assert sandbox._toolchain_trees == (git,)
    assert sandbox._toolchain_files == ()
    assert sandbox._environment(tmp_path)["PATH"] == f"{git}/bin:/usr/bin:/bin:/usr/sbin:/sbin"


def test_homebrew_executables_outside_a_formula_are_not_trusted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    prefix = _fake_homebrew(tmp_path)
    (prefix / "bin" / "ls").touch(mode=0o755)
    (prefix / "bin" / "planted").touch(mode=0o755)
    (prefix / "bin" / "unexecutable").touch(mode=0o644)
    _use_toolchains(monkeypatch, homebrew=prefix, developer=None)

    # A plain file in the prefix falls through to the base system's tool of the same name.
    assert MacOSSandbox(SandboxProfile((SandboxRoot(root),), tools=("ls",)))._toolchain_trees == ()
    for tool in ("planted", "unexecutable", "missing"):
        with pytest.raises(SandboxError, match=f"sandbox tool is not installed: {tool}"):
            MacOSSandbox(SandboxProfile((SandboxRoot(root),), tools=(tool,)))


@_unprivileged
@pytest.mark.parametrize("developer_tools", ["absent", "without-the-tool"])
def test_shims_without_their_developer_tool_are_not_installed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, system_bin: Path, developer_tools: str
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    # A /usr/bin shim runs the selected install's copy of its tool, so without one it only fails.
    (system_bin / "git").write_bytes(b"\xcf\xfa\xed\xfe\0_xcselect_invoke_xcrun\0")
    (system_bin / "unreadable").touch(mode=0o311)
    developer = None if developer_tools == "absent" else _fake_xcode(tmp_path, "clang")
    _use_toolchains(monkeypatch, homebrew=None, developer=developer)

    for tool in ("git", "unreadable"):
        with pytest.raises(SandboxError, match=f"sandbox tool is not installed: {tool}"):
            MacOSSandbox(SandboxProfile((SandboxRoot(root),), tools=(tool,)))
    sandbox = MacOSSandbox(SandboxProfile((SandboxRoot(root),), tools=("ls",), optional_tools=("git",)))
    assert sandbox.tools == ("ls",)


def test_tools_fail_closed_on_broken_formulae(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    prefix = _fake_homebrew(tmp_path)
    _install_formula(prefix, "broken", tools=("broken",), dependencies=("absent",))
    garbled = _install_formula(prefix, "garbled", tools=("garbled",))
    (garbled / "INSTALL_RECEIPT.json").write_text('{"runtime_dependencies": [{"version": "1.0"}]}')
    # An opt link must name one installed version, not a whole formula directory.
    _install_formula(prefix, "partial", tools=("partial",), dependencies=("unversioned",))
    (prefix / "Cellar" / "unversioned").mkdir()
    (prefix / "opt" / "unversioned").symlink_to(Path("..") / "Cellar" / "unversioned", target_is_directory=True)
    _use_toolchains(monkeypatch, homebrew=prefix, developer=None)

    with pytest.raises(SandboxError, match="dependency of broken is not installed: absent"):
        MacOSSandbox(SandboxProfile((SandboxRoot(root),), tools=("broken",)))
    with pytest.raises(SandboxError, match="dependency of partial is not installed: unversioned"):
        MacOSSandbox(SandboxProfile((SandboxRoot(root),), tools=("partial",)))
    with pytest.raises(SandboxError, match="cannot read the Homebrew install receipt"):
        MacOSSandbox(SandboxProfile((SandboxRoot(root),), tools=("garbled",)))


def test_optional_tools_grant_what_is_installed_and_skip_the_rest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    prefix = _fake_homebrew(tmp_path)
    _install_formula(prefix, "pcre2")
    ripgrep = _install_formula(prefix, "ripgrep", tools=("rg",))
    git = _install_formula(prefix, "git", tools=("git",))
    _install_formula(prefix, "broken", tools=("broken",), dependencies=("pcre2", "absent"))
    developer = _fake_xcode(tmp_path, "clang")
    _use_toolchains(monkeypatch, homebrew=prefix, developer=developer)

    sandbox = MacOSSandbox(
        SandboxProfile(
            (SandboxRoot(root),), tools=("git",), optional_tools=("missing", "rg", "broken", "ls", "missing")
        )
    )

    assert sandbox.profile.optional_tools == ("missing", "rg", "broken", "ls")
    assert sandbox.tools == ("git", "rg", "ls")
    # A skipped tool grants nothing, not even the parts of its formula that resolved.
    assert sandbox._toolchain_trees == (git, ripgrep)
    assert sandbox._toolchain_files == (prefix / "opt" / "git", prefix / "opt" / "ripgrep")
    assert sandbox._developer is None
    assert sandbox._environment(tmp_path)["PATH"] == f"{git}/bin:{ripgrep}/bin:/usr/bin:/bin:/usr/sbin:/sbin"

    # Required tools still fail closed alongside optional ones.
    with pytest.raises(SandboxError, match="sandbox tool is not installed: missing"):
        MacOSSandbox(SandboxProfile((SandboxRoot(root),), tools=("missing",), optional_tools=("rg",)))
    with pytest.raises(SandboxError, match="dependency of broken is not installed: absent"):
        MacOSSandbox(SandboxProfile((SandboxRoot(root),), tools=("broken",), optional_tools=("rg",)))

    only_optional = MacOSSandbox(SandboxProfile((SandboxRoot(root),), tools=(), optional_tools=("clang", "missing")))
    assert only_optional.tools == ("clang",)
    assert only_optional._developer == developer


def test_tools_reject_a_toolchain_overlapping_a_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    prefix = _fake_homebrew(tmp_path)
    git = _install_formula(prefix, "git", tools=("git",))
    developer = _fake_xcode(tmp_path, "clang")
    _use_toolchains(monkeypatch, homebrew=prefix, developer=developer)

    for root, tool in ((tmp_path, "git"), (git / "bin", "git"), (developer / "usr", "clang")):
        with pytest.raises(SandboxError, match="overlap a sandbox root"):
            MacOSSandbox(SandboxProfile((SandboxRoot(root),), tools=(tool,)))
    # Only the granted kegs count; the rest of the prefix can be a root.
    assert MacOSSandbox(SandboxProfile((SandboxRoot(prefix / "share"),), tools=("git",)))._toolchain_trees


def test_tools_reject_programs_they_run_inside_a_root(tmp_path: Path, system_bin: Path) -> None:
    interpreters = tmp_path / "interpreters"
    interpreters.mkdir()
    (interpreters / "perl").touch(mode=0o755)
    (system_bin / "script").write_text(f"#!{interpreters / 'perl'}\n")
    (system_bin / "script").chmod(0o755)

    with pytest.raises(SandboxError, match=re.escape(f"overlap a sandbox root: {system_bin / 'cat'}")):
        MacOSSandbox(SandboxProfile((SandboxRoot(system_bin),), tools=("cat",)))
    with pytest.raises(SandboxError, match=re.escape(f"overlap a sandbox root: {interpreters / 'perl'}")):
        MacOSSandbox(SandboxProfile((SandboxRoot(interpreters),), tools=("script",)))


def test_shell_runs_the_selected_shell(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    selected = tmp_path / "zsh"
    selected.touch(mode=0o755)
    selection = tmp_path / "sh"
    selection.symlink_to(selected)
    monkeypatch.setattr(seatbelt, "_SHELL_SELECTION", selection)

    assert seatbelt._shell_executables() == (Path("/bin/sh"), selected.resolve())
    selection.unlink()
    assert seatbelt._shell_executables() == _SHELL
    monkeypatch.setattr(seatbelt, "_SHELL", selected.resolve())
    selection.symlink_to(selected)
    assert seatbelt._shell_executables() == (selected.resolve(),)


@pytest.mark.parametrize(
    ("first_line", "interpreter"),
    [
        (b"#!{interpreter} -w\n", "interpreter"),
        (b"#! {interpreter}\n", "interpreter"),
        (b"#!\n", None),
        (b"\xcf\xfa\xed\xfe binary", None),
    ],
)
def test_scripts_name_their_interpreter(tmp_path: Path, first_line: bytes, interpreter: str | None) -> None:
    program = tmp_path / "program"
    program.write_bytes(first_line.replace(b"{interpreter}", os.fsencode(tmp_path / "interpreter")))

    assert seatbelt._interpreter(program) == (None if interpreter is None else (tmp_path / interpreter).resolve())
    assert seatbelt._interpreter(tmp_path / "missing") is None


def test_tools_run_only_their_programs_helpers_and_interpreters(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, system_bin: Path
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    prefix = _fake_homebrew(tmp_path)
    git = _install_formula(prefix, "git", tools=("git", "scalar"))
    (git / "libexec" / "git-core").mkdir(parents=True)
    (git / "libexec" / "bin").mkdir()
    python = _install_formula(prefix, "python@3.13")
    framework = python / "Frameworks" / "Python.framework" / "Versions" / "3.13"
    (framework / "bin").mkdir(parents=True)
    (framework / "bin" / "python3.13").touch(mode=0o755)
    linter = _install_formula(prefix, "linter", tools=("lint",), dependencies=("python@3.13",))
    (linter / "bin" / "lint").write_text(
        f"#!{prefix}/opt/python@3.13/Frameworks/Python.framework/Versions/3.13/bin/python3.13\n"
    )
    _use_toolchains(monkeypatch, homebrew=prefix, developer=None)

    sandbox = MacOSSandbox(SandboxProfile((SandboxRoot(root),), tools=("git", "lint", "cat")))

    assert sandbox._executables == (*_SHELL, git / "bin" / "git", linter / "bin" / "lint", system_bin / "cat")
    # A tool's own helpers are in libexec, and a framework interpreter re-launches from its bundle.
    assert sandbox._executable_trees == (git / "libexec" / "git-core", framework)
    assert sandbox._interpreters == (framework / "bin" / "python3.13",)
    profile = sandbox._compiled_profile()
    assert "(allow process-exec)" not in profile
    assert '(allow process-exec (literal (param "EXECUTABLE_4")))' in profile
    assert '(allow process-exec (subpath (param "EXECUTABLE_TREE_1")))' in profile
    assert '(allow process-exec-interpreter (literal (param "INTERPRETER_0")))' in profile
    assert "EXECUTABLE_5" not in profile
    # Another tool from the same formula does not bring git's helpers.
    assert MacOSSandbox(SandboxProfile((SandboxRoot(root),), tools=("scalar",)))._executable_trees == ()


def test_developer_tools_run_through_their_shims(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    developer = tmp_path.resolve() / "CommandLineTools"
    framework = developer / "Library" / "Frameworks" / "Python3.framework" / "Versions" / "3.9"
    (framework / "bin").mkdir(parents=True)
    (framework / "bin" / "python3.9").touch(mode=0o755)
    (developer / "usr" / "bin").mkdir(parents=True)
    (developer / "usr" / "bin" / "python3").symlink_to(framework / "bin" / "python3.9")
    (developer / "usr" / "bin" / "git").touch(mode=0o755)
    (developer / "usr" / "bin" / "swift").touch(mode=0o755)
    (developer / "usr" / "bin" / "clang").touch(mode=0o755)
    libexec = developer / "usr" / "libexec"
    (libexec / "git-core").mkdir(parents=True)
    (libexec / "swift").mkdir()
    (libexec / "migcom").touch(mode=0o755)
    (seatbelt._SHIM_DIRECTORY / "git").touch(mode=0o755)
    _use_toolchains(monkeypatch, homebrew=None, developer=developer)

    sandbox = MacOSSandbox(SandboxProfile((SandboxRoot(root),), tools=("git", "python3", "swift")))

    git = (developer / "usr" / "bin" / "git", seatbelt._SHIM_DIRECTORY / "git")
    assert sandbox._executables == (*_SHELL, *git, framework / "bin" / "python3.9", developer / "usr" / "bin" / "swift")
    # Each tool runs only its own helpers from the libexec the install's tools share.
    assert sandbox._executable_trees == (libexec / "git-core", framework, libexec / "swift")
    assert sandbox._interpreters == ()
    assert MacOSSandbox(SandboxProfile((SandboxRoot(root),), tools=("clang",)))._executable_trees == ()


def test_toolchains_are_not_looked_up_without_tools(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(seatbelt, "_developer_directory", lambda: pytest.fail("must not look up"))
    monkeypatch.setattr(seatbelt, "_homebrew_prefix", lambda: pytest.fail("must not look up"))

    sandbox = MacOSSandbox(SandboxProfile((SandboxRoot(tmp_path),), tools=()))

    assert sandbox.tools == ()
    assert sandbox._toolchain_trees == ()
    # Only the shell runs.
    assert sandbox._executables == _SHELL
    profile = sandbox._compiled_profile()
    assert "TOOLCHAIN_" not in profile
    assert "EXECUTABLE_TREE_" not in profile
    assert "INTERPRETER_" not in profile
    environment = sandbox._environment(tmp_path)
    assert "DEVELOPER_DIR" not in environment
    assert "xcrun_db" not in environment
    assert environment["PATH"] == "/usr/bin:/bin:/usr/sbin:/sbin"
    assert environment["TMPDIR"] == str(tmp_path)


@pytest.mark.parametrize("field", ["tools", "optional_tools"])
@pytest.mark.parametrize("tools", ["git", ("",), (".",), ("..",), ("bin/git",), ("/usr/bin/git",), ("g\0",)])
def test_profile_rejects_tools_that_are_not_command_names(tmp_path: Path, field: str, tools: Any) -> None:
    with pytest.raises(SandboxError, match="command name"):
        SandboxProfile((SandboxRoot(tmp_path),), **{field: tools})


def test_profile_rejects_tools_that_are_both_required_and_optional(tmp_path: Path) -> None:
    with pytest.raises(SandboxError, match="both required and optional: git, rg"):
        SandboxProfile((SandboxRoot(tmp_path),), tools=("rg", "git", "ls"), optional_tools=("git", "rg", "jq"))


def test_sandbox_tool_description_names_the_granted_tools(tmp_path: Path) -> None:
    def description(**tools: Any) -> str:
        return seatbelt.sandbox_tools(SandboxProfile((SandboxRoot(tmp_path),), **tools), budget=_Budget())[
            0
        ].description

    plain = description()
    assert plain == seatbelt.DEFAULT_TOOL_DESCRIPTION
    assert f"Besides shell builtins, only these commands can run: {', '.join(INSPECTION_TOOLS)}." in plain
    assert "only these commands can run: ls, cat." in description(tools=("ls", "cat", "ls"))
    # Only the optional tools that are installed are offered.
    assert "only these commands can run: ls, cat." in description(tools=("ls",), optional_tools=("rg", "cat"))
    assert "Only shell builtins can run." in description(tools=())
