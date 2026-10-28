"""Real Linux enforcement; required CI jobs must never pass by skipping it all."""

from __future__ import annotations

import asyncio
import http.server
import os
import shlex
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from localmcp.sandbox import (
    INSPECTION_TOOLS,
    MAX_ADDITIONAL_PROCESSES,
    WRITE_TOOLS,
    CommandResult,
    RootAccess,
    SandboxError,
    SandboxProfile,
    SandboxRoot,
)
from localmcp.sandbox.bubblewrap import LinuxSandbox, _elf_interpreter

_AVAILABLE = sys.platform == "linux" and Path("/usr/bin/bwrap").is_file() and Path("/usr/bin/python3").is_file()
if not _AVAILABLE:
    _reason = "Linux bubblewrap requires Linux, /usr/bin/bwrap, and /usr/bin/python3"
    if os.environ.get("LOCALMCP_REQUIRE_BUBBLEWRAP") == "1":
        pytest.fail(f"LOCALMCP_REQUIRE_BUBBLEWRAP=1 but {_reason}", pytrace=False)
    pytest.skip(_reason, allow_module_level=True)

pytestmark = pytest.mark.bubblewrap
_TOOLS = (*INSPECTION_TOOLS, *WRITE_TOOLS)


@dataclass(frozen=True)
class Workspace:
    writable: Path
    readonly: Path
    outside: Path

    def profile(self, *, tools: tuple[str, ...] = _TOOLS, **kwargs: object) -> SandboxProfile:
        return SandboxProfile(
            (SandboxRoot(self.writable, RootAccess.READ_WRITE), SandboxRoot(self.readonly)),
            tools=tools,
            **kwargs,  # type: ignore[arg-type]
        )


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    paths = Workspace(*(tmp_path / name for name in ("rw", "ro", "outside")))
    for path in (paths.writable, paths.readonly, paths.outside):
        path.mkdir()
    return paths


async def _run(profile: SandboxProfile, command: str, **kwargs: object) -> CommandResult:
    return await LinuxSandbox(profile, **kwargs).run(command)  # type: ignore[arg-type]


def _denied(result: CommandResult, secret: str = "secret-sentinel") -> None:
    assert result.exit_code != 0, result
    assert not result.timed_out, result
    assert secret not in result.stdout
    assert secret not in result.stderr


async def test_first_root_is_cwd_and_root_permissions_reach_host(workspace: Workspace) -> None:
    (workspace.readonly / "data").write_text("readable")
    result = await _run(workspace.profile(), f"pwd; cat {workspace.readonly}/data; printf written > created")
    assert result.exit_code == 0, result
    assert result.stdout == f"{workspace.writable}\nreadable"
    assert (workspace.writable / "created").read_text() == "written"


async def test_default_file_tool_finds_system_magic_database(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    (workspace.readonly / "data").write_text("plain ASCII text\n")
    monkeypatch.delenv("MAGIC", raising=False)
    result = await _run(SandboxProfile((SandboxRoot(workspace.readonly),)), "file data")
    assert result.exit_code == 0, result
    assert result.stdout.startswith("data:")
    assert "ASCII text" in result.stdout


@pytest.mark.parametrize("command", ["printf changed > data", "chmod 600 data", "rm data"])
async def test_read_only_root_cannot_be_changed(workspace: Workspace, command: str) -> None:
    target = workspace.readonly / "data"
    target.write_text("original")
    result = await _run(workspace.profile(), f"cd {workspace.readonly}; {command}")
    _denied(result)
    assert target.read_text() == "original"
    assert target.stat().st_mode & 0o777 == 0o644


@pytest.mark.parametrize("child_first", [False, True], ids=["parent-first", "child-first"])
async def test_explicit_writable_child_of_read_only_root_retains_write_authority(
    workspace: Workspace, child_first: bool
) -> None:
    parent = workspace.readonly
    child = parent / "child"
    child.mkdir()
    roots = (SandboxRoot(parent), SandboxRoot(child, RootAccess.READ_WRITE))
    if child_first:
        roots = tuple(reversed(roots))
    command = (
        f"if printf blocked > {parent}/blocked; then exit 99; fi; "
        f'printf child > {child}/created && printf scratch > "$TMPDIR/item" '
        f'&& pwd && cat {child}/created && cat "$TMPDIR/item"'
    )
    result = await _run(SandboxProfile(roots, tools=("cat",)), command)
    assert result.exit_code == 0, result
    assert result.stdout == f"{roots[0].path}\nchildscratch"
    assert not (parent / "blocked").exists()
    assert (child / "created").read_text() == "child"


@pytest.mark.parametrize("command", ["cat {outside}/secret", "ls {outside}", "printf changed > {outside}/created"])
async def test_outside_roots_is_inaccessible(workspace: Workspace, command: str) -> None:
    (workspace.outside / "secret").write_text("secret-sentinel")
    result = await _run(workspace.profile(), command.format(outside=workspace.outside))
    _denied(result)
    assert not (workspace.outside / "created").exists()


@pytest.mark.parametrize("create_inside", [False, True])
async def test_symlinks_cannot_escape_roots(workspace: Workspace, create_inside: bool) -> None:
    (workspace.outside / "secret").write_text("secret-sentinel")
    command = "cat alias/secret"
    if create_inside:
        command = f"ln -s {workspace.outside} alias && {command}"
    else:
        (workspace.writable / "alias").symlink_to(workspace.outside, target_is_directory=True)
    _denied(await _run(workspace.profile(), command))


@pytest.mark.parametrize("command", ["cat secret", "cat private/token", "ls private", "ln secret alias && cat alias"])
async def test_denied_paths_fail_without_disclosing_or_substituting_empty_data(
    workspace: Workspace, command: str
) -> None:
    secret = workspace.writable / "secret"
    secret.write_text("secret-sentinel")
    private = workspace.writable / "private"
    private.mkdir()
    (private / "token").write_text("secret-sentinel")
    profile = workspace.profile(denied_paths=(secret, private))
    _denied(await _run(profile, command))
    allowed = await _run(profile, "printf visible > sibling && cat sibling")
    assert allowed.exit_code == 0, allowed
    assert allowed.stdout == "visible"


async def test_host_hard_links_are_rechecked_before_each_invocation(workspace: Workspace) -> None:
    secret = workspace.writable / "secret"
    secret.write_text("secret-sentinel")
    profile = workspace.profile(denied_paths=(secret,))
    sandbox = LinuxSandbox(profile)
    os.link(secret, workspace.readonly / "alias")
    with pytest.raises(SandboxError, match="hard link"):
        await sandbox.run("cat secret")
    with pytest.raises(SandboxError, match="hard link"):
        LinuxSandbox(profile)


@pytest.mark.parametrize(
    "command",
    [
        "printf changed > docs/source/data",
        "printf changed >> docs/source/data",
        "chmod 600 docs/source/data",
        "touch docs/source/created",
        "mkdir docs/source/created",
        "rm docs/source/data",
        "mv docs/source/data moved",
        "mv docs/source moved",
        "mv docs moved",
        "ln docs/source/data alias && printf changed >> alias",
        "rm -rf docs",
    ],
)
async def test_nested_read_only_root_protects_contents_and_ancestors(workspace: Workspace, command: str) -> None:
    protected = workspace.writable / "docs" / "source"
    protected.mkdir(parents=True)
    (protected / "data").write_text("original")
    profile = SandboxProfile(
        (SandboxRoot(workspace.writable, RootAccess.READ_WRITE), SandboxRoot(protected)), tools=_TOOLS
    )
    result = await _run(profile, command)
    _denied(result)
    assert (protected / "data").read_text() == "original"
    assert (protected / "data").stat().st_nlink == 1
    assert sorted(path.name for path in protected.iterdir()) == ["data"]
    allowed = await _run(profile, "printf ok > sibling && mv sibling renamed && cat renamed")
    assert allowed.exit_code == 0, allowed
    assert allowed.stdout == "ok"


async def test_environment_is_sanitized(workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCALMCP_TEST_SECRET", "secret-sentinel")
    monkeypatch.setenv("LD_PRELOAD", str(workspace.outside / "inject.so"))
    result = await _run(workspace.profile(tools=("env",)), "env")
    assert result.exit_code == 0, result
    assert "secret-sentinel" not in result.stdout
    assert "LOCALMCP_TEST_SECRET" not in result.stdout
    assert "LD_PRELOAD" not in result.stdout
    values = dict(line.split("=", 1) for line in result.stdout.splitlines())
    assert values["HOME"] == "/var/empty"
    assert values["PATH"] == "/usr/bin:/bin:/usr/sbin:/sbin"


async def test_workload_has_no_capabilities_and_cannot_gain_privileges(workspace: Workspace) -> None:
    profile = SandboxProfile((SandboxRoot(workspace.readonly),))
    result = await _run(profile, "grep -E '^(Cap(Inh|Prm|Eff|Bnd|Amb)|NoNewPrivs):' /proc/self/status")
    assert result.exit_code == 0, result
    values = dict(line.split(":", 1) for line in result.stdout.splitlines())
    capabilities = {"CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb"}
    assert set(values) == capabilities | {"NoNewPrivs"}
    assert all(int(values[name].strip(), 16) == 0 for name in capabilities), values
    assert values["NoNewPrivs"].strip() == "1"


async def test_explicitly_granted_unshare_cannot_create_namespaces(workspace: Workspace) -> None:
    result = await _run(workspace.profile(tools=("unshare",)), "unshare --user /bin/sh -c 'printf namespace-created'")
    _denied(result, secret="namespace-created")


async def test_scratch_is_private_per_invocation_and_removed(workspace: Workspace) -> None:
    sandbox = LinuxSandbox(workspace.profile(tools=()))
    command = 'test ! -e "$TMPDIR/marker" && printf x > "$TMPDIR/marker" && printf %s "$TMPDIR"'
    first, second = await sandbox.run(command), await sandbox.run(command)
    assert first.exit_code == second.exit_code == 0, (first, second)
    assert first.stdout != second.stdout
    assert not Path(first.stdout).exists()
    assert not Path(second.stdout).exists()


@pytest.mark.parametrize("target", ['"$TMPDIR/file"', "/tmp/file", "/dev/shm/file", "/new-file"])
async def test_read_only_profiles_gain_no_writable_filesystem_space(workspace: Workspace, target: str) -> None:
    profile = SandboxProfile((SandboxRoot(workspace.readonly),), tools=())
    _denied(await _run(profile, f"printf x > {target}"))


@pytest.mark.parametrize("target", ["/dev/testdir", "/dev/shm/testdir"])
async def test_read_only_profiles_cannot_create_device_directories(workspace: Workspace, target: str) -> None:
    profile = SandboxProfile((SandboxRoot(workspace.readonly),), tools=("mkdir",))
    _denied(await _run(profile, f"mkdir {target}"))


async def test_only_listed_tools_run_and_missing_optional_is_not_found(workspace: Workspace) -> None:
    result = await _run(
        workspace.profile(tools=("cat",), optional_tools=("localmcp-no-such-tool",)),
        "printf builtin; /bin/ls; printf ':%s:' $?; localmcp-no-such-tool; printf '%s' $?",
    )
    assert result.exit_code == 0, result
    assert result.stdout == "builtin:126:127"


@pytest.mark.parametrize(
    "command",
    ["awk 'BEGIN { exit system(\"/bin/ls /\") }'", "find . -exec /bin/ls / {} +", "printf / | xargs /bin/ls"],
)
async def test_listed_tools_cannot_execute_unlisted_tools(workspace: Workspace, command: str) -> None:
    _denied(await _run(workspace.profile(tools=("awk", "find", "xargs")), command))


@pytest.mark.parametrize(
    "command",
    ["cp /bin/ls copy && ./copy /", "printf '#!/bin/sh\\necho escaped\\n' > script && chmod +x script && ./script"],
)
async def test_workspace_executables_are_denied(workspace: Workspace, command: str) -> None:
    _denied(await _run(workspace.profile(tools=("cp", "chmod")), command), secret="escaped")


async def test_dynamic_loader_cannot_bypass_executable_policy(workspace: Workspace) -> None:
    loader = _elf_interpreter(Path("/bin/sh"))
    assert loader is not None
    result = await _run(workspace.profile(tools=("cp",)), f"cp /bin/ls copy && {loader} ./copy /")
    _denied(result)
    result = await _run(workspace.profile(tools=()), f"{loader} /bin/ls /")
    _denied(result)


async def test_workload_exit_code_and_signal_are_reported(workspace: Workspace) -> None:
    result = await _run(workspace.profile(tools=()), "printf out; printf err >&2; exit 7")
    assert result == CommandResult(exit_code=7, stdout="out", stderr="err")
    signal_result = await _run(workspace.profile(tools=()), "kill -TERM $$")
    assert signal_result.exit_code == -15, signal_result


async def test_timeout_terminates_promptly(workspace: Workspace) -> None:
    started = time.monotonic()
    result = await _run(workspace.profile(tools=("sleep",)), "sleep 30", timeout_seconds=1)
    assert result.timed_out and result.exit_code == -1, result
    assert "command timed out" in result.stderr
    assert time.monotonic() - started < 10


async def test_combined_output_limit_is_enforced(workspace: Workspace) -> None:
    result = await _run(
        workspace.profile(tools=()), "while :; do printf 1234567890; printf abcdefghij >&2; done", max_output_bytes=4096
    )
    assert result.truncated and result.exit_code == -1, result
    assert len(result.stdout.encode()) <= 4096
    assert "command output exceeded its bounded safety limit" in result.stderr
    captured_stderr = result.stderr.removesuffix("command output exceeded its bounded safety limit").rstrip("\n")
    assert len(result.stdout.encode()) + len(captured_stderr.encode()) == 4096


async def test_closed_pipes_end_writers_quietly(workspace: Workspace) -> None:
    result = await _run(workspace.profile(tools=("yes", "head")), "yes | head -n 1")
    assert result == CommandResult(exit_code=0, stdout="y\n", stderr=""), result


_FORKS = """import os
count = 0
try:
    while count < 1024:
        if os.fork() == 0:
            os.pause()
            os._exit(0)
        count += 1
except BlockingIOError:
    print(count)
"""


async def test_process_limit_ignores_the_callers_other_processes(workspace: Workspace) -> None:
    # Host processes the caller already runs must not enlarge the command's allowance.
    holders = [subprocess.Popen(["/usr/bin/sleep", "30"]) for _ in range(16)]
    try:
        result = await _run(workspace.profile(tools=("python3",)), f"python3 -I -S -c {shlex.quote(_FORKS)}")
    finally:
        for holder in holders:
            holder.kill()
            holder.wait()
    assert result.exit_code == 0 and result.stdout.strip(), result
    assert 0 < int(result.stdout) <= MAX_ADDITIONAL_PROCESSES


_DETACHED = """import os,time
if os.fork(): os._exit(0)
os.setsid()
if os.fork(): os._exit(0)
for fd in (0,1,2): os.close(fd)
open('started','w').close()
time.sleep(1.5)
open('late','w').close()
"""


@pytest.mark.parametrize("finish", ["exit", "timeout", "cancel"])
async def test_detached_descendants_are_dead_before_run_finishes(workspace: Workspace, finish: str) -> None:
    sandbox = LinuxSandbox(
        workspace.profile(tools=("python3", "sleep")), timeout_seconds=1 if finish == "timeout" else 10
    )
    command = f"python3 -I -S -c {shlex.quote(_DETACHED)}; while [ ! -e started ]; do sleep 0.02; done"
    if finish != "exit":
        command += "; sleep 30"
    task = asyncio.create_task(sandbox.run(command))
    if finish == "cancel":
        async with asyncio.timeout(5):
            while not (workspace.writable / "started").exists():
                await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        result = await task
        assert result.timed_out is (finish == "timeout"), result
    assert (workspace.writable / "started").exists()
    await asyncio.sleep(1.7)
    assert not (workspace.writable / "late").exists()


async def test_supervision_spares_concurrent_commands(workspace: Workspace) -> None:
    sandbox = LinuxSandbox(workspace.profile(tools=("sleep",)))
    running = asyncio.create_task(sandbox.run("sleep 1; printf survived"))
    await asyncio.sleep(0.15)
    quick = await sandbox.run("true")
    result = await running
    assert quick.exit_code == 0, quick
    assert result.exit_code == 0 and result.stdout == "survived", result


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"network-ok")

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def http_port() -> Iterator[int]:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.mark.parametrize("network", [False, True])
async def test_network_requires_explicit_grant(workspace: Workspace, http_port: int, network: bool) -> None:
    result = await _run(
        workspace.profile(tools=("curl",), network=network), f"curl -sS --max-time 2 http://127.0.0.1:{http_port}"
    )
    if network:
        assert result.exit_code == 0 and result.stdout == "network-ok", result
    else:
        _denied(result, secret="network-ok")


async def test_network_grant_resolves_host_names(workspace: Workspace) -> None:
    result = await _run(workspace.profile(tools=("getent",), network=True), "getent hosts localhost")
    assert result.exit_code == 0 and "localhost" in result.stdout, result


@pytest.mark.parametrize("ipc", [False, True])
async def test_shared_memory_files_require_ipc_grant(workspace: Workspace, ipc: bool) -> None:
    result = await _run(
        workspace.profile(tools=("cat",), ipc=ipc), "printf shared > /dev/shm/item && cat /dev/shm/item"
    )
    if ipc:
        assert result.exit_code == 0 and result.stdout == "shared", result
    else:
        _denied(result)


@pytest.mark.parametrize("ipc", [False, True])
async def test_sysv_shared_memory_requires_ipc_grant(workspace: Workspace, ipc: bool) -> None:
    result = await _run(workspace.profile(tools=("ipcmk",), ipc=ipc), "ipcmk -M 4096")
    if ipc:
        assert result.exit_code == 0, result
        assert result.stdout.startswith("Shared memory id:")
    else:
        _denied(result)


@pytest.mark.parametrize("command", ["ipcmk -S 1", "ipcmk -Q"])
async def test_ipc_grant_does_not_allow_semaphores_or_message_queues(workspace: Workspace, command: str) -> None:
    _denied(await _run(workspace.profile(tools=("ipcmk",), ipc=True), command))


async def test_git_operates_inside_writable_root(workspace: Workspace) -> None:
    subprocess.run(["/usr/bin/git", "init", "-q", str(workspace.writable)], check=True)
    result = await _run(
        workspace.profile(tools=("git",)), "printf tracked > tracked; git add tracked; git diff --cached --stat"
    )
    assert result.exit_code == 0, result
    assert "tracked" in result.stdout
    assert (workspace.writable / ".git" / "index").is_file()
