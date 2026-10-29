"""Real Linux enforcement; required CI jobs must never pass by skipping it all."""

from __future__ import annotations

import asyncio
import errno
import http.server
import os
import shlex
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from localmcp.sandbox import (
    INSPECTION_TOOLS,
    WRITE_TOOLS,
    CommandResult,
    RootAccess,
    SandboxError,
    SandboxProfile,
    SandboxRoot,
)
from localmcp.sandbox.bubblewrap import LinuxSandbox
from localmcp.sandbox.runtime import owned_process_limit

_AVAILABLE = sys.platform == "linux"
if not _AVAILABLE:
    _reason = "Linux bubblewrap requires Linux"
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


async def test_missing_read_only_denials_stay_hidden_as_they_come_and_go(workspace: Workspace) -> None:
    (workspace.readonly / "data").write_text("visible")
    secret = workspace.readonly / "secret"
    token = workspace.readonly / "config" / "token"
    sandbox = LinuxSandbox(workspace.profile(denied_paths=(secret, token)))
    listing = f"ls -A {workspace.readonly} && cat {workspace.readonly}/data"

    result = await sandbox.run(listing)
    assert result.exit_code == 0, result
    assert result.stdout == "data\nvisible"
    for command in (f"printf leaked > {secret}", f"mkdir {token.parent}"):
        _denied(await sandbox.run(command))
    assert sorted(path.name for path in workspace.readonly.iterdir()) == ["data"]

    secret.write_text("secret-sentinel")
    token.parent.mkdir()
    token.write_text("secret-sentinel")
    for command in (f"cat {secret}", f"cat {token}", f"cp {secret} copy", f"rm {secret}"):
        _denied(await sandbox.run(command))
    assert secret.read_text() == "secret-sentinel"

    secret.unlink()
    token.unlink()
    token.parent.rmdir()
    result = await sandbox.run(listing)
    assert result.exit_code == 0, result
    assert result.stdout == "data\nvisible"


def _create_when_read(ready: Path, secret: Path) -> None:
    """Create ``secret`` once a command is blocked reading ``ready``, then let it continue."""
    deadline = time.monotonic() + 5
    while True:
        try:
            descriptor = os.open(ready, os.O_WRONLY | os.O_NONBLOCK)
        except OSError as exc:
            if exc.errno != errno.ENXIO or time.monotonic() > deadline:
                raise
            time.sleep(0.01)
        else:
            break
    try:
        secret.write_text("secret-sentinel")
        os.write(descriptor, b"go\n")
    finally:
        os.close(descriptor)


async def test_read_only_denial_created_while_running_is_covered_from_next_command(workspace: Workspace) -> None:
    secret = workspace.readonly / "secret"
    ready = workspace.writable / "ready"
    os.mkfifo(ready)
    sandbox = LinuxSandbox(workspace.profile(denied_paths=(secret,)))
    running = asyncio.create_task(sandbox.run(f"read line < ready && ls -A {workspace.readonly} && cat {secret}"))
    await asyncio.to_thread(_create_when_read, ready, secret)
    result = await running
    assert result.exit_code == 0, result
    assert result.stdout == "secret\nsecret-sentinel"
    _denied(await sandbox.run(f"cat {secret}"))


async def test_denied_fifos_and_sockets_are_covered(workspace: Workspace) -> None:
    fifo = workspace.readonly / "fifo"
    os.mkfifo(fifo)
    address = workspace.readonly / "socket"
    with socket.socket(socket.AF_UNIX) as server:
        server.bind(str(address))
        server.listen()
        profile = workspace.profile(denied_paths=(fifo, address))
        for command in (f"test -p {fifo}", f"test -S {address}", f"cat {fifo}", f"cat {address}"):
            _denied(await _run(profile, command, timeout_seconds=5))


async def test_writable_denial_created_by_a_command_is_covered_from_the_next(workspace: Workspace) -> None:
    secret = workspace.writable / "config" / "secret"
    sandbox = LinuxSandbox(workspace.profile(denied_paths=(secret,)))
    # A command can create a missing denied path in a writable directory. Seatbelt denies reading it
    # at once; this backend covers it from the next command.
    created = await sandbox.run("mkdir config && printf secret-sentinel > config/secret")
    assert created.exit_code == 0, created
    for command in ("cat config/secret", "rm config/secret", "mv config moved", "ln config/secret alias"):
        _denied(await sandbox.run(command))
    assert secret.read_text() == "secret-sentinel"
    assert sorted(path.name for path in workspace.writable.iterdir()) == ["config"]


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
    assert values["PATH"].endswith("/usr/bin:/bin:/usr/sbin:/sbin")


async def test_workload_has_no_capabilities_and_cannot_gain_privileges(workspace: Workspace) -> None:
    probe = """import ctypes
class Header(ctypes.Structure):
    _fields_ = [('version', ctypes.c_uint32), ('pid', ctypes.c_int)]
class Data(ctypes.Structure):
    _fields_ = [('effective', ctypes.c_uint32), ('permitted', ctypes.c_uint32), ('inheritable', ctypes.c_uint32)]
libc = ctypes.CDLL(None)
header, data = Header(0x20080522, 0), (Data * 2)()
assert libc.capget(ctypes.byref(header), ctypes.byref(data)) == 0
assert all(entry.effective == entry.permitted == entry.inheritable == 0 for entry in data)
assert all(libc.prctl(23, capability, 0, 0, 0) <= 0 for capability in range(64))
assert all(libc.prctl(47, 1, capability, 0, 0) <= 0 for capability in range(64))
assert libc.prctl(39, 0, 0, 0, 0) == 1
print('unprivileged')
"""
    profile = SandboxProfile((SandboxRoot(workspace.readonly),), tools=("python3",))
    result = await _run(profile, f"python3 -I -S -c {shlex.quote(probe)}")
    assert result.exit_code == 0, result
    assert result.stdout == "unprivileged\n"


async def test_proc_is_not_mounted(workspace: Workspace) -> None:
    result = await _run(workspace.profile(tools=("cat",)), "cat /proc/self/status")
    _denied(result)


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


@pytest.mark.parametrize("command", ["ls", "/bin/ls", "localmcp-no-such-tool", '"$missing"'])
async def test_ungranted_or_dynamic_command_rejects_entire_command(workspace: Workspace, command: str) -> None:
    result = await _run(
        workspace.profile(tools=("cat",), optional_tools=("localmcp-no-such-tool",)), f"printf ran; {command}"
    )
    assert result.exit_code == 126, result
    assert result.stdout == ""
    assert result.stderr.startswith("localmcp:")


@pytest.mark.parametrize(
    "command",
    [
        "awk 'BEGIN { exit system(\"/bin/ls data\") }'",
        "find data -maxdepth 0 -exec /bin/ls {} +",
        "printf data | xargs /bin/ls",
    ],
)
async def test_listed_tools_can_execute_unlisted_programs(workspace: Workspace, command: str) -> None:
    # Tool arguments are outside the shared shell check's authority on Linux.
    (workspace.writable / "data").write_text("visible")
    result = await _run(workspace.profile(tools=("awk", "find", "xargs")), command)
    assert result.exit_code == 0, result
    assert result.stdout == "data\n"


@pytest.mark.parametrize(
    "command",
    ["cp /bin/ls copy && ./copy /", "printf '#!/bin/sh\\necho escaped\\n' > script && chmod +x script && ./script"],
)
async def test_workspace_executables_are_denied(workspace: Workspace, command: str) -> None:
    _denied(await _run(workspace.profile(tools=("cp", "chmod")), command), secret="escaped")


async def test_workload_exit_code_and_signal_are_reported(workspace: Workspace) -> None:
    result = await _run(workspace.profile(tools=()), "printf out; printf err >&2; exit 7")
    assert result == CommandResult(exit_code=7, stdout="out", stderr="err")
    signal_result = await _run(workspace.profile(tools=()), "kill -TERM $$")
    assert signal_result.exit_code == 128 + 15, signal_result


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


async def test_process_limit_bounds_forks_even_with_other_host_processes(workspace: Workspace) -> None:
    holders = [subprocess.Popen(["/usr/bin/sleep", "30"]) for _ in range(16)]
    try:
        # Limits are applied before bwrap exec, so allow the processes already owned
        # by this UID on the host as well as the sandbox's additional allowance.
        maximum = owned_process_limit()
        assert maximum is not None, "the host needs process-limit headroom to exercise bounded forks"
        result = await _run(workspace.profile(tools=("python3",)), f"python3 -I -S -c {shlex.quote(_FORKS)}")
    finally:
        for holder in holders:
            holder.kill()
            holder.wait()
    assert result.exit_code == 0 and result.stdout.strip(), result
    assert 0 < int(result.stdout) <= maximum


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


async def test_network_grant_reads_the_hosts_file(workspace: Workspace) -> None:
    result = await _run(workspace.profile(tools=("getent",), network=True), "getent hosts localhost")
    assert result.exit_code == 0 and "localhost" in result.stdout, result


async def test_network_grant_resolves_names_through_dns(workspace: Workspace) -> None:
    # The hosts file does not list this name, so only the dns source can answer it.
    name = "example.com"
    try:
        await asyncio.to_thread(socket.getaddrinfo, name, None)
    except socket.gaierror:
        pytest.skip(f"the host cannot resolve {name}")
    result = await _run(workspace.profile(tools=("getent",), network=True), f"getent hosts {name}")
    assert result.exit_code == 0 and name in result.stdout, result


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
    _host_git(workspace.writable, "init", "-q")
    result = await _run(
        workspace.profile(tools=("git",)), "printf tracked > tracked; git add tracked; git diff --cached --stat"
    )
    assert result.exit_code == 0, result
    assert "tracked" in result.stdout
    assert (workspace.writable / ".git" / "index").is_file()


def _host_git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-c", "user.name=Tester", "-c", "user.email=tester@example.com", *arguments],
        cwd=root,
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(root), "GIT_CONFIG_NOSYSTEM": "1"},
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


@pytest.fixture
def seeded_repository(workspace: Workspace) -> Path:
    root = workspace.readonly
    _host_git(root, "init", "-q")
    (root / "review.txt").write_text("first\nsecond\n")
    _host_git(root, "add", "review.txt")
    _host_git(root, "commit", "-qm", "initial")
    (root / "review.txt").write_text("first\nsecond\nthird\n")
    _host_git(root, "commit", "-qam", "update")
    return root


async def test_review_commands_use_the_real_backend(seeded_repository: Path) -> None:
    sandbox = LinuxSandbox(
        SandboxProfile((SandboxRoot(seeded_repository),), tools=(*INSPECTION_TOOLS, "git"), optional_tools=("rg",))
    )
    probes = [
        ("git log --oneline -2", "initial"),
        ("git show --stat HEAD", "review.txt"),
        ("git diff HEAD~1 --stat", "review.txt"),
        ("git blame -L1,2 review.txt", "second"),
        # Git otherwise starts one grep worker per CPU; hosts with hundreds of
        # CPUs can exhaust the sandbox's process allowance for this tiny fixture.
        ("git -c grep.threads=2 grep -n second", "review.txt:2:second"),
        ("git ls-files | xargs wc -l", "3 review.txt"),
        ("find . -name '*.txt' -not -path './.git/*'", "./review.txt"),
        ("sed -n 2p review.txt; grep -c first review.txt; awk 'NR==1' review.txt | sort | uniq -c", "second"),
        ('files=$(git ls-files); printf "%s" "$files"', "review.txt"),
        ("file review.txt", "ASCII text"),
    ]
    if "rg" in sandbox.tools:
        probes.append(("rg -n second review.txt", "2:second"))
    for command, expected in probes:
        result = await sandbox.run(command)
        assert result.exit_code == 0, (command, result)
        assert expected in result.stdout, (command, result)
    status = await sandbox.run("git status --porcelain=v1")
    assert status == CommandResult(exit_code=0, stdout="", stderr="")


async def test_managed_worktree_requires_explicit_common_git_root(
    seeded_repository: Path, workspace: Workspace
) -> None:
    worktree = workspace.outside / "managed"
    _host_git(seeded_repository, "worktree", "add", "-qb", "review", str(worktree))
    command = "git log --oneline -1 && git blame -L1,2 review.txt && git status --porcelain=v1"
    _denied(await _run(SandboxProfile((SandboxRoot(worktree),), tools=("git",)), command))

    result = await _run(
        SandboxProfile((SandboxRoot(worktree), SandboxRoot(seeded_repository / ".git")), tools=("git",)), command
    )
    assert result.exit_code == 0, result
    assert "update" in result.stdout and "second" in result.stdout


async def test_workload_preserves_the_callers_uid(workspace: Workspace) -> None:
    result = await _run(workspace.profile(tools=("id",)), "id -u")
    assert result.exit_code == 0, result
    assert result.stdout.strip() == str(os.getuid())


async def test_startup_timing(workspace: Workspace, record_property: Callable[[str, object], None]) -> None:
    sandbox = LinuxSandbox(workspace.profile(tools=()))
    started = time.monotonic()
    for _ in range(5):
        result = await sandbox.run("true")
        assert result == CommandResult(exit_code=0, stdout="", stderr="")
    elapsed = time.monotonic() - started
    record_property("five_sandbox_runs_seconds", elapsed)
    assert elapsed < 10
