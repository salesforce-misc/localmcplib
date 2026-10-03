"""End-to-end enforcement tests against the real macOS Seatbelt sandbox.

These tests run real commands under ``/usr/bin/sandbox-exec`` with no mocking of
OS boundaries. They skip on platforms without Seatbelt unless
``LOCALMCP_REQUIRE_SEATBELT=1`` is set, in which case collection fails loudly so
a CI job dedicated to enforcement cannot pass with every test skipped.
"""

from __future__ import annotations

import asyncio
import http.server
import os
import socket
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
    WRITE_TOOLS,
    CommandResult,
    RootAccess,
    SandboxError,
    SandboxProfile,
    SandboxRoot,
    seatbelt,
)
from localmcp.sandbox.seatbelt import MacOSSandbox, _developer_directory, _homebrew_prefix, _homebrew_tool

_SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")
_SEATBELT_AVAILABLE = sys.platform == "darwin" and _SANDBOX_EXEC.is_file()

if not _SEATBELT_AVAILABLE:
    _reason = f"macOS Seatbelt unavailable (platform={sys.platform}, {_SANDBOX_EXEC} present={_SANDBOX_EXEC.is_file()})"
    if os.environ.get("LOCALMCP_REQUIRE_SEATBELT") == "1":
        pytest.fail(f"LOCALMCP_REQUIRE_SEATBELT=1 but {_reason}", pytrace=False)
    pytest.skip(_reason, allow_module_level=True)

pytestmark = pytest.mark.seatbelt

_CURL = "/usr/bin/curl"
_NC = "/usr/bin/nc"
_PERL = "/usr/bin/perl"
# Lets the filesystem tests read, list, link, rename and remove files.
_TOOLS = (*INSPECTION_TOOLS, *WRITE_TOOLS)


@dataclass(frozen=True)
class Workspace:
    """Host directories used as sandbox roots plus a sibling outside every root."""

    read_write: Path
    read_only: Path
    outside: Path

    def profile(
        self,
        *,
        denied_paths: tuple[Path, ...] = (),
        network: bool = False,
        tools: tuple[str, ...] = _TOOLS,
        optional_tools: tuple[str, ...] = (),
    ) -> SandboxProfile:
        return SandboxProfile(
            roots=(
                SandboxRoot(self.read_write, RootAccess.READ_WRITE),
                SandboxRoot(self.read_only, RootAccess.READ_ONLY),
            ),
            denied_paths=denied_paths,
            network=network,
            tools=tools,
            optional_tools=optional_tools,
        )


@pytest.fixture
def workspace(tmp_path: Path) -> Workspace:
    base = tmp_path.resolve()
    directories = Workspace(base / "rw", base / "ro", base / "outside")
    for directory in (directories.read_write, directories.read_only, directories.outside):
        directory.mkdir()
    return directories


async def _run(
    profile: SandboxProfile, command: str, *, timeout_seconds: float = 10, max_output_bytes: int = 1024 * 1024
) -> CommandResult:
    sandbox = MacOSSandbox(profile, timeout_seconds=timeout_seconds, max_output_bytes=max_output_bytes)
    return await sandbox.run(command)


def _assert_denied(result: CommandResult, *, leaked: str | None = None) -> None:
    assert result.exit_code != 0, result
    assert not result.timed_out
    if leaked is not None:
        assert leaked not in result.stdout
        assert leaked not in result.stderr


# Filesystem: baseline allowances


async def test_reads_file_in_read_only_root(workspace: Workspace) -> None:
    (workspace.read_only / "data.txt").write_text("read-only-content\n")

    result = await _run(workspace.profile(), f"cat {workspace.read_only / 'data.txt'}")

    assert result.exit_code == 0, result
    assert result.stdout == "read-only-content\n"


async def test_writes_file_in_read_write_root_visible_on_host(workspace: Workspace) -> None:
    result = await _run(workspace.profile(), "printf written > created.txt")

    assert result.exit_code == 0, result
    assert (workspace.read_write / "created.txt").read_text() == "written"


async def test_runs_with_first_root_as_working_directory(workspace: Workspace) -> None:
    result = await _run(workspace.profile(), "pwd")

    assert result.exit_code == 0, result
    assert result.stdout.strip() == str(workspace.read_write)


# Filesystem: write denials


async def test_write_to_read_only_root_is_denied(workspace: Workspace) -> None:
    target = workspace.read_only / "blocked.txt"

    result = await _run(workspace.profile(), f"printf nope > {target}")

    _assert_denied(result)
    assert not target.exists()


async def test_modifying_existing_file_in_read_only_root_is_denied(workspace: Workspace) -> None:
    target = workspace.read_only / "existing.txt"
    target.write_text("original")

    result = await _run(workspace.profile(), f"printf changed > {target} || rm -f {target}")

    _assert_denied(result)
    assert target.read_text() == "original"


async def test_write_outside_every_root_is_denied(workspace: Workspace) -> None:
    target = workspace.outside / "escaped.txt"

    result = await _run(workspace.profile(), f"printf nope > {target}")

    _assert_denied(result)
    assert not target.exists()


async def test_write_via_tmp_is_denied(workspace: Workspace) -> None:
    marker = f"localmcp-seatbelt-{os.getpid()}-{time.monotonic_ns()}"
    target = Path("/private/tmp") / marker

    result = await _run(workspace.profile(), f"printf nope > {target}")

    _assert_denied(result)
    assert not target.exists()


# Filesystem: read denials


async def test_read_outside_every_root_is_denied(workspace: Workspace) -> None:
    secret = workspace.outside / "secret.txt"
    secret.write_text("outside-secret-value")

    result = await _run(workspace.profile(), f"cat {secret}")

    _assert_denied(result, leaked="outside-secret-value")


async def test_listing_directory_outside_every_root_is_denied(workspace: Workspace) -> None:
    (workspace.outside / "hidden-name.txt").write_text("x")

    result = await _run(workspace.profile(), f"ls {workspace.outside}")

    _assert_denied(result, leaked="hidden-name.txt")


async def test_listing_real_home_directory_is_denied(workspace: Workspace) -> None:
    home = Path.home().resolve()
    if not home.is_dir() or any(home.is_relative_to(root) for root in (workspace.read_write, workspace.read_only)):
        pytest.skip("home directory unavailable or overlaps a sandbox root")

    result = await _run(workspace.profile(), f"ls -a {home}")

    _assert_denied(result)
    assert result.stdout == ""


async def test_symlink_in_root_to_file_outside_roots_cannot_be_read(workspace: Workspace) -> None:
    secret = workspace.outside / "secret.txt"
    secret.write_text("symlinked-secret-value")
    (workspace.read_write / "link.txt").symlink_to(secret)

    result = await _run(workspace.profile(), "cat link.txt")

    _assert_denied(result, leaked="symlinked-secret-value")


async def test_symlink_in_root_to_directory_outside_roots_cannot_be_listed(workspace: Workspace) -> None:
    (workspace.outside / "secret.txt").write_text("symlinked-dir-secret")
    (workspace.read_write / "linkdir").symlink_to(workspace.outside, target_is_directory=True)

    result = await _run(workspace.profile(), "cat linkdir/secret.txt")

    _assert_denied(result, leaked="symlinked-dir-secret")


async def test_sandbox_created_symlink_cannot_escape_roots(workspace: Workspace) -> None:
    secret = workspace.outside / "secret.txt"
    secret.write_text("created-link-secret")

    result = await _run(workspace.profile(), f"ln -s {secret} made.txt && cat made.txt")

    _assert_denied(result, leaked="created-link-secret")


# Filesystem: denied paths inside roots


@pytest.fixture
def denied_layout(workspace: Workspace) -> tuple[Path, Path, Path]:
    """Return (denied file, file inside denied directory, allowed sibling) inside the read-write root."""
    denied_file = workspace.read_write / "credentials.txt"
    denied_file.write_text("denied-file-secret")
    denied_directory = workspace.read_write / "private"
    denied_directory.mkdir()
    nested = denied_directory / "key.txt"
    nested.write_text("denied-dir-secret")
    allowed = workspace.read_write / "public.txt"
    allowed.write_text("public-content")
    return denied_file, nested, allowed


@pytest.fixture
def denied_profile(workspace: Workspace, denied_layout: tuple[Path, Path, Path]) -> SandboxProfile:
    denied_file, nested, _ = denied_layout
    return workspace.profile(denied_paths=(denied_file, nested.parent))


async def test_denied_file_inside_root_cannot_be_read(
    denied_profile: SandboxProfile, denied_layout: tuple[Path, Path, Path]
) -> None:
    result = await _run(denied_profile, f"cat {denied_layout[0].name}")

    _assert_denied(result, leaked="denied-file-secret")


async def test_file_inside_denied_directory_cannot_be_read(
    denied_profile: SandboxProfile, denied_layout: tuple[Path, Path, Path]
) -> None:
    result = await _run(denied_profile, "cat private/key.txt")

    _assert_denied(result, leaked="denied-dir-secret")


async def test_denied_directory_cannot_be_listed(
    denied_profile: SandboxProfile, denied_layout: tuple[Path, Path, Path]
) -> None:
    result = await _run(denied_profile, "ls private")

    _assert_denied(result, leaked="key.txt")


async def test_sibling_of_denied_paths_remains_readable(
    denied_profile: SandboxProfile, denied_layout: tuple[Path, Path, Path]
) -> None:
    result = await _run(denied_profile, f"cat {denied_layout[2].name}")

    assert result.exit_code == 0, result
    assert result.stdout == "public-content"


async def test_symlink_to_denied_file_cannot_be_read(
    denied_profile: SandboxProfile, denied_layout: tuple[Path, Path, Path]
) -> None:
    result = await _run(denied_profile, f"ln -s {denied_layout[0]} alias.txt && cat alias.txt")

    _assert_denied(result, leaked="denied-file-secret")


async def test_hard_link_to_denied_file_cannot_be_read(
    denied_profile: SandboxProfile, denied_layout: tuple[Path, Path, Path]
) -> None:
    result = await _run(denied_profile, f"ln {denied_layout[0].name} hard.txt && cat hard.txt")

    _assert_denied(result, leaked="denied-file-secret")


@pytest.mark.parametrize("linked", [0, 1], ids=["denied-file", "file-in-denied-directory"])
async def test_existing_hard_link_to_denied_file_fails_closed(
    workspace: Workspace, denied_layout: tuple[Path, Path, Path], linked: int
) -> None:
    # Seatbelt denies paths, so a link the host made earlier would expose the file under its other name.
    denied_file, nested, _ = denied_layout
    profile = workspace.profile(denied_paths=(denied_file, nested.parent))
    sandbox = MacOSSandbox(profile)
    os.link(denied_layout[linked], workspace.read_only / "alias.txt")

    with pytest.raises(SandboxError, match="another hard link"):
        await sandbox.run(f"cat {workspace.read_only / 'alias.txt'}")
    with pytest.raises(SandboxError, match="another hard link"):
        MacOSSandbox(profile)


async def test_hard_link_into_an_unreadable_denied_directory_fails_closed(
    workspace: Workspace, denied_layout: tuple[Path, Path, Path]
) -> None:
    # The check cannot see this link, which would leave the file readable under its other name.
    locked = denied_layout[1].parent / "locked"
    locked.mkdir()
    (locked / "key.txt").write_text("locked-secret")
    os.link(locked / "key.txt", workspace.read_only / "alias.txt")
    locked.chmod(0o000)
    try:
        with pytest.raises(SandboxError, match="cannot be checked for hard links"):
            MacOSSandbox(workspace.profile(denied_paths=(locked.parent,)))
    finally:
        locked.chmod(0o755)


async def test_denied_path_through_a_symlink_fails_closed(
    workspace: Workspace, denied_layout: tuple[Path, Path, Path]
) -> None:
    # Seatbelt matches the real path a file is opened by, so denying the link's path would hide nothing.
    (workspace.read_write / "link").symlink_to(denied_layout[1].parent)

    with pytest.raises(SandboxError, match="goes through a symlink"):
        MacOSSandbox(workspace.profile(denied_paths=(workspace.read_write / "link" / "key.txt",)))


@pytest.mark.parametrize(
    ("command", "leaked"),
    [
        pytest.param("mv credentials.txt renamed.txt && cat renamed.txt", "denied-file-secret", id="denied-file"),
        pytest.param("mv private exposed && cat exposed/key.txt", "denied-dir-secret", id="denied-directory"),
        pytest.param(
            "mv private/key.txt moved.txt && cat moved.txt", "denied-dir-secret", id="file-in-denied-directory"
        ),
    ],
)
async def test_renaming_denied_path_in_read_write_root_does_not_expose_it(
    denied_profile: SandboxProfile, denied_layout: tuple[Path, Path, Path], command: str, leaked: str
) -> None:
    result = await _run(denied_profile, command)

    _assert_denied(result, leaked=leaked)


async def test_renaming_ancestor_of_denied_path_does_not_expose_it(workspace: Workspace) -> None:
    denied = workspace.read_write / "config" / "secrets" / "token.txt"
    denied.parent.mkdir(parents=True)
    denied.write_text("nested-secret")

    result = await _run(workspace.profile(denied_paths=(denied,)), "mv config exposed && cat exposed/secrets/token.txt")

    _assert_denied(result, leaked="nested-secret")


async def test_paths_unrelated_to_denied_paths_can_be_renamed(
    denied_profile: SandboxProfile, denied_layout: tuple[Path, Path, Path]
) -> None:
    result = await _run(denied_profile, "mv public.txt renamed.txt && mkdir d && mv d e && cat renamed.txt")

    assert result.exit_code == 0, result
    assert result.stdout == "public-content"


# Network


@pytest.fixture
def listening_port() -> Iterator[int]:
    """A loopback TCP listener; the kernel backlog completes handshakes without accept()."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.bind(("127.0.0.1", 0))
        server.listen(8)
        yield server.getsockname()[1]


@pytest.mark.skipif(not Path(_NC).is_file(), reason="/usr/bin/nc unavailable")
async def test_outbound_connection_denied_without_network(workspace: Workspace, listening_port: int) -> None:
    result = await _run(workspace.profile(network=False, tools=("nc",)), f"{_NC} -n -z -w 2 127.0.0.1 {listening_port}")

    _assert_denied(result)


@pytest.mark.skipif(not Path(_NC).is_file(), reason="/usr/bin/nc unavailable")
async def test_outbound_connection_allowed_with_network(workspace: Workspace, listening_port: int) -> None:
    result = await _run(workspace.profile(network=True, tools=("nc",)), f"{_NC} -n -z -w 2 127.0.0.1 {listening_port}")

    assert result.exit_code == 0, result


class _OkHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:
        pass


@pytest.fixture
def http_port() -> Iterator[int]:
    """A loopback HTTP server answering every request with an empty 200."""
    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), _OkHandler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server.server_address[1]
        finally:
            server.shutdown()
            thread.join()


@pytest.mark.skipif(not Path(_CURL).is_file(), reason="/usr/bin/curl unavailable")
async def test_system_curl_works_with_network(workspace: Workspace, http_port: int) -> None:
    result = await _run(
        workspace.profile(network=True, tools=("curl",)),
        f"curl -sS -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{http_port}/",
    )

    assert result.exit_code == 0, result
    assert result.stdout == "200"


@pytest.mark.parametrize("network", [False, True], ids=["offline", "online"])
async def test_system_trust_store_is_readable_only_with_network(workspace: Workspace, network: bool) -> None:
    result = await _run(
        workspace.profile(network=network, tools=("grep",)), "grep -c 'BEGIN CERTIFICATE' /etc/ssl/cert.pem"
    )

    if network:
        assert result.exit_code == 0, result
        assert int(result.stdout) > 0
    else:
        _assert_denied(result)


# Environment


async def test_caller_environment_is_not_inherited(workspace: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCALMCP_TEST_SECRET", "caller-env-sentinel")

    result = await _run(workspace.profile(tools=("env",)), "env")

    assert result.exit_code == 0, result
    assert "LOCALMCP_TEST_SECRET" not in result.stdout
    assert "caller-env-sentinel" not in result.stdout
    environment = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    assert environment["HOME"] == "/var/empty"
    assert environment["PATH"] == "/usr/bin:/bin:/usr/sbin:/sbin"


async def test_each_command_gets_a_private_tmpdir_that_is_removed(workspace: Workspace) -> None:
    sandbox = MacOSSandbox(workspace.profile())
    command = 'test ! -e "$TMPDIR/marker" && printf x > "$TMPDIR/marker" && printf %s "$TMPDIR"'

    first = await sandbox.run(command)
    second = await sandbox.run(command)

    assert first.exit_code == 0, first
    assert second.exit_code == 0, second
    assert first.stdout != second.stdout
    for scratch in (Path(first.stdout), Path(second.stdout)):
        assert not any(scratch.is_relative_to(root) for root in (workspace.read_write, workspace.read_only))
        assert not scratch.exists()


async def test_tmpdir_is_read_only_without_a_read_write_root(workspace: Workspace) -> None:
    # Nothing bounds what a command writes, so a read-only profile gets no space to fill.
    sandbox = MacOSSandbox(SandboxProfile((SandboxRoot(workspace.read_only, RootAccess.READ_ONLY),)))

    result = await sandbox.run('test -d "$TMPDIR" && printf x > "$TMPDIR/marker"')

    _assert_denied(result)
    assert "Operation not permitted" in result.stderr


# Commands


async def test_only_builtins_and_listed_tools_run(workspace: Workspace) -> None:
    (workspace.read_write / "data.txt").write_text("listed\n")

    result = await _run(workspace.profile(tools=("cat",)), "cat data.txt && echo builtin && ls; echo $?")

    assert result.exit_code == 0, result
    assert result.stdout == "listed\nbuiltin\n126\n"
    assert "/bin/ls: Operation not permitted" in result.stderr


@pytest.mark.parametrize(
    "command",
    [
        pytest.param("awk 'BEGIN { exit system(\"/bin/ls /\") }'", id="awk"),
        pytest.param("find / -maxdepth 0 -exec /bin/ls {} +", id="find"),
        pytest.param("echo / | xargs /bin/ls", id="xargs"),
    ],
)
async def test_listed_tools_cannot_run_unlisted_commands(workspace: Workspace, command: str) -> None:
    result = await _run(workspace.profile(tools=("awk", "find", "xargs")), command)

    _assert_denied(result, leaked="Library")
    assert "Operation not permitted" in result.stderr


@pytest.mark.parametrize(
    "command",
    [
        pytest.param("cp /bin/ls copy && ./copy /", id="copied-binary"),
        pytest.param("printf '#!/bin/sh\\necho ran\\n' > script && chmod +x script && ./script", id="written-script"),
    ],
)
async def test_programs_in_a_read_write_root_cannot_run(workspace: Workspace, command: str) -> None:
    result = await _run(workspace.profile(tools=("chmod", "cp")), command)

    _assert_denied(result, leaked="ran")
    assert "Library" not in result.stdout
    assert "Operation not permitted" in result.stderr


async def test_script_tools_run_their_interpreter_only_for_themselves(workspace: Workspace) -> None:
    shasum = Path("/usr/bin/shasum")
    if not shasum.is_file() or not shasum.read_bytes().startswith(b"#!/usr/bin/perl"):
        pytest.skip("/usr/bin/shasum is not a perl script")

    result = await _run(workspace.profile(tools=("shasum",)), "printf x | shasum && perl -e 'print qq(ran)'")

    assert result.exit_code == 126, result
    assert result.stdout == "11f6ad8ec52a2984abaafd7c3b516503785c2072  -\n"
    assert "/usr/bin/perl: Operation not permitted" in result.stderr


# Developer tools

# The /usr/bin shims re-resolve the toolchain on every call, which is slow under the sandbox.
_TOOLCHAIN_TIMEOUT_SECONDS = 30


def _homebrew_tool_or_skip(*candidates: str) -> tuple[Path, str]:
    """Return the Homebrew prefix and the first candidate it installs as a formula tool."""
    prefix = _homebrew_prefix()
    if prefix is None:
        pytest.skip("Homebrew not installed")
    for tool in candidates:
        if _homebrew_tool(prefix, tool) is not None:
            return prefix, tool
    pytest.skip(f"Homebrew installs none of {candidates}")


@pytest.fixture
def developer(monkeypatch: pytest.MonkeyPatch) -> Path:
    """The selected Xcode or Command Line Tools install, with Homebrew hidden so it serves every tool."""
    selected = _developer_directory()
    if selected is None:
        pytest.skip("no Xcode or Command Line Tools selected")
    monkeypatch.setattr(seatbelt, "_HOMEBREW_PREFIXES", ())
    return selected


@pytest.fixture(params=["homebrew", "developer"])
def git_bin(request: pytest.FixtureRequest) -> Path:
    """Run git tests with Homebrew's git, then with the developer toolchain's.

    Returns the directory the sandbox is expected to run git from.
    """
    if request.param == "homebrew":
        prefix, tool = _homebrew_tool_or_skip("git")
        return (prefix / "bin" / tool).resolve().parent
    selected = request.getfixturevalue("developer")
    assert isinstance(selected, Path)
    return selected / "usr" / "bin"


def _host_git(cwd: Path, *args: str) -> None:
    environment = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(cwd),
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    if "DEVELOPER_DIR" in os.environ:
        environment["DEVELOPER_DIR"] = os.environ["DEVELOPER_DIR"]
    subprocess.run(
        ["git", "-c", "user.email=t@e.st", "-c", "user.name=Tester", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        env=environment,
    )


@pytest.fixture
def seeded_repository(workspace: Workspace) -> Path:
    """A one-commit repository in the read-only root."""
    (workspace.read_only / "file.txt").write_text("first\nsecond\n")
    _host_git(workspace.read_only, "init", "-q")
    _host_git(workspace.read_only, "add", "file.txt")
    _host_git(workspace.read_only, "commit", "-qm", "seed")
    return workspace.read_only


async def test_git_works_inside_read_write_root(workspace: Workspace, git_bin: Path) -> None:
    result = await _run(
        workspace.profile(tools=("git",)),
        "command -v git && git init -q && git status --porcelain=v1 --branch",
        timeout_seconds=_TOOLCHAIN_TIMEOUT_SECONDS,
    )

    assert result.exit_code == 0, result
    located, status = result.stdout.split("\n", 1)
    assert Path(located) == git_bin / "git"
    assert status.startswith("## ")
    assert (workspace.read_write / ".git" / "HEAD").is_file()


async def test_git_reads_history_in_read_only_root(
    workspace: Workspace, seeded_repository: Path, git_bin: Path
) -> None:
    result = await _run(
        workspace.profile(tools=("git",)),
        f"cd {seeded_repository} && git log --oneline && git blame file.txt",
        timeout_seconds=_TOOLCHAIN_TIMEOUT_SECONDS,
    )

    assert result.exit_code == 0, result
    assert "seed" in result.stdout
    assert "first" in result.stdout and "second" in result.stdout


async def test_git_dispatches_exec_helpers(workspace: Workspace, seeded_repository: Path, git_bin: Path) -> None:
    # git finds these in its exec path (git-core), never on PATH: file:// transport execs
    # git-upload-pack, and request-pull is a script that exists only there.
    result = await _run(
        workspace.profile(tools=("git",)),
        f"git ls-remote file://{seeded_repository} && git clone -q file://{seeded_repository} clone"
        " && { git request-pull 2>&1; true; }",
        timeout_seconds=_TOOLCHAIN_TIMEOUT_SECONDS,
    )

    assert result.exit_code == 0, result
    assert "HEAD" in result.stdout
    assert (workspace.read_write / "clone" / "file.txt").read_text() == "first\nsecond\n"
    assert "is not a git command" not in result.stdout
    assert "not a git repository" in result.stdout


async def test_tools_ignore_tools_planted_on_the_caller_path(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    planted = workspace.read_only / "bin"
    planted.mkdir()
    (planted / "git").write_text("#!/bin/sh\necho planted\n")
    (planted / "git").chmod(0o755)
    monkeypatch.setenv("PATH", f"{planted}{os.pathsep}{os.environ.get('PATH', '/usr/bin:/bin')}")

    result = await _run(workspace.profile(tools=("git",)), 'command -v git; printf %s "$PATH"')

    assert result.exit_code == 0, result
    assert str(planted) not in result.stdout


@pytest.mark.parametrize("decoy", ["homebrew", "developer"])
async def test_caller_environment_cannot_widen_tool_grants(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, decoy: str
) -> None:
    # A git install outside every root, selected the way a caller's environment would select it.
    if decoy == "homebrew":
        prefix = workspace.outside / "homebrew"
        install = prefix / "Cellar" / "git" / "1.0"
        (prefix / "bin").mkdir(parents=True)
        (prefix / "bin" / "brew").write_text("#!/bin/sh\n")
        (prefix / "bin" / "brew").chmod(0o755)
        (prefix / "bin" / "git").symlink_to(install / "bin" / "git")
        (install / "bin").mkdir(parents=True)
        (install / "INSTALL_RECEIPT.json").write_text("{}")
        monkeypatch.setenv("HOMEBREW_PREFIX", str(prefix))
    else:
        install = workspace.outside / "Developer"
        (install / "usr" / "bin").mkdir(parents=True)
        monkeypatch.setenv("DEVELOPER_DIR", str(install))
        # Let the developer directory serve git, looked up afresh rather than from the cache.
        monkeypatch.setattr(seatbelt, "_HOMEBREW_PREFIXES", ())
        monkeypatch.setattr(seatbelt, "_developer_directory", _developer_directory.__wrapped__)
    decoy_git = install / "bin" / "git" if decoy == "homebrew" else install / "usr" / "bin" / "git"
    decoy_git.write_text("#!/bin/sh\necho decoy\n")
    decoy_git.chmod(0o755)
    (install / "canary.txt").write_text("canary\n")

    sandbox = MacOSSandbox(workspace.profile(tools=("cat", "git")))
    result = await sandbox.run(f"command -v git; cat {install / 'canary.txt'} 2>/dev/null; true")

    assert result.exit_code == 0, result
    assert not any(tree.is_relative_to(workspace.outside) for tree in sandbox._toolchain_trees)
    assert str(workspace.outside) not in result.stdout
    assert "canary" not in result.stdout


async def test_tools_expose_only_their_own_homebrew_formulae(workspace: Workspace) -> None:
    prefix, tool = _homebrew_tool_or_skip("rg", "git")
    sandbox = MacOSSandbox(workspace.profile(tools=("ls", tool)))
    granted = set(sandbox._toolchain_trees)
    others = sorted(
        version
        for formula in (prefix / "Cellar").iterdir()
        for version in formula.iterdir()
        if version.is_dir() and version.resolve() not in granted
    )
    hidden = [prefix / name for name in ("bin", "opt", "Cellar", "lib", "share", "etc", "var", "Library")]
    hidden += [path for path in (prefix / "share" / "man", prefix / "etc" / "openssl@3", *others[:5]) if path.exists()]
    probes = [f"{tool} --version >/dev/null && echo ran"]
    probes.extend(f"ls {path} >/dev/null 2>&1 && echo readable:{path}" for path in hidden)

    result = await sandbox.run("; ".join(probes) + "; true")

    assert result.exit_code == 0, result
    assert result.stdout.startswith("ran\n")
    assert "readable:" not in result.stdout


async def test_installed_optional_tools_run_beside_missing_ones(workspace: Workspace, git_bin: Path) -> None:
    sandbox = MacOSSandbox(
        workspace.profile(tools=(), optional_tools=("localmcp-absent-tool", "git")),
        timeout_seconds=_TOOLCHAIN_TIMEOUT_SECONDS,
    )

    result = await sandbox.run("command -v git && git --version")

    assert sandbox.tools == ("git",)
    assert result.exit_code == 0, result
    located, version = result.stdout.split("\n", 1)
    assert Path(located) == git_bin / "git"
    assert version.startswith("git version ")


async def test_developer_tools_cannot_run_each_others_helpers(workspace: Workspace, developer: Path) -> None:
    daemon = developer / "usr" / "libexec" / "git-core" / "git-daemon"
    if not (developer / "usr" / "bin" / "clang").is_file() or not daemon.is_file():
        pytest.skip("selected developer directory has no clang or git-daemon")

    result = await _run(
        workspace.profile(tools=("clang",)), f"{daemon} --version", timeout_seconds=_TOOLCHAIN_TIMEOUT_SECONDS
    )

    _assert_denied(result)
    assert f"{daemon}: Operation not permitted" in result.stderr


async def test_shims_are_not_tools_without_a_developer_install(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not Path("/usr/bin/git").is_file():
        pytest.skip("/usr/bin/git unavailable")
    monkeypatch.setattr(seatbelt, "_HOMEBREW_PREFIXES", ())
    monkeypatch.setattr(seatbelt, "_developer_directory", lambda: None)

    sandbox = MacOSSandbox(workspace.profile(tools=(), optional_tools=("git",)))

    assert sandbox.tools == ()


async def test_tools_run_toolchain_python_through_the_shim(workspace: Workspace, developer: Path) -> None:
    if not (developer / "usr" / "bin" / "python3").is_file():
        pytest.skip("selected developer directory has no python3")

    result = await _run(
        workspace.profile(tools=("python3",)),
        "/usr/bin/python3 -c 'import json, sqlite3; print(json.dumps(sqlite3.sqlite_version_info[0]))'",
        timeout_seconds=_TOOLCHAIN_TIMEOUT_SECONDS,
    )

    assert result.exit_code == 0, result
    assert result.stdout.strip() == "3"
    # The shim's libxcrun caches lookups in TMPDIR and reports when it cannot.
    assert "xcrun" not in result.stderr


def _granting_tools() -> tuple[str, ...]:
    """Return the tools whose installed code is outside the base system on this host."""
    tools = []
    for tool in ("git", "rg", "clang"):
        try:
            if MacOSSandbox(SandboxProfile((SandboxRoot(Path("/var/empty")),), tools=(tool,)))._toolchain_trees:
                tools.append(tool)
        except SandboxError:
            continue
    if not tools:
        pytest.skip("no developer tools installed")
    return tuple(tools)


async def test_tools_grant_no_writes_to_the_toolchains(workspace: Workspace) -> None:
    sandbox = MacOSSandbox(workspace.profile(tools=_granting_tools()))
    marker = f"localmcp-seatbelt-{os.getpid()}-{time.monotonic_ns()}"
    targets = [tree / marker for tree in sandbox._toolchain_trees]

    try:
        result = await sandbox.run("; ".join(f"printf nope > {target}" for target in targets))
    finally:
        leaked = [target for target in targets if target.exists()]
        for target in leaked:
            target.unlink()

    assert not result.timed_out
    assert leaked == []


async def test_toolchains_are_unreadable_without_tools(workspace: Workspace) -> None:
    trees = MacOSSandbox(workspace.profile(tools=_granting_tools()))._toolchain_trees

    result = await _run(
        workspace.profile(tools=("ls",)),
        "; ".join(f"ls {tree} >/dev/null 2>&1 && echo readable:{tree}" for tree in trees) + "; true",
    )

    assert result.exit_code == 0, result
    assert "readable:" not in result.stdout


# Resource limits


async def test_long_running_command_times_out_promptly(workspace: Workspace) -> None:
    started = time.monotonic()

    result = await _run(workspace.profile(tools=("sleep",)), "sleep 30", timeout_seconds=1)

    elapsed = time.monotonic() - started
    assert result.timed_out is True
    assert result.exit_code == -1
    assert "command timed out" in result.stderr
    assert elapsed < 10


async def test_excess_output_is_truncated_to_limit(workspace: Workspace) -> None:
    limit = 4096

    result = await _run(
        workspace.profile(tools=("head", "yes")), "yes localmcp | head -c 1000000", max_output_bytes=limit
    )

    assert result.truncated is True
    assert result.exit_code == -1
    assert len(result.stdout.encode()) == limit
    assert result.stdout == ("localmcp\n" * limit)[:limit]
    assert "command output exceeded its bounded safety limit" in result.stderr


# Process supervision

# Leaves a grandchild in its own session with its output streams closed, so neither the process group nor the
# pipes tie it to the command. It records that it started, then acts after the command has returned.
_DETACHED = (
    f"{_PERL} -e 'use POSIX; exit if fork; POSIX::setsid(); exit if fork; close STDIN; close STDOUT; close STDERR;"
    ' open my $f, ">", "started"; close $f; sleep 1; open $f, ">", "late"; close $f\';'
    " while [ ! -e started ]; do sleep 0.05; done"
)
_BACKGROUND = "(sleep 1; touch late) >/dev/null 2>&1 & touch started"
_SUPERVISION_TOOLS = ("perl", "seq", "sleep", "touch")


@pytest.mark.skipif(not Path(_PERL).is_file(), reason="/usr/bin/perl unavailable")
@pytest.mark.parametrize(
    ("command", "timed_out"),
    [(_DETACHED, False), (_DETACHED + "; sleep 30", True), (_BACKGROUND, False)],
    ids=["detached-after-exit", "detached-after-timeout", "background-after-exit"],
)
async def test_descendants_cannot_act_after_the_command_returns(
    workspace: Workspace, command: str, timed_out: bool
) -> None:
    result = await _run(workspace.profile(tools=_SUPERVISION_TOOLS), command, timeout_seconds=1)
    await asyncio.sleep(2)

    assert result.timed_out is timed_out, result
    assert (workspace.read_write / "started").exists()
    assert not (workspace.read_write / "late").exists()


async def test_supervision_spares_concurrent_commands_with_the_same_profile(workspace: Workspace) -> None:
    sandbox = MacOSSandbox(workspace.profile(tools=_SUPERVISION_TOOLS), timeout_seconds=10)
    running = asyncio.create_task(sandbox.run("sleep 1.5; echo survived"))
    await asyncio.sleep(0.3)

    for _ in range(2):
        assert (await sandbox.run("true")).exit_code == 0
    result = await running

    assert result.exit_code == 0, result
    assert result.stdout == "survived\n"


@pytest.mark.skipif(not Path(_PERL).is_file(), reason="/usr/bin/perl unavailable")
async def test_detached_fork_storm_is_stopped_at_the_timeout(workspace: Workspace) -> None:
    storm = (
        "for i in $(seq 1 40); do"
        f' ( {_PERL} -e \'use POSIX; POSIX::setsid(); fork; fork; sleep 2; open my $f, ">", "late"\' & );'
        " done; sleep 30"
    )
    started = time.monotonic()

    result = await _run(workspace.profile(tools=_SUPERVISION_TOOLS), storm, timeout_seconds=1)
    elapsed = time.monotonic() - started
    await asyncio.sleep(2.5)

    assert result.timed_out is True, result
    assert elapsed < 10
    assert not (workspace.read_write / "late").exists()
