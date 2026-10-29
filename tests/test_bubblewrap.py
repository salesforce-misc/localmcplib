"""Portable policy, discovery, argv, and supervision tests for the Linux backend."""

from __future__ import annotations

import asyncio
import json
import os
import pwd
import signal
from dataclasses import dataclass
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


@dataclass
class Host:
    path: Path
    root: Path

    def program(self, name: str, *, directory: Path | None = None) -> Path:
        program = (directory or self.path / "usr/bin") / name
        program.parent.mkdir(parents=True, exist_ok=True)
        program.write_text("#!/bin/sh\nexit 0\n")
        program.chmod(0o755)
        return program

    def sandbox(self, **kwargs: Any) -> LinuxSandbox:
        return LinuxSandbox(SandboxProfile((SandboxRoot(self.root, RootAccess.READ_WRITE),), tools=("cat",), **kwargs))


@pytest.fixture
def host(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Host:
    host = Host(tmp_path / "host", tmp_path / "root")
    host.root.mkdir()
    for directory in ("usr/bin", "usr/lib", "lib", "nix/store", "etc", "home"):
        (host.path / directory).mkdir(parents=True, exist_ok=True)
    (host.path / "bin").symlink_to("usr/bin")
    (host.path / "etc/ld.so.cache").touch()
    (host.path / "etc/hosts").write_text("127.0.0.1 localhost\n")
    for program in ("bash", "bwrap", "cat"):
        host.program(program)
    monkeypatch.setattr(linux, "_require_linux", lambda: None)
    monkeypatch.setattr(linux, "_BIN_DIRECTORIES", (host.path / "usr/bin", host.path / "bin"))
    monkeypatch.setattr(linux, "_RUNTIME_TREES", (host.path / "usr", host.path / "nix/store"))
    monkeypatch.setattr(linux, "_RUNTIME_ALIASES", (host.path / "bin", host.path / "lib"))
    monkeypatch.setattr(linux, "_ETC_PATHS", (host.path / "etc/ld.so.cache",))
    monkeypatch.setattr(linux, "_NETWORK_PATHS", (host.path / "etc/hosts",))
    monkeypatch.setattr(linux, "_NIX_STORE", host.path / "nix/store")
    monkeypatch.setattr(linux, "_nix_profile_links", lambda: (host.path / "home/.nix-profile",))
    monkeypatch.setattr(linux, "find_library", lambda profiles: host.path / "lib/libseccomp.so.2")
    monkeypatch.setattr(linux, "build_filter", lambda *args, **kwargs: b"compiled-filter")
    monkeypatch.setattr(linux, "owned_process_limit", lambda: 300)
    monkeypatch.setattr(linux, "limited_command", lambda argv, _timeout, _maximum: argv)
    return host


@pytest.fixture
def backend(host: Host) -> LinuxSandbox:
    return host.sandbox()


def _operations(argv: list[str], option: str, arity: int = 2) -> list[tuple[str, ...]]:
    return [tuple(argv[index + 1 : index + 1 + arity]) for index, value in enumerate(argv) if value == option]


def test_backend_preserves_first_root_and_nested_protection(host: Host) -> None:
    protected = host.root / "docs/source"
    protected.mkdir(parents=True)
    (protected / "inner").mkdir()
    sandbox = LinuxSandbox(
        SandboxProfile((SandboxRoot(host.root, RootAccess.READ_WRITE), SandboxRoot(protected)), tools=("cat",))
    )
    assert sandbox.root == host.root
    assert sandbox._protected_paths == (protected,)
    with pytest.raises(SandboxError, match="write-protected"):
        LinuxSandbox(
            SandboxProfile(
                (
                    SandboxRoot(host.root, RootAccess.READ_WRITE),
                    SandboxRoot(protected),
                    SandboxRoot(protected / "inner", RootAccess.READ_WRITE),
                ),
                tools=("cat",),
            )
        )


def test_denied_paths_need_not_exist_but_must_lie_in_roots(host: Host) -> None:
    host.sandbox(denied_paths=(host.root / "missing/deeper",))
    with pytest.raises(SandboxError, match="outside"):
        host.sandbox(denied_paths=(host.path / "secret",))


def test_denied_symlinks_and_hard_links_fail_closed(host: Host) -> None:
    secret = host.root / "secret"
    secret.write_text("private")
    alias = host.root / "alias"
    alias.symlink_to(secret)
    with pytest.raises(SandboxError, match="symlink"):
        host.sandbox(denied_paths=(alias,))
    alias.unlink()
    os.link(secret, alias)
    with pytest.raises(SandboxError, match="hard link"):
        host.sandbox(denied_paths=(secret,))


def test_runtime_cannot_overlap_roots(host: Host) -> None:
    with pytest.raises(SandboxError, match="overlap"):
        LinuxSandbox(SandboxProfile((SandboxRoot(host.path / "usr"),), tools=("cat",)))


def test_mount_order_preserves_protection_denials_and_pinned_ancestors(host: Host, tmp_path: Path) -> None:
    protected = host.root / "docs/source"
    protected.mkdir(parents=True)
    secret = protected / "secret"
    secret.touch()
    sandbox = LinuxSandbox(
        SandboxProfile(
            (SandboxRoot(host.root, RootAccess.READ_WRITE), SandboxRoot(protected)),
            denied_paths=(secret,),
            tools=("cat",),
        )
    )
    control, scratch = tmp_path / "control", tmp_path / "scratch"
    argv = sandbox._invocation("printf ok", scratch, control)
    assert argv.index(str(host.root)) < argv.index(str(host.root / "docs")) < argv.index(str(protected))
    assert argv.index(str(protected)) < argv.index(str(secret))
    assert (str(host.root), str(host.root)) in _operations(argv, "--bind")
    assert (str(protected), str(protected)) in _operations(argv, "--ro-bind")
    assert (str(control / "deny-file"), str(secret)) in _operations(argv, "--ro-bind")


def test_namespace_process_environment_and_shell_arguments(backend: LinuxSandbox, tmp_path: Path) -> None:
    argv = backend._invocation("printf ok", tmp_path / "scratch", tmp_path / "control")
    for option in (
        "--unshare-user",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-net",
        "--unshare-cgroup",
    ):
        assert option in argv
    assert argv[argv.index("--uid") + 1] == str(os.getuid())
    assert argv[argv.index("--gid") + 1] == str(os.getgid())
    assert "--die-with-parent" in argv and "--new-session" in argv
    assert _operations(argv, "--cap-drop", 1) == [("ALL",)]
    assert "--cap-add" not in argv and "--proc" not in argv and "--as-pid-1" not in argv
    assert "--clearenv" in argv
    environment = dict(_operations(argv, "--setenv"))
    assert environment["HOME"] == "/var/empty" and environment["TMPDIR"] == str(tmp_path / "scratch")
    assert environment["PATH"].endswith("/usr/bin:/bin:/usr/sbin:/sbin")
    assert argv[argv.index("--") + 1 :] == [
        str(backend._shell),
        "--noprofile",
        "--norc",
        "-c",
        "printf ok",
        backend.process_label,
    ]


def test_runtime_handles_both_merged_and_split_usr(host: Host, tmp_path: Path) -> None:
    argv = host.sandbox()._invocation("true", tmp_path / "scratch", tmp_path / "control")
    assert ("usr/bin", str(host.path / "bin")) in _operations(argv, "--symlink")
    assert (str(host.path / "lib"), str(host.path / "lib")) in _operations(argv, "--ro-bind")
    assert (str(host.path / "usr"), str(host.path / "usr")) in _operations(argv, "--ro-bind")


@pytest.mark.parametrize("network", [False, True])
def test_network_namespace_and_curated_files_follow_profile(host: Host, tmp_path: Path, network: bool) -> None:
    argv = host.sandbox(network=network)._invocation("true", tmp_path / "scratch", tmp_path / "control")
    assert ("--unshare-net" in argv) is not network
    paths = {target for _, target in _operations(argv, "--ro-bind")}
    assert str(host.path / "etc/ld.so.cache") in paths
    assert (str(host.path / "etc/hosts") in paths) is network
    assert str(host.path / "etc") not in paths


@pytest.mark.parametrize("writable", [False, True])
@pytest.mark.parametrize("ipc", [False, True])
def test_scratch_and_device_writes_are_explicit(host: Host, tmp_path: Path, writable: bool, ipc: bool) -> None:
    access = RootAccess.READ_WRITE if writable else RootAccess.READ_ONLY
    sandbox = LinuxSandbox(SandboxProfile((SandboxRoot(host.root, access),), tools=(), ipc=ipc))
    scratch = tmp_path / "scratch"
    argv = sandbox._invocation("true", scratch, tmp_path / "control")
    # Scratch is the host directory itself, so its writes use disk rather than memory.
    assert (str(scratch), str(scratch)) in _operations(argv, "--bind" if writable else "--ro-bind")
    assert (str(scratch),) not in _operations(argv, "--tmpfs", 1)
    readonly = _operations(argv, "--remount-ro", 1)
    assert ("/",) in readonly and ("/dev",) in readonly
    assert (("/dev/shm",) in _operations(argv, "--tmpfs", 1)) is ipc


def test_missing_denial_is_covered_from_next_invocation(host: Host, tmp_path: Path) -> None:
    secret = host.root / "secret"
    sandbox = host.sandbox(denied_paths=(secret,))
    first = sandbox._invocation("true", tmp_path / "scratch", tmp_path / "control")
    assert str(secret) not in first
    secret.touch()
    second = sandbox._invocation("true", tmp_path / "scratch", tmp_path / "control")
    assert (str(tmp_path / "control/deny-file"), str(secret)) in _operations(second, "--ro-bind")
    assert str(host.root) not in {path for (path,) in _operations(second, "--tmpfs", 1)}


def test_denied_special_files_use_file_placeholder(host: Host, tmp_path: Path) -> None:
    fifo = host.root / "fifo"
    os.mkfifo(fifo)
    sandbox = host.sandbox(denied_paths=(fifo,))
    argv = sandbox._invocation("true", tmp_path / "scratch", tmp_path / "control")
    assert (str(tmp_path / "control/deny-file"), str(fifo)) in _operations(argv, "--ro-bind")


def test_tool_discovery_ignores_caller_path_and_missing_optional(host: Host, monkeypatch: pytest.MonkeyPatch) -> None:
    planted = host.root / "bin"
    host.program("planted", directory=planted)
    monkeypatch.setenv("PATH", str(planted))
    sandbox = host.sandbox(optional_tools=("planted", "missing"))
    assert sandbox.tools == ("cat",)
    with pytest.raises(SandboxError, match="not installed: planted"):
        LinuxSandbox(SandboxProfile((SandboxRoot(host.root),), tools=("planted",)))


def test_required_tool_link_outside_mounted_runtime_fails_closed(host: Host) -> None:
    outside = host.program("cat", directory=host.path / "opt/bin")
    tool = host.path / "usr/bin/cat"
    tool.unlink()
    tool.symlink_to(outside)
    with pytest.raises(SandboxError, match="outside.*runtime|runtime.*outside"):
        host.sandbox()


def test_optional_tool_link_outside_mounted_runtime_grants_nothing(host: Host) -> None:
    outside = host.program("rg", directory=host.path / "opt/bin")
    (host.path / "usr/bin/rg").symlink_to(outside)
    assert host.sandbox(optional_tools=("rg",)).tools == ("cat",)


def test_shell_link_outside_mounted_runtime_fails_closed(host: Host) -> None:
    outside = host.program("bash", directory=host.path / "opt/bin")
    shell = host.path / "usr/bin/bash"
    shell.unlink()
    shell.symlink_to(outside)
    with pytest.raises(SandboxError, match="outside.*runtime|runtime.*outside"):
        host.sandbox()


def test_bwrap_host_launcher_can_resolve_outside_workload_runtime(host: Host, tmp_path: Path) -> None:
    launcher = host.program("bwrap", directory=host.path / "opt/bin")
    system_entry = host.path / "usr/bin/bwrap"
    system_entry.unlink()
    system_entry.symlink_to(launcher)
    sandbox = host.sandbox()
    assert sandbox._bwrap == launcher
    replacement = host.program("other-bwrap", directory=host.path / "opt/bin")
    system_entry.unlink()
    system_entry.symlink_to(replacement)
    argv = sandbox._invocation("true", tmp_path / "scratch", tmp_path / "control")
    assert argv[0] == str(launcher)


@pytest.mark.parametrize("access", [RootAccess.READ_WRITE, RootAccess.READ_ONLY])
def test_bwrap_host_launcher_cannot_be_inside_a_writable_root(host: Host, access: RootAccess) -> None:
    launcher = host.program("bwrap", directory=host.path / "opt/bin")
    system_entry = host.path / "usr/bin/bwrap"
    system_entry.unlink()
    system_entry.symlink_to(launcher)
    profile = SandboxProfile(
        (SandboxRoot(host.root, RootAccess.READ_WRITE), SandboxRoot(launcher.parent, access)), tools=("cat",)
    )
    if access == RootAccess.READ_WRITE:
        with pytest.raises(SandboxError, match="writable sandbox root overlaps the host bubblewrap launcher"):
            LinuxSandbox(profile)
    else:
        assert LinuxSandbox(profile)._bwrap == launcher


def test_tool_link_into_mounted_nix_store_is_granted(host: Host) -> None:
    actual = host.program("cat", directory=host.path / "nix/store/cat/bin")
    tool = host.path / "usr/bin/cat"
    tool.unlink()
    tool.symlink_to(actual)
    assert host.sandbox().tools == ("cat",)


_DEFAULT_NIX_PROFILE_LINKS = linux._nix_profile_links
_SYSTEM_NIX_PROFILES = (Path("/nix/var/nix/profiles/default"), Path("/run/current-system/sw"))


def _account(name: str, home: str) -> pwd.struct_passwd:
    return pwd.struct_passwd((name, "x", os.getuid(), os.getgid(), "", home, "/bin/sh"))


def test_default_nix_profiles_include_account_profiles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pwd, "getpwuid", lambda _uid: _account("tester", "/home/tester"))
    assert _DEFAULT_NIX_PROFILE_LINKS() == (
        Path("/nix/var/nix/profiles/default"),
        Path("/nix/var/nix/profiles/per-user/tester/profile"),
        Path("/run/current-system/sw"),
        Path("/home/tester/.nix-profile"),
        Path("/home/tester/.local/state/nix/profiles/profile"),
    )


@pytest.mark.parametrize(
    ("name", "home", "expected"),
    [
        ("", "", _SYSTEM_NIX_PROFILES),
        (
            "tester",
            "relative/home",
            (_SYSTEM_NIX_PROFILES[0], Path("/nix/var/nix/profiles/per-user/tester/profile"), _SYSTEM_NIX_PROFILES[1]),
        ),
        (
            "bad/name",
            "/home/tester",
            (
                *_SYSTEM_NIX_PROFILES,
                Path("/home/tester/.nix-profile"),
                Path("/home/tester/.local/state/nix/profiles/profile"),
            ),
        ),
    ],
)
def test_account_nix_profiles_require_usable_account_fields(
    monkeypatch: pytest.MonkeyPatch, name: str, home: str, expected: tuple[Path, ...]
) -> None:
    monkeypatch.setattr(pwd, "getpwuid", lambda _uid: _account(name, home))
    assert _DEFAULT_NIX_PROFILE_LINKS() == expected


def test_sandbox_starts_without_an_account_entry(host: Host, monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(uid: int) -> pwd.struct_passwd:
        raise KeyError(f"getpwuid(): uid not found: {uid}")

    monkeypatch.setattr(pwd, "getpwuid", missing)
    monkeypatch.setattr(linux, "_nix_profile_links", _DEFAULT_NIX_PROFILE_LINKS)
    assert _DEFAULT_NIX_PROFILE_LINKS() == _SYSTEM_NIX_PROFILES
    assert host.sandbox().tools == ("cat",)


def test_legacy_per_user_profile_is_resolved_and_recreated(
    host: Host, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    profile = host.path / "nix/var/nix/profiles/per-user/tester/profile"
    profile.parent.mkdir(parents=True)
    store = host.path / "nix/store/legacy-profile"
    host.program("rg", directory=store / "bin")
    generation = profile.with_name("profile-1-link")
    generation.symlink_to(store)
    profile.symlink_to(generation.name)
    monkeypatch.setattr(linux, "_nix_profile_links", lambda: (profile,))
    sandbox = host.sandbox(optional_tools=("rg",))
    assert sandbox.tools == ("cat", "rg")
    argv = sandbox._invocation("true", tmp_path / "scratch", tmp_path / "control")
    assert (str(store), str(profile)) in _operations(argv, "--symlink")
    assert dict(_operations(argv, "--setenv"))["PATH"].startswith(f"{store / 'bin'}:")


def test_nix_profile_paths_are_resolved_once_and_recreated(
    host: Host, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    profile = host.path / "home/.nix-profile"
    store = host.path / "nix/store/profile-one"
    bin_directory = store / "bin"
    host.program("rg", directory=bin_directory)
    host.program("cat", directory=bin_directory)
    profile.symlink_to(store)
    sandbox = host.sandbox(optional_tools=("rg", "missing"))
    assert sandbox.tools == ("cat", "rg")
    assert sandbox._bin_directories == (bin_directory,)
    argv = sandbox._invocation("true", tmp_path / "scratch", tmp_path / "control")
    assert (str(store), str(profile)) in _operations(argv, "--symlink")
    assert dict(_operations(argv, "--setenv"))["PATH"].startswith(f"{bin_directory}:")
    assert sandbox._bwrap == host.path / "usr/bin/bwrap"
    profile.unlink()
    profile.symlink_to(host.root)
    assert sandbox._bin_directories == (bin_directory,)


def test_nix_supplies_bwrap_and_bash_when_system_lacks_them(host: Host) -> None:
    for name in ("bwrap", "bash"):
        (host.path / "usr/bin" / name).unlink()
    store = host.path / "nix/store/profile"
    for name in ("bwrap", "bash"):
        host.program(name, directory=store / "bin")
    (host.path / "home/.nix-profile").symlink_to(store)
    sandbox = host.sandbox()
    assert sandbox._bwrap == store / "bin/bwrap"
    assert sandbox._shell == store / "bin/bash"


def test_nix_profile_outside_store_is_ignored(host: Host) -> None:
    host.program("rg", directory=host.root / "bin")
    (host.path / "home/.nix-profile").symlink_to(host.root)
    assert host.sandbox(optional_tools=("rg",)).tools == ("cat",)


def test_root_cannot_cover_recreated_nix_profile_link(host: Host) -> None:
    store = host.path / "nix/store/profile"
    host.program("cat", directory=store / "bin")
    (host.path / "home/.nix-profile").symlink_to(store)
    with pytest.raises(SandboxError, match="overlap"):
        LinuxSandbox(SandboxProfile((SandboxRoot(host.path / "home"),), tools=("cat",)))


def test_filter_is_built_once_per_sandbox_with_profile_flags(host: Host, monkeypatch: pytest.MonkeyPatch) -> None:
    builds: list[tuple[Path, bool, bool]] = []

    def build(library: Path, *, network: bool, ipc: bool) -> bytes:
        builds.append((library, network, ipc))
        return b"filter"

    monkeypatch.setattr(linux, "build_filter", build)
    sandbox = host.sandbox(network=True, ipc=True)
    assert sandbox._seccomp_filter == b"filter"
    assert builds == [(host.path / "lib/libseccomp.so.2", True, True)]


@pytest.mark.parametrize("name", ["bash", "bwrap"])
def test_missing_backend_dependency_fails_closed(host: Host, name: str) -> None:
    (host.path / "usr/bin" / name).unlink()
    with pytest.raises(SandboxError, match="requires bubblewrap and bash"):
        host.sandbox()


class _Process:
    pid = 12345

    def __init__(self, returncode: int | None = 0) -> None:
        self.returncode = returncode
        self.communicated = False

    async def communicate(self) -> tuple[bytes, bytes]:
        self.communicated = True
        return b"", b""


def _fake_launch(
    backend: LinuxSandbox,
    monkeypatch: pytest.MonkeyPatch,
    *,
    status: str = '{"child-pid": 4321}\n{"exit-code": 7}\n',
    returncode: int = 7,
) -> tuple[_Process, dict[str, Any]]:
    process = _Process(returncode)
    captured: dict[str, Any] = {}

    async def create(*argv: str, **kwargs: Any) -> _Process:
        captured["argv"] = argv
        captured.update(kwargs)
        status_fd = int(argv[argv.index("--json-status-fd") + 1])
        os.write(status_fd, status.encode())
        seccomp_fd = int(argv[argv.index("--seccomp") + 1])
        captured["filter"] = os.read(seccomp_fd, 1000)
        return process

    async def read(_process: Any, stdout: bytearray, stderr: bytearray) -> None:
        stdout.extend(b"out\xff")
        stderr.extend(b"warning")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    monkeypatch.setattr(backend, "_read_output", read)
    return process, captured


async def test_success_uses_bwrap_exit_and_removes_scratch(
    backend: LinuxSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, captured = _fake_launch(backend, monkeypatch)
    result = await backend.run("exit 7")
    assert result == CommandResult(exit_code=7, stdout="out\ufffd", stderr="warning")
    assert captured["filter"] == b"compiled-filter"
    assert captured["env"]["HOME"] == "/var/empty"
    assert captured["stdin"] == asyncio.subprocess.DEVNULL
    assert "preexec_fn" not in captured
    assert len(captured["pass_fds"]) == 3
    assert not Path(captured["env"]["TMPDIR"]).exists()


async def test_each_invocation_recounts_host_tasks_and_hands_limits_to_host_wrapper(
    host: Host, monkeypatch: pytest.MonkeyPatch
) -> None:
    counts: list[int] = []
    ceilings = iter((300, 450))

    def count() -> int:
        ceiling = next(ceilings)
        counts.append(ceiling)
        return ceiling

    monkeypatch.setattr(linux, "owned_process_limit", count)
    sandbox = host.sandbox()
    assert counts == [], "construction must not snapshot a long-lived sandbox's process allowance"
    handed_limits: list[tuple[float, int | None]] = []

    def limited(argv: list[str], timeout: float, maximum: int | None) -> list[str]:
        handed_limits.append((timeout, maximum))
        return ["host-limit-wrapper", *argv]

    monkeypatch.setattr(linux, "limited_command", limited)
    _, captured = _fake_launch(sandbox, monkeypatch)
    for _ in range(2):
        result = await sandbox.run("exit 7")
        assert result.exit_code == 7
        assert captured["argv"][0] == "host-limit-wrapper"
        assert captured["argv"][1] == str(sandbox._bwrap)
        assert "preexec_fn" not in captured
    assert counts == [300, 450]
    assert handed_limits == [(sandbox.timeout_seconds, 300), (sandbox.timeout_seconds, 450)]


async def test_concurrent_runs_receive_independent_filter_descriptors(
    backend: LinuxSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    filter_descriptors: list[int] = []
    filters: list[bytes] = []
    both_started = asyncio.Event()

    async def create(*argv: str, **kwargs: Any) -> _Process:
        descriptor = int(argv[argv.index("--seccomp") + 1])
        filter_descriptors.append(descriptor)
        filters.append(os.read(descriptor, 1000))
        os.write(int(argv[argv.index("--json-status-fd") + 1]), b'{"exit-code": 0}\n')
        if len(filters) == 2:
            both_started.set()
        return _Process()

    async def read(*args: Any) -> None:
        await asyncio.wait_for(both_started.wait(), 5)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    monkeypatch.setattr(backend, "_read_output", read)
    results = await asyncio.gather(backend.run("true"), backend.run("true"))
    assert len(set(filter_descriptors)) == 2
    assert filters == [b"compiled-filter", b"compiled-filter"]
    assert results == [CommandResult(exit_code=0, stdout="", stderr="")] * 2


@pytest.mark.parametrize(
    "status", ["", '{"child-pid": 4321}\n', '{"exit-code": "7"}\n', '{"exit-code": 0}\n', "invalid"]
)
async def test_setup_or_status_failure_is_not_a_workload_exit(
    backend: LinuxSandbox, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    _fake_launch(backend, monkeypatch, status=status)
    with pytest.raises(SandboxError, match="setup"):
        await backend.run("true")


async def test_spawn_failure_raises_sandbox_error(backend: LinuxSandbox, monkeypatch: pytest.MonkeyPatch) -> None:
    async def fail(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("cannot create namespace")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fail)
    with pytest.raises(SandboxError, match="cannot start Linux sandbox"):
        await backend.run("true")


async def test_command_check_runs_before_spawn_or_path_revalidation(
    backend: LinuxSandbox, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unexpected(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("rejected command reached backend launch")

    monkeypatch.setattr(backend, "_validate_paths", unexpected)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", unexpected)
    result = await backend.run("printf ran; ls")
    assert result.exit_code == 126 and result.stdout == ""
    assert result.stderr.startswith("localmcp:")


@pytest.mark.parametrize("command", ["", "\0", "x" * (MAX_COMMAND_CHARACTERS + 1)])
async def test_command_validation_preserves_existing_error_api(backend: LinuxSandbox, command: str) -> None:
    with pytest.raises(SandboxError):
        await backend.run(command)


@pytest.mark.parametrize("failure", [TimeoutError(), _OutputLimitExceeded(), asyncio.CancelledError()])
async def test_limits_and_cancellation_terminate_before_cleanup(
    backend: LinuxSandbox, monkeypatch: pytest.MonkeyPatch, failure: BaseException
) -> None:
    process, captured = _fake_launch(backend, monkeypatch, status="")
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
        assert result.exit_code == -1 and result.stdout == "partial"
        assert result.timed_out is isinstance(failure, TimeoutError)
        assert result.truncated is isinstance(failure, _OutputLimitExceeded)
    assert killed == [process]
    assert not Path(captured["env"]["TMPDIR"]).exists()


async def test_kill_targets_namespace_init_then_process_group(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    process = _Process(None)
    info = tmp_path / "info"
    info.write_text(json.dumps({"child-pid": 56789}))
    signals: list[tuple[str, int, int]] = []
    monkeypatch.setattr(os, "kill", lambda pid, value: signals.append(("pid", pid, value)))
    monkeypatch.setattr(os, "killpg", lambda pid, value: signals.append(("group", pid, value)))
    await LinuxSandbox._kill(process, info)  # type: ignore[arg-type]
    assert signals == [("pid", 56789, signal.SIGKILL), ("group", process.pid, signal.SIGKILL)]
    assert process.communicated
