"""Test trusted setup decisions without applying restrictions to the test process."""

from __future__ import annotations

import ctypes
import errno
import io
import json
import os
import resource
import signal
from types import SimpleNamespace
from typing import Any

import pytest

from localmcp.sandbox import _linux_runner as runner


def _profile() -> dict[str, Any]:
    return {
        "executables": ["/usr/bin/cat"],
        "executable_trees": ["/usr/lib/git-core"],
        "executable_maps": ["/usr/lib/libc.so"],
        "readonly_mounts": ["/usr", "/readonly"],
        "read_directories": ["/work", "/scratch", "/usr/bin"],
        "write_paths": ["/work", "/scratch", "/dev/null"],
        "network": False,
        "ipc": False,
        "timeout_seconds": 0.5,
        "process_limit": 300,
        "max_open_files": 256,
        "command": "printf ok",
        "process_label": "test-process",
    }


class _Libc:
    def __init__(self) -> None:
        self.mount_calls: list[tuple[bytes, int]] = []
        self.prctl_calls: list[tuple[int, int]] = []
        self.capsets: list[list[tuple[int, int, int]]] = []

    def mount(self, _source: None, target: bytes, _type: None, flags: int, _data: None) -> int:
        self.mount_calls.append((target, flags))
        return 0

    def prctl(self, option: int, argument: int, *_args: int) -> int:
        self.prctl_calls.append((option, argument))
        if option == runner._PR_CAPBSET_READ:
            if argument >= 2:
                ctypes.set_errno(errno.EINVAL)
                return -1
            return 1
        return 0

    def capset(self, _header: Any, data: Any) -> int:
        self.capsets.append([(entry.effective, entry.permitted, entry.inheritable) for entry in data])
        return 0


@pytest.mark.parametrize(
    ("path", "directory", "expected"),
    [("/root/file", "/root", True), ("/root", "/root", True), ("/rooted", "/root", False), ("/other", "/", True)],
)
def test_directory_boundaries_are_not_string_prefixes(path: str, directory: str, expected: bool) -> None:
    assert runner._within(path, directory) is expected


def test_mountinfo_unescapes_paths_and_combines_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    content = (
        "1 0 0:1 / / rw,nosuid - tmpfs tmpfs rw\n"
        "2 1 0:2 / /work\\040space rw,nodev - ext4 /dev/test ro,noexec\n"
        "3 1 0:3 / /work\\040space/nested\\134name rw - tmpfs tmpfs rw\n"
    )
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: io.StringIO(content))
    visible = {"/": 1, "/work space": 2, "/work space/nested\\name": 3}
    monkeypatch.setattr(runner, "_visible_mount_id", visible.get)
    mounts = runner._mounts()
    assert mounts[0][0] == "/work space/nested\\name"
    flags = dict(mounts)["/work space"]
    assert flags & runner._MS_RDONLY
    assert flags & runner._MS_NODEV
    assert flags & runner._MS_NOEXEC
    assert dict(mounts)["/"] & runner._MS_NOSUID


def test_mountinfo_without_root_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: io.StringIO(""))
    with pytest.raises(RuntimeError, match="root mount"):
        runner._mounts()


def test_mountinfo_ignores_hidden_parent_children_and_missing_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    content = (
        "1 0 0:1 / / rw,nosuid - tmpfs tmpfs rw\n"
        "2 1 0:2 / /tree/child ro,noexec - tmpfs tmpfs ro\n"
        "3 1 0:3 / /tree ro,noexec - tmpfs tmpfs ro\n"
        "4 1 0:4 / /tree rw - tmpfs tmpfs rw\n"
        "5 1 0:5 / /gone ro - tmpfs tmpfs ro\n"
    )
    visible = {"/": 1, "/tree/child": 4, "/tree": 4, "/gone": None}
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: io.StringIO(content))
    monkeypatch.setattr(runner, "_visible_mount_id", visible.get)

    mounts = runner._mounts()

    assert {path for path, _flags in mounts} == {"/tree", "/"}
    # Restrictions on a hidden mount must not be attributed to the visible bind.
    assert dict(mounts)["/tree"] == 0


def test_mountinfo_requires_visible_root_record(monkeypatch: pytest.MonkeyPatch) -> None:
    content = "1 0 0:1 / / rw - tmpfs tmpfs rw\n"
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: io.StringIO(content))
    monkeypatch.setattr(runner, "_visible_mount_id", lambda path: 99)
    with pytest.raises(RuntimeError, match="root mount is missing"):
        runner._mounts()


def test_visible_mount_id_uses_reachable_descriptor_and_closes_it(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[tuple[str, int]] = []
    closed: list[int] = []

    def open_path(path: str, flags: int) -> int:
        opened.append((path, flags))
        return 42

    def fdinfo(path: str, **kwargs: Any) -> io.StringIO:
        assert path == "/proc/self/fdinfo/42"
        return io.StringIO("pos:\t0\nflags:\t012000000\nmnt_id:\t87\nino:\t10\n")

    monkeypatch.setattr(os, "open", open_path)
    monkeypatch.setattr(os, "close", closed.append)
    monkeypatch.setattr("builtins.open", fdinfo)

    assert runner._visible_mount_id("/visible") == 87
    assert opened == [("/visible", getattr(os, "O_PATH", 0x200000) | os.O_CLOEXEC)]
    assert closed == [42]


@pytest.mark.parametrize(
    "error", [FileNotFoundError(errno.ENOENT, "gone"), NotADirectoryError(errno.ENOTDIR, "hidden")]
)
def test_visible_mount_id_ignores_paths_hidden_by_overmounts(monkeypatch: pytest.MonkeyPatch, error: OSError) -> None:
    def open_path(*args: object) -> int:
        raise error

    monkeypatch.setattr(os, "open", open_path)
    assert runner._visible_mount_id("/hidden/child") is None


def test_visible_mount_id_permission_errors_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def open_path(*args: object) -> int:
        raise PermissionError(errno.EACCES, "cannot inspect")

    monkeypatch.setattr(os, "open", open_path)
    with pytest.raises(PermissionError, match="cannot inspect"):
        runner._visible_mount_id("/restricted")


@pytest.mark.parametrize("content", ["pos:\t0\n", "mnt_id:\tinvalid\n"])
def test_visible_mount_id_rejects_unidentifiable_fdinfo_and_closes_descriptor(
    monkeypatch: pytest.MonkeyPatch, content: str
) -> None:
    closed: list[int] = []
    monkeypatch.setattr(os, "open", lambda *args: 42)
    monkeypatch.setattr(os, "close", closed.append)
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: io.StringIO(content))
    with pytest.raises((RuntimeError, ValueError)):
        runner._visible_mount_id("/restricted")
    assert closed == [42]


def test_visible_mount_id_fdinfo_errors_fail_closed_and_close_descriptor(monkeypatch: pytest.MonkeyPatch) -> None:
    closed: list[int] = []
    monkeypatch.setattr(os, "open", lambda *args: 42)
    monkeypatch.setattr(os, "close", closed.append)

    def fdinfo(*args: object, **kwargs: object) -> io.StringIO:
        raise OSError(errno.EIO, "cannot read fdinfo")

    monkeypatch.setattr("builtins.open", fdinfo)
    with pytest.raises(OSError, match="cannot read fdinfo"):
        runner._visible_mount_id("/restricted")
    assert closed == [42]


def test_mount_lockdown_allows_only_selected_code_and_preserves_readonly(monkeypatch: pytest.MonkeyPatch) -> None:
    library = _Libc()
    mounts = [
        ("/usr/bin/cat", 0),
        ("/usr/bin/ls", 0),
        ("/usr/lib/libc.so", 0),
        ("/usr/lib/git-core/helper", 0),
        ("/readonly/child", 0),
        ("/work", 0),
        ("/dev", 0),
        ("/", 0),
    ]
    monkeypatch.setattr(runner, "_mounts", lambda: mounts)
    permissions: list[tuple[str, int]] = []
    monkeypatch.setattr(os, "chmod", lambda path, mode: permissions.append((path, mode)))

    runner._lock_mounts(_profile(), library)  # type: ignore[arg-type]

    flags = dict(library.mount_calls)
    assert not flags[b"/usr/bin/cat"] & runner._MS_NOEXEC
    assert not flags[b"/usr/lib/libc.so"] & runner._MS_NOEXEC
    assert not flags[b"/usr/lib/git-core/helper"] & runner._MS_NOEXEC
    assert flags[b"/usr/bin/ls"] & runner._MS_NOEXEC
    assert flags[b"/work"] & runner._MS_NOEXEC
    assert flags[b"/readonly/child"] & runner._MS_RDONLY
    assert flags[b"/"] & runner._MS_RDONLY
    assert flags[b"/"] & runner._MS_NOEXEC
    assert all(value & runner._MS_NOSUID for value in flags.values())
    assert not flags[b"/dev"] & runner._MS_NODEV
    assert permissions == [("/localmcp", 0)]
    assert library.mount_calls[-1][0] == b"/"


@pytest.mark.parametrize("host_readonly", [False, True])
def test_mount_lockdown_honors_specific_writable_roots_without_relaxing_host_readonly(
    monkeypatch: pytest.MonkeyPatch, host_readonly: bool
) -> None:
    library = _Libc()
    previous = runner._MS_RDONLY if host_readonly else 0
    mounts = [
        ("/parent/child/protected/data", 0),
        ("/parent/child/nested", previous),
        ("/parent/child", 0),
        ("/parent", 0),
        ("/", 0),
    ]
    monkeypatch.setattr(runner, "_mounts", lambda: mounts)
    monkeypatch.setattr(os, "chmod", lambda *args: None)
    profile = _profile() | {
        "readonly_mounts": ["/parent", "/parent/child/protected"],
        "write_paths": ["/parent/child"],
    }

    runner._lock_mounts(profile, library)  # type: ignore[arg-type]

    flags = dict(library.mount_calls)
    assert flags[b"/parent"] & runner._MS_RDONLY
    assert not flags[b"/parent/child"] & runner._MS_RDONLY
    assert bool(flags[b"/parent/child/nested"] & runner._MS_RDONLY) is host_readonly
    assert flags[b"/parent/child/protected/data"] & runner._MS_RDONLY


def test_mount_failure_prevents_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    library = _Libc()
    monkeypatch.setattr(runner, "_mounts", lambda: [("/work", 0), ("/", 0)])
    ctypes.set_errno(errno.EPERM)
    library.mount = lambda *args: -1  # type: ignore[method-assign]
    with pytest.raises(OSError, match="lock mount /work"):
        runner._lock_mounts(_profile(), library)  # type: ignore[arg-type]


def test_rebuilt_entries_must_be_the_ones_the_host_listed(tmp_path: Any) -> None:
    listed, removed = tmp_path / "listed", tmp_path / "removed"
    listed.write_text("listed")
    info = os.lstat(listed)
    # The profile arrives as JSON, so each identity is a list.
    identities = {str(listed): [info.st_dev, info.st_ino], str(removed): [info.st_dev, info.st_ino + 1]}
    runner._check_entry_identities({"entry_identities": identities})
    runner._check_entry_identities({})

    # Create the replacement before swapping it in, so it cannot reuse the listed inode.
    swap = tmp_path / "swap"
    swap.symlink_to(tmp_path)
    swap.replace(listed)
    with pytest.raises(RuntimeError, match=f"entry changed while the sandbox was being prepared: {listed}"):
        runner._check_entry_identities({"entry_identities": identities})


def test_capabilities_are_removed_before_no_new_privileges(monkeypatch: pytest.MonkeyPatch) -> None:
    library = _Libc()
    runner._drop_capabilities(library)  # type: ignore[arg-type]
    assert library.capsets == [[(0, 0, 0), (0, 0, 0)]]
    assert (runner._PR_CAPBSET_DROP, 0) in library.prctl_calls
    assert (runner._PR_CAPBSET_DROP, 1) in library.prctl_calls
    assert (runner._PR_CAP_AMBIENT, runner._PR_CAP_AMBIENT_CLEAR_ALL) in library.prctl_calls
    assert library.prctl_calls[-2:] == [(runner._PR_SET_NO_NEW_PRIVS, 1), (runner._PR_SET_DUMPABLE, 0)]


def test_capability_failure_is_not_ignored() -> None:
    library = _Libc()
    ctypes.set_errno(errno.EPERM)
    library.capset = lambda *args: -1  # type: ignore[method-assign]
    with pytest.raises(OSError, match="drop sandbox capabilities"):
        runner._drop_capabilities(library)  # type: ignore[arg-type]


class _Landlock:
    def __init__(self, abi: int = 3, failed_rule: bool = False) -> None:
        self.abi = abi
        self.failed_rule = failed_rule
        self.rules: list[tuple[int, int]] = []
        self.enforced = False

    def syscall(self, number: int, *args: Any) -> int:
        if number == 444:
            return self.abi if args[0] is None else 100
        if number == 445:
            rule = ctypes.cast(args[2], ctypes.POINTER(runner._LandlockPath)).contents
            self.rules.append((rule.parent_fd, rule.allowed_access))
            if self.failed_rule:
                ctypes.set_errno(errno.EPERM)
                return -1
            return 0
        assert number == 446
        self.enforced = True
        return 0


@pytest.fixture
def landlock_fds(monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, int], list[int]]:
    monkeypatch.setattr(os, "uname", lambda: SimpleNamespace(machine="aarch64"))
    opened: dict[str, int] = {}
    closed: list[int] = []

    def open_path(path: str, _flags: int) -> int:
        opened[path] = len(opened) + 200
        return opened[path]

    monkeypatch.setattr(os, "open", open_path)
    monkeypatch.setattr(os, "close", closed.append)
    monkeypatch.setattr(os.path, "isdir", lambda path: path in {"/work", "/scratch"})
    return opened, closed


def test_landlock_rules_separate_execution_mapping_and_write_authority(
    landlock_fds: tuple[dict[str, int], list[int]],
) -> None:
    library = _Landlock()
    runner._landlock(_profile(), library)  # type: ignore[arg-type]
    opened, closed = landlock_fds
    rules = dict(library.rules)
    assert rules[opened["/usr/bin/cat"]] == runner._LANDLOCK_EXECUTE
    assert rules[opened["/usr/lib/git-core"]] == runner._LANDLOCK_EXECUTE
    assert "/usr/lib/libc.so" not in opened
    assert rules[opened["/work"]] == (
        runner._LANDLOCK_READ_DIR | runner._LANDLOCK_WRITE_FILE | runner._LANDLOCK_TRUNCATE | runner._LANDLOCK_REFER
    )
    assert not rules[opened["/dev/null"]] & runner._LANDLOCK_REFER
    assert library.enforced
    assert set(closed) == {100, *opened.values()}


@pytest.mark.parametrize("abi", [0, 1, 2])
def test_old_landlock_abi_is_rejected(landlock_fds: tuple[dict[str, int], list[int]], abi: int) -> None:
    library = _Landlock(abi)
    with pytest.raises(RuntimeError, match="ABI >= 3"):
        runner._landlock(_profile(), library)  # type: ignore[arg-type]
    assert not library.enforced and not library.rules


def test_landlock_rule_failure_closes_descriptors(landlock_fds: tuple[dict[str, int], list[int]]) -> None:
    library = _Landlock(failed_rule=True)
    with pytest.raises(OSError, match="add Landlock rule"):
        runner._landlock(_profile(), library)  # type: ignore[arg-type]
    opened, closed = landlock_fds
    assert closed == [opened["/usr/bin/cat"], 100]
    assert not library.enforced


class _Policy:
    def __init__(self) -> None:
        self.rules: list[tuple[str, tuple[tuple[int, int, int, int], ...], int]] = []
        self.loaded = False
        self.closed = False

    def deny(self, name: str, *comparisons: runner._SeccompComparison, error: int = errno.EPERM) -> None:
        values = tuple((value.arg, value.op, value.datum_a, value.datum_b) for value in comparisons)
        self.rules.append((name, values, error))

    def load(self) -> None:
        self.loaded = True

    def close(self) -> None:
        self.closed = True

    def denied(self, name: str, *args: int) -> bool:
        def matches(value: tuple[int, int, int, int]) -> bool:
            index, operation, first, second = value
            argument = args[index]
            if operation == runner._SCMP_CMP_EQ:
                return argument == first
            if operation == runner._SCMP_CMP_GE:
                return argument >= first
            assert operation == runner._SCMP_CMP_MASKED_EQ
            return argument & first == second

        return any(
            syscall == name and all(matches(value) for value in comparisons)
            for syscall, comparisons, _error in self.rules
        )


@pytest.mark.parametrize("network", [False, True])
@pytest.mark.parametrize("ipc", [False, True])
def test_seccomp_grants_network_and_shared_memory_independently(network: bool, ipc: bool) -> None:
    profile = _profile() | {"network": network, "ipc": ipc}
    policy = _Policy()
    runner._seccomp(profile, policy)  # type: ignore[arg-type]
    assert policy.loaded
    assert policy.denied("socket", 2, 1, 0) is not network
    assert policy.denied("socket", 10, 2 | 0x80000, 17) is not network
    assert policy.denied("socket", 1, 1, 0)
    assert policy.denied("socket", 46, 1, 0)
    assert policy.denied("socket", 2, 3, 0)
    assert policy.denied("socket", 2, 2, 1)
    assert policy.denied("socket", 2, 2, 256)
    assert policy.denied("shmget") is not ipc
    for name in ("socketpair", "bind", "listen", "accept", "semget", "msgget", "ptrace", "io_uring_setup", "unshare"):
        assert policy.denied(name), name
    assert policy.denied("clone", 17) is False
    assert all(policy.denied("clone", flag | 17) for flag in runner._CLONE_NAMESPACE_FLAGS)
    assert ("clone3", (), errno.ENOSYS) in policy.rules


@pytest.mark.parametrize("process_limit", [None, 300])
def test_resource_limits_are_hard_and_soft(monkeypatch: pytest.MonkeyPatch, process_limit: int | None) -> None:
    calls: list[tuple[int, tuple[int, int]]] = []
    monkeypatch.setattr(resource, "setrlimit", lambda kind, limits: calls.append((kind, limits)))
    runner._limits(_profile() | {"process_limit": process_limit})
    assert calls[:2] == [(resource.RLIMIT_CPU, (1, 1)), (resource.RLIMIT_NOFILE, (256, 256))]
    assert (resource.RLIMIT_NPROC, (300, 300)) in calls if process_limit else len(calls) == 2


def test_status_write_handles_partial_writes(monkeypatch: pytest.MonkeyPatch) -> None:
    output = bytearray()

    def write(_fd: int, data: bytes) -> int:
        output.extend(data[:2])
        return min(2, len(data))

    monkeypatch.setattr(os, "write", write)
    runner._status(42, {"exit_code": 7})
    assert json.loads(output) == {"exit_code": 7}
    assert output.endswith(b"\n")


def test_closed_status_pipe_is_a_setup_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "write", lambda *_args: 0)
    with pytest.raises(RuntimeError, match="status pipe closed"):
        runner._write(42, b"status")


def test_child_cannot_inherit_other_control_descriptors(monkeypatch: pytest.MonkeyPatch) -> None:
    closed: list[int] = []
    monkeypatch.setattr(os, "listdir", lambda _: ["0", "1", "2", "3", "4", "5"])
    monkeypatch.setattr(os, "close", closed.append)
    runner._close_other_fds(4)
    assert closed == [3, 5]


class _Exit(BaseException):
    pass


def test_child_setup_failure_never_executes_workload(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[str] = []
    output: list[bytes] = []
    monkeypatch.setattr(runner, "_close_other_fds", lambda _: events.append("fds"))
    monkeypatch.setattr(runner, "_limits", lambda _: events.append("limits"))

    def fail(*_args: object) -> None:
        events.append("landlock")
        raise RuntimeError("Landlock unavailable")

    monkeypatch.setattr(runner, "_landlock", fail)
    monkeypatch.setattr(runner, "_seccomp", lambda *_args: events.append("seccomp"))
    monkeypatch.setattr(os, "execv", lambda *_args: events.append("exec"))
    monkeypatch.setattr(runner, "_write", lambda _fd, data: output.append(data))

    def exit_process(code: int) -> None:
        assert code == 125
        raise _Exit

    monkeypatch.setattr(os, "_exit", exit_process)
    with pytest.raises(_Exit):
        runner._child(_profile(), _Libc(), _Policy(), 42)  # type: ignore[arg-type]
    assert events == ["fds", "limits", "landlock"]
    assert output == [b"RuntimeError: Landlock unavailable"]


def test_child_restores_default_signals_for_the_workload(monkeypatch: pytest.MonkeyPatch) -> None:
    # Python ignores SIGPIPE and SIGXFSZ at startup, and execv keeps ignored signals ignored.
    events: list[object] = []
    monkeypatch.setattr(runner, "_close_other_fds", lambda _: None)
    monkeypatch.setattr(runner, "_limits", lambda _: None)
    monkeypatch.setattr(runner, "_landlock", lambda *_args: None)
    monkeypatch.setattr(runner, "_seccomp", lambda *_args: None)
    monkeypatch.setattr(runner, "_write", lambda *_args: None)
    monkeypatch.setattr(signal, "signal", lambda number, handler: events.append((number, handler)))
    monkeypatch.setattr(os, "execv", lambda path, _argv: events.append(path))

    runner._child(_profile(), _Libc(), _Policy(), 42)  # type: ignore[arg-type]

    assert events == [(signal.SIGPIPE, signal.SIG_DFL), (signal.SIGXFSZ, signal.SIG_DFL), "/bin/sh"]


class _Function:
    def __init__(self, function: Any) -> None:
        self.function = function

    def __call__(self, *args: Any) -> Any:
        return self.function(*args)


def test_seccomp_library_errors_are_reported_and_released(monkeypatch: pytest.MonkeyPatch) -> None:
    released: list[int] = []
    library = SimpleNamespace(
        seccomp_init=_Function(lambda action: 1),
        seccomp_release=_Function(released.append),
        seccomp_syscall_resolve_name=_Function(lambda name: -1 if name == b"absent" else 1),
        seccomp_rule_add_array=_Function(lambda *args: -errno.EINVAL),
        seccomp_load=_Function(lambda context: -errno.EPERM),
    )
    monkeypatch.setattr(ctypes, "CDLL", lambda *_args, **_kwargs: library)
    policy = runner._Seccomp()
    policy.deny("absent")
    with pytest.raises(OSError, match="cannot restrict syscall mount"):
        policy.deny("mount")
    with pytest.raises(OSError, match="cannot enforce"):
        policy.load()
    policy.close()
    assert released == [1]
