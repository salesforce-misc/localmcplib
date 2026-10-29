"""Host-side libseccomp discovery, failure handling, and policy rules."""

from __future__ import annotations

import ctypes
import errno
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from localmcp.sandbox import SandboxError
from localmcp.sandbox import _seccomp as seccomp


class _Function:
    def __init__(self, function: Any) -> None:
        self.function = function

    def __call__(self, *args: Any) -> Any:
        return self.function(*args)


@pytest.fixture
def library(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    released: list[int] = []
    added: list[tuple[int, int, int]] = []
    resolved: list[bytes] = []

    def resolve(name: bytes) -> int:
        resolved.append(name)
        return len(resolved)

    def add(context: int, action: int, number: int, count: int, arguments: Any) -> int:
        added.append((action, number, count))
        return 0

    def export(context: int, descriptor: int) -> int:
        os.write(descriptor, b"compiled-bpf")
        return 0

    value = SimpleNamespace(
        seccomp_init=_Function(lambda action: 1),
        seccomp_release=_Function(released.append),
        seccomp_syscall_resolve_name=_Function(resolve),
        seccomp_rule_add_array=_Function(add),
        seccomp_export_bpf=_Function(export),
        released=released,
        added=added,
        resolved=resolved,
    )
    monkeypatch.setattr(ctypes, "CDLL", lambda *_args, **_kwargs: value)
    return value


def test_library_discovery_prefers_system_and_resolves_nix_links(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    system, store, profile = tmp_path / "system", tmp_path / "store", tmp_path / "profile"
    for directory in (system, store, profile / "lib"):
        directory.mkdir(parents=True)
    actual = store / "libseccomp.so.2.6"
    actual.touch()
    (profile / "lib/libseccomp.so.2").symlink_to(actual)
    installed = system / "libseccomp.so.2"
    installed.touch()
    monkeypatch.setattr(seccomp, "_LIBRARY_DIRECTORIES", (system,))
    assert seccomp.find_library((profile,)) == installed
    installed.unlink()
    assert seccomp.find_library((profile,)) == actual
    actual.unlink()
    with pytest.raises(SandboxError, match="requires libseccomp"):
        seccomp.find_library((profile,))


@pytest.mark.parametrize(
    ("machine", "native", "foreign"),
    [("x86_64", "x86_64-linux-gnu", "aarch64-linux-gnu"), ("aarch64", "aarch64-linux-gnu", "x86_64-linux-gnu")],
)
def test_library_discovery_skips_other_architectures(machine: str, native: str, foreign: str) -> None:
    # A multiarch host can carry another architecture's libseccomp, which cannot be loaded.
    directories = seccomp._library_directories(machine)
    assert directories[:2] == (Path("/lib") / native, Path("/usr/lib") / native)
    assert not any(foreign in str(directory) for directory in directories)
    assert seccomp._library_directories("riscv64") == tuple(map(Path, ("/lib64", "/usr/lib64", "/lib", "/usr/lib")))


def test_library_loading_requires_an_absolute_path(library: SimpleNamespace) -> None:
    with pytest.raises(SandboxError, match="must be absolute"):
        seccomp.build_filter(Path("libseccomp.so.2"), network=False, ipc=False)


def test_build_exports_complete_filter_and_releases_context(library: SimpleNamespace) -> None:
    result = seccomp.build_filter(Path("/system/libseccomp.so.2"), network=False, ipc=False)
    assert result == b"compiled-bpf"
    assert library.released == [1]
    assert {b"mount", b"unshare", b"clone3", b"socket", b"shmget"} <= set(library.resolved)


def test_unknown_required_syscall_fails_closed_and_releases(library: SimpleNamespace) -> None:
    library.seccomp_syscall_resolve_name.function = lambda name: -1 if name == b"mount" else 1
    exported: list[bool] = []
    library.seccomp_export_bpf.function = lambda *_args: exported.append(True) or 0
    with pytest.raises(SandboxError, match="does not recognize required syscall: mount"):
        seccomp.build_filter(Path("/system/libseccomp.so.2"), network=False, ipc=False)
    assert library.released == [1]
    assert not exported


def test_known_architecture_pseudo_syscalls_are_passed_to_library(library: SimpleNamespace) -> None:
    library.seccomp_syscall_resolve_name.function = lambda name: -10001
    policy = seccomp._Seccomp(Path("/system/libseccomp.so.2"))
    try:
        policy.deny("umount")
    finally:
        policy.close()
    assert library.added == [(seccomp._SCMP_ACT_ERRNO | errno.EPERM, -10001, 0)]


@pytest.mark.parametrize("failure", ["initialize", "add", "export", "empty"])
def test_filter_builder_fails_closed_on_library_errors(library: SimpleNamespace, failure: str) -> None:
    if failure == "initialize":
        library.seccomp_init.function = lambda *_args: 0
    elif failure == "add":
        library.seccomp_rule_add_array.function = lambda *_args: -errno.EINVAL
    elif failure == "export":
        library.seccomp_export_bpf.function = lambda *_args: -errno.EPERM
    else:
        library.seccomp_export_bpf.function = lambda *_args: 0
    with pytest.raises(SandboxError):
        seccomp.build_filter(Path("/system/libseccomp.so.2"), network=False, ipc=False)
    assert library.released == ([] if failure == "initialize" else [1])


def test_library_load_failure_is_reported_as_sandbox_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def failed(*args: Any, **kwargs: Any) -> None:
        raise OSError("incompatible shared library")

    monkeypatch.setattr(ctypes, "CDLL", failed)
    with pytest.raises(SandboxError, match="cannot build sandbox seccomp filter"):
        seccomp.build_filter(Path("/system/libseccomp.so.2"), network=False, ipc=False)


class _Rules:
    def __init__(self) -> None:
        self.rules: list[tuple[str, tuple[Any, ...], int]] = []

    def deny(self, name: str, *comparisons: Any, error: int = errno.EPERM) -> None:
        self.rules.append((name, comparisons, error))

    def error(self, name: str, *arguments: int) -> int | None:
        def matches(comparison: Any) -> bool:
            value = arguments[comparison.arg]
            if comparison.op == seccomp._SCMP_CMP_EQ:
                return value == comparison.datum_a
            if comparison.op == seccomp._SCMP_CMP_GE:
                return value >= comparison.datum_a
            assert comparison.op == seccomp._SCMP_CMP_MASKED_EQ
            return value & comparison.datum_a == comparison.datum_b

        for syscall, comparisons, error in self.rules:
            if syscall == name and all(matches(comparison) for comparison in comparisons):
                return error
        return None


@pytest.mark.parametrize("network", [False, True])
@pytest.mark.parametrize("ipc", [False, True])
def test_syscall_rules_preserve_namespace_network_and_ipc_boundaries(network: bool, ipc: bool) -> None:
    policy = _Rules()
    seccomp._rules(policy, network=network, ipc=ipc)  # type: ignore[arg-type]
    for syscall in ("mount", "unshare", "setns", "ptrace", "memfd_create", "io_uring_setup", "bind", "listen"):
        assert policy.error(syscall) == errno.EPERM
    for flag in seccomp._CLONE_NAMESPACE_FLAGS:
        assert policy.error("clone", flag | 17) == errno.EPERM
    assert policy.error("clone", 17) is None
    assert policy.error("clone3") == errno.ENOSYS
    for syscall in ("shmget", "shmat", "shmdt", "shmctl"):
        assert policy.error(syscall) == (None if ipc else errno.EPERM)
    for syscall in ("semget", "msgget", "socketpair"):
        assert policy.error(syscall) == errno.EPERM
    for arguments in ((2, 1, 6), (10, 2, 17), (2, 1 | 0x80000 | 0x800, 0)):
        assert policy.error("socket", *arguments) == (None if network else errno.EPERM)
    for arguments in ((1, 1, 0), (999, 1, 0), (2, 3, 0), (2, 1, 255), (2, 1, 999)):
        assert policy.error("socket", *arguments) == errno.EPERM
    assert policy.error("connect") == (None if network else errno.EPERM)
