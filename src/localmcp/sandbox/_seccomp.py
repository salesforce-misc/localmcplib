"""Build Linux seccomp BPF on the host; no library is loaded inside the sandbox."""

from __future__ import annotations

import ctypes
import errno
import os
import tempfile
from pathlib import Path

from localmcp.sandbox.base import SandboxError

_SCMP_ACT_ALLOW = 0x7FFF0000
_SCMP_ACT_ERRNO = 0x00050000
_SCMP_CMP_EQ = 4
_SCMP_CMP_GE = 5
_SCMP_CMP_MASKED_EQ = 7
_CLONE_NAMESPACE_FLAGS = (0x80, 0x20000, 0x2000000, 0x4000000, 0x8000000, 0x10000000, 0x20000000, 0x40000000)
_LIBRARY_DIRECTORIES = tuple(
    map(
        Path,
        (
            "/lib/x86_64-linux-gnu",
            "/usr/lib/x86_64-linux-gnu",
            "/lib/aarch64-linux-gnu",
            "/usr/lib/aarch64-linux-gnu",
            "/lib64",
            "/usr/lib64",
            "/lib",
            "/usr/lib",
        ),
    )
)


def find_library(nix_profiles: tuple[Path, ...] = ()) -> Path:
    """Find libseccomp by absolute path, preferring system installations to Nix."""
    for directory in (*_LIBRARY_DIRECTORIES, *(profile / "lib" for profile in nix_profiles)):
        candidate = directory / "libseccomp.so.2"
        if candidate.is_absolute() and candidate.is_file():
            return candidate.resolve(strict=True)
    raise SandboxError("Linux sandbox requires libseccomp in a system library directory or a Nix profile")


class _SeccompComparison(ctypes.Structure):
    _fields_ = [
        ("arg", ctypes.c_uint),
        ("op", ctypes.c_int),
        ("datum_a", ctypes.c_uint64),
        ("datum_b", ctypes.c_uint64),
    ]


class _Seccomp:
    def __init__(self, library: Path) -> None:
        if not library.is_absolute():
            raise SandboxError("sandbox libseccomp path must be absolute")
        self.library = ctypes.CDLL(os.fspath(library), use_errno=True)
        self.library.seccomp_init.argtypes = [ctypes.c_uint32]
        self.library.seccomp_init.restype = ctypes.c_void_p
        self.library.seccomp_release.argtypes = [ctypes.c_void_p]
        self.library.seccomp_release.restype = None
        self.library.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
        self.library.seccomp_syscall_resolve_name.restype = ctypes.c_int
        self.library.seccomp_rule_add_array.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.c_uint,
            ctypes.POINTER(_SeccompComparison),
        ]
        self.library.seccomp_rule_add_array.restype = ctypes.c_int
        self.library.seccomp_export_bpf.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.library.seccomp_export_bpf.restype = ctypes.c_int
        self.context = self.library.seccomp_init(_SCMP_ACT_ALLOW)
        if not self.context:
            raise SandboxError("cannot initialize sandbox seccomp filter")

    def deny(self, name: str, *comparisons: _SeccompComparison, error: int = errno.EPERM) -> None:
        number = int(self.library.seccomp_syscall_resolve_name(name.encode("ascii")))
        if number == -1:
            raise SandboxError(f"libseccomp does not recognize required syscall: {name}")
        # Other negative values are libseccomp's known pseudo syscall numbers for
        # calls absent on this architecture. Let the library translate those rules.
        arguments = (_SeccompComparison * len(comparisons))(*comparisons)
        result = int(
            self.library.seccomp_rule_add_array(
                self.context, _SCMP_ACT_ERRNO | error, number, len(comparisons), arguments
            )
        )
        if result < 0:
            raise SandboxError(f"cannot restrict syscall {name}: {os.strerror(-result)}")

    def export(self) -> bytes:
        with tempfile.TemporaryFile() as output:
            result = int(self.library.seccomp_export_bpf(self.context, output.fileno()))
            if result < 0:
                raise SandboxError(f"cannot export sandbox seccomp filter: {os.strerror(-result)}")
            output.seek(0)
            program = output.read()
        if not program:
            raise SandboxError("libseccomp exported an empty sandbox filter")
        return program

    def close(self) -> None:
        self.library.seccomp_release(self.context)


def build_filter(library: Path, *, network: bool, ipc: bool) -> bytes:
    """Export the complete native-architecture filter or fail without running a command."""
    try:
        policy = _Seccomp(library)
        try:
            _rules(policy, network=network, ipc=ipc)
            return policy.export()
        finally:
            policy.close()
    except (OSError, AttributeError) as exc:
        raise SandboxError(f"cannot build sandbox seccomp filter: {exc}") from exc


def _rules(policy: _Seccomp, *, network: bool, ipc: bool) -> None:
    for name in (
        "ptrace",
        "process_vm_readv",
        "process_vm_writev",
        "pidfd_getfd",
        "mount",
        "umount",
        "umount2",
        "unshare",
        "setns",
        "chroot",
        "pivot_root",
        "move_mount",
        "fsopen",
        "fsconfig",
        "fsmount",
        "open_tree",
        "mount_setattr",
        "io_uring_setup",
        "io_uring_enter",
        "io_uring_register",
        "memfd_create",
        "bpf",
        "perf_event_open",
        "keyctl",
        "add_key",
        "request_key",
        "reboot",
        "kexec_load",
        "kexec_file_load",
        "init_module",
        "finit_module",
        "delete_module",
        "swapon",
        "swapoff",
        "iopl",
        "ioperm",
        "userfaultfd",
        "semget",
        "semop",
        "semctl",
        "semtimedop",
        "msgget",
        "msgsnd",
        "msgrcv",
        "msgctl",
        "ipc",
    ):
        policy.deny(name)
    for flag in _CLONE_NAMESPACE_FLAGS:
        policy.deny("clone", _SeccompComparison(0, _SCMP_CMP_MASKED_EQ, flag, flag))
    policy.deny("clone3", error=errno.ENOSYS)
    if not ipc:
        for name in ("shmget", "shmat", "shmdt", "shmctl"):
            policy.deny(name)
    for name in ("socketpair", "bind", "listen", "accept", "accept4"):
        policy.deny(name)
    if not network:
        for name in ("socket", "connect", "sendto", "sendmsg", "sendmmsg", "recvfrom", "recvmsg", "recvmmsg"):
            policy.deny(name)
    else:
        # Only IPv4/IPv6 TCP and UDP, including CLOEXEC/NONBLOCK socket flags.
        for family in range(46):
            if family not in {2, 10}:
                policy.deny("socket", _SeccompComparison(0, _SCMP_CMP_EQ, family, 0))
        policy.deny("socket", _SeccompComparison(0, _SCMP_CMP_GE, 46, 0))
        for kind in range(16):
            if kind not in {1, 2}:
                policy.deny("socket", _SeccompComparison(1, _SCMP_CMP_MASKED_EQ, 0xF, kind))
        for protocol in range(256):
            if protocol not in {0, 6, 17}:
                policy.deny("socket", _SeccompComparison(2, _SCMP_CMP_EQ, protocol, 0))
        policy.deny("socket", _SeccompComparison(2, _SCMP_CMP_GE, 256, 0))
