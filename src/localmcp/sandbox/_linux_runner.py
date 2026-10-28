"""Trusted PID-namespace supervisor, run by the sandbox's system Python.

This file deliberately has no localmcp or third-party Python imports. It is
mounted into an otherwise isolated filesystem and invoked with ``python -I -S``.
No workload runs until mount, capability, Landlock, seccomp, and resource-limit
setup has succeeded. Exiting namespace PID 1 terminates every descendant.
"""

from __future__ import annotations

import ctypes
import errno
import json
import os
import resource
import signal
import sys
from typing import Any

_STATUS_PATH = "/localmcp/status"
_CONTROL_DIRECTORY = "/localmcp"
_CHILD_READY = b"ready\n"

_MS_RDONLY = 1
_MS_NOSUID = 2
_MS_NODEV = 4
_MS_NOEXEC = 8
_MS_REMOUNT = 32
_MS_BIND = 4096
_MOUNT_OPTIONS = {
    "ro": _MS_RDONLY,
    "nosuid": _MS_NOSUID,
    "nodev": _MS_NODEV,
    "noexec": _MS_NOEXEC,
    "sync": 16,
    "mand": 64,
    "dirsync": 128,
    "noatime": 1024,
    "nodiratime": 2048,
    "relatime": 1 << 21,
    "strictatime": 1 << 24,
    "lazytime": 1 << 25,
}

_PR_SET_DUMPABLE = 4
_PR_CAPBSET_READ = 23
_PR_CAPBSET_DROP = 24
_PR_SET_NO_NEW_PRIVS = 38
_PR_CAP_AMBIENT = 47
_PR_CAP_AMBIENT_CLEAR_ALL = 4
_LINUX_CAPABILITY_VERSION_3 = 0x20080522

_LANDLOCK_CREATE_RULESET_VERSION = 1
_LANDLOCK_RULE_PATH_BENEATH = 1
_LANDLOCK_EXECUTE = 1 << 0
_LANDLOCK_WRITE_FILE = 1 << 1
_LANDLOCK_READ_DIR = 1 << 3
_LANDLOCK_REFER = 1 << 13
_LANDLOCK_TRUNCATE = 1 << 14
_LANDLOCK_HANDLED = _LANDLOCK_EXECUTE | _LANDLOCK_WRITE_FILE | _LANDLOCK_READ_DIR | _LANDLOCK_REFER | _LANDLOCK_TRUNCATE

_SCMP_ACT_ALLOW = 0x7FFF0000
_SCMP_ACT_ERRNO = 0x00050000
_SCMP_CMP_EQ = 4
_SCMP_CMP_GE = 5
_SCMP_CMP_MASKED_EQ = 7
_CLONE_NAMESPACE_FLAGS = (0x80, 0x20000, 0x2000000, 0x4000000, 0x8000000, 0x10000000, 0x20000000, 0x40000000)


class _CapabilityHeader(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]


class _CapabilityData(ctypes.Structure):
    _fields_ = [("effective", ctypes.c_uint32), ("permitted", ctypes.c_uint32), ("inheritable", ctypes.c_uint32)]


class _LandlockRuleset(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _LandlockPath(ctypes.Structure):
    # TODO: Python 3.14 deprecates _pack_ without an explicit _layout_, and 3.19 will
    # reject it. This runs under the host's /usr/bin/python3, so the sandbox would fail
    # to start there. Setting _layout_ = "ms" likely fixes it; confirm the struct stays
    # 12 bytes (packed landlock_path_beneath_attr) and that older Pythons ignore it.
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


class _SeccompComparison(ctypes.Structure):
    _fields_ = [
        ("arg", ctypes.c_uint),
        ("op", ctypes.c_int),
        ("datum_a", ctypes.c_uint64),
        ("datum_b", ctypes.c_uint64),
    ]


def _libc() -> ctypes.CDLL:
    library = ctypes.CDLL(None, use_errno=True)
    library.mount.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_ulong, ctypes.c_void_p]
    library.mount.restype = ctypes.c_int
    library.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    library.prctl.restype = ctypes.c_int
    library.capset.argtypes = [ctypes.POINTER(_CapabilityHeader), ctypes.POINTER(_CapabilityData)]
    library.capset.restype = ctypes.c_int
    # syscall is variadic; arguments after the syscall number are explicitly typed.
    library.syscall.argtypes = [ctypes.c_long]
    library.syscall.restype = ctypes.c_long
    return library


def _checked(result: int, operation: str) -> int:
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, f"{operation}: {os.strerror(error)}")
    return result


def _prctl(library: ctypes.CDLL, option: int, argument: int = 0) -> None:
    _checked(int(library.prctl(option, argument, 0, 0, 0)), "prctl")


def _within(path: str, directory: str) -> bool:
    return path == directory or path.startswith(directory.rstrip("/") + "/")


def _mount_path(value: str) -> str:
    # mountinfo escapes these four characters, including literal backslashes.
    for escaped, literal in (("\\040", " "), ("\\011", "\t"), ("\\012", "\n"), ("\\134", "\\")):
        value = value.replace(escaped, literal)
    return value


def _visible_mount_id(path: str) -> int | None:
    """Identify the currently reachable mount, ignoring hidden overmounts."""
    try:
        descriptor = os.open(path, getattr(os, "O_PATH", 0x200000) | os.O_CLOEXEC)
    except (FileNotFoundError, NotADirectoryError):
        return None
    try:
        with open(f"/proc/self/fdinfo/{descriptor}", encoding="ascii") as stream:
            for line in stream:
                if line.startswith("mnt_id:"):
                    return int(line.split(":", 1)[1])
    finally:
        os.close(descriptor)
    raise RuntimeError(f"cannot identify visible sandbox mount: {path}")


def _mounts() -> list[tuple[str, int]]:
    mounts: dict[str, int] = {}
    with open("/proc/self/mountinfo", encoding="utf-8", errors="surrogateescape") as stream:
        for line in stream:
            fields = line.split()
            separator = fields.index("-")
            mountpoint = _mount_path(fields[4])
            # Later parent binds can hide earlier children without removing
            # their mountinfo records. Remounting those paths would target
            # something else, sometimes a regular file rather than a mount.
            if _visible_mount_id(mountpoint) != int(fields[0]):
                continue
            options = set(fields[5].split(",")) | set(fields[separator + 3].split(","))
            flags = mounts.get(mountpoint, 0)
            for option, flag in _MOUNT_OPTIONS.items():
                if option in options:
                    flags |= flag
            mounts[mountpoint] = flags
    if "/" not in mounts:
        raise RuntimeError("sandbox root mount is missing")
    return sorted(mounts.items(), key=lambda item: len(item[0].split("/")), reverse=True)


def _check_entry_identities(profile: dict[str, Any]) -> None:
    """Fail unless each entry mounted into a rebuilt directory is the one the host listed.

    bwrap resolves each entry again when it mounts it. One the host replaced in between,
    such as with a symlink to elsewhere, would mount whatever the replacement names.
    """
    for path, (device, inode) in profile.get("entry_identities", {}).items():
        try:
            info = os.lstat(path)
        except FileNotFoundError:
            # Removed since it was listed, so bwrap mounted nothing.
            continue
        if (info.st_dev, info.st_ino) != (device, inode):
            raise RuntimeError(f"sandbox entry changed while the sandbox was being prepared: {path}")


def _lock_mounts(profile: dict[str, Any], library: ctypes.CDLL) -> None:
    executables = set(profile["executables"]) | set(profile.get("executable_maps", ()))
    executable_trees = tuple(profile["executable_trees"])
    readonly = tuple(profile["readonly_mounts"])
    devices = {"/dev", "/dev/null", "/dev/zero", "/dev/random", "/dev/urandom"}
    mounts = _mounts()
    for mountpoint, previous in mounts:
        flags = previous | _MS_BIND | _MS_REMOUNT | _MS_NOSUID
        if mountpoint not in devices:
            flags |= _MS_NODEV
        if mountpoint in executables or any(_within(mountpoint, tree) for tree in executable_trees):
            flags &= ~_MS_NOEXEC
        else:
            flags |= _MS_NOEXEC
        read_depth = max((len(path) for path in readonly if _within(mountpoint, path)), default=-1)
        write_depth = max(
            (len(path) for path in profile.get("write_paths", ()) if _within(mountpoint, path)), default=-1
        )
        # An explicitly writable root may sit inside an ordinary read-only
        # root. A protected read-only subtree is more specific than its outer
        # writable root, and therefore still locks every mount beneath it.
        if read_depth >= 0 and read_depth >= write_depth:
            flags |= _MS_RDONLY
        # Root is locked after the control directory becomes inaccessible.
        if mountpoint == "/":
            continue
        _checked(int(library.mount(None, os.fsencode(mountpoint), None, flags, None)), f"lock mount {mountpoint}")
    os.chmod(_CONTROL_DIRECTORY, 0)
    root_flags = dict(mounts)["/"] | _MS_BIND | _MS_REMOUNT | _MS_RDONLY | _MS_NOSUID | _MS_NODEV | _MS_NOEXEC
    _checked(int(library.mount(None, b"/", None, root_flags, None)), "lock sandbox root")


def _drop_capabilities(library: ctypes.CDLL) -> None:
    # CAP_SETPCAP is retained only for trusted setup. Dropping a capability from
    # the bounding set does not remove it from the current effective set, so it
    # can safely be dropped before the remaining bounding capabilities.
    for capability in range(64):
        present = int(library.prctl(_PR_CAPBSET_READ, capability, 0, 0, 0))
        if present < 0 and ctypes.get_errno() == errno.EINVAL:
            break
        _checked(present, "read capability bounding set")
        if present:
            _prctl(library, _PR_CAPBSET_DROP, capability)
    _prctl(library, _PR_CAP_AMBIENT, _PR_CAP_AMBIENT_CLEAR_ALL)
    header = _CapabilityHeader(_LINUX_CAPABILITY_VERSION_3, 0)
    data = (_CapabilityData * 2)()
    _checked(int(library.capset(ctypes.byref(header), data)), "drop sandbox capabilities")
    _prctl(library, _PR_SET_NO_NEW_PRIVS, 1)
    _prctl(library, _PR_SET_DUMPABLE, 0)


def _landlock(profile: dict[str, Any], library: ctypes.CDLL) -> None:
    # These syscall numbers are shared by the supported x86-64 and AArch64 ABIs.
    if os.uname().machine not in {"x86_64", "aarch64"}:
        raise RuntimeError("sandbox Landlock supports x86_64 and aarch64 only")
    abi = _checked(
        int(library.syscall(444, None, ctypes.c_size_t(0), ctypes.c_uint(_LANDLOCK_CREATE_RULESET_VERSION))),
        "query Landlock ABI",
    )
    if abi < 3:
        raise RuntimeError(f"sandbox requires Landlock ABI >= 3 (available: {abi})")
    ruleset = _LandlockRuleset(_LANDLOCK_HANDLED)
    descriptor = _checked(
        int(library.syscall(444, ctypes.byref(ruleset), ctypes.c_size_t(ctypes.sizeof(ruleset)), ctypes.c_uint(0))),
        "create Landlock ruleset",
    )
    try:
        paths: dict[str, int] = {}
        for path in (*profile["executables"], *profile["executable_trees"]):
            paths[path] = paths.get(path, 0) | _LANDLOCK_EXECUTE
        for path in profile["read_directories"]:
            paths[path] = paths.get(path, 0) | _LANDLOCK_READ_DIR
        for path in profile["write_paths"]:
            access = _LANDLOCK_WRITE_FILE | _LANDLOCK_TRUNCATE
            if os.path.isdir(path):
                access |= _LANDLOCK_REFER
            paths[path] = paths.get(path, 0) | access
        for path, access in paths.items():
            path_fd = os.open(path, getattr(os, "O_PATH", 0x200000) | os.O_CLOEXEC)
            try:
                rule = _LandlockPath(access, path_fd)
                _checked(
                    int(
                        library.syscall(
                            445,
                            ctypes.c_int(descriptor),
                            ctypes.c_int(_LANDLOCK_RULE_PATH_BENEATH),
                            ctypes.byref(rule),
                            ctypes.c_uint(0),
                        )
                    ),
                    f"add Landlock rule for {path}",
                )
            finally:
                os.close(path_fd)
        _checked(int(library.syscall(446, ctypes.c_int(descriptor), ctypes.c_uint(0))), "enforce Landlock ruleset")
    finally:
        os.close(descriptor)


class _Seccomp:
    def __init__(self) -> None:
        self.library = ctypes.CDLL("libseccomp.so.2", use_errno=True)
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
        self.library.seccomp_load.argtypes = [ctypes.c_void_p]
        self.library.seccomp_load.restype = ctypes.c_int
        self.context = self.library.seccomp_init(_SCMP_ACT_ALLOW)
        if not self.context:
            raise RuntimeError("cannot initialize sandbox seccomp filter")

    def deny(self, name: str, *comparisons: _SeccompComparison, error: int = errno.EPERM) -> None:
        number = int(self.library.seccomp_syscall_resolve_name(name.encode("ascii")))
        if number == -1:
            return  # This syscall does not exist on the current architecture.
        arguments = (_SeccompComparison * len(comparisons))(*comparisons)
        result = int(
            self.library.seccomp_rule_add_array(
                self.context, _SCMP_ACT_ERRNO | error, number, len(comparisons), arguments
            )
        )
        if result < 0:
            raise OSError(-result, f"cannot restrict syscall {name}: {os.strerror(-result)}")

    def load(self) -> None:
        result = int(self.library.seccomp_load(self.context))
        if result < 0:
            raise OSError(-result, f"cannot enforce sandbox seccomp filter: {os.strerror(-result)}")

    def close(self) -> None:
        self.library.seccomp_release(self.context)


def _seccomp(profile: dict[str, Any], policy: _Seccomp) -> None:
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
    if not profile["ipc"]:
        for name in ("shmget", "shmat", "shmdt", "shmctl"):
            policy.deny(name)
    for name in ("socketpair", "bind", "listen", "accept", "accept4"):
        policy.deny(name)
    if not profile["network"]:
        for name in ("socket", "connect", "sendto", "sendmsg", "sendmmsg", "recvfrom", "recvmsg", "recvmmsg"):
            policy.deny(name)
    else:
        # Only IPv4/IPv6 stream/datagram sockets, including CLOEXEC/NONBLOCK flags.
        # Enumerating the other families avoids unsupported duplicate comparisons
        # on one argument. All future address families are denied by the upper bound.
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
    policy.load()


def _limits(profile: dict[str, Any]) -> None:
    cpu = max(1, int(profile["timeout_seconds"]))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    maximum = int(profile.get("max_open_files", 256))
    resource.setrlimit(resource.RLIMIT_NOFILE, (maximum, maximum))
    if profile["process_limit"] is not None:
        maximum = int(profile["process_limit"])
        resource.setrlimit(resource.RLIMIT_NPROC, (maximum, maximum))


def _write(descriptor: int, data: bytes) -> None:
    while data:
        count = os.write(descriptor, data)
        if count <= 0:
            raise RuntimeError("sandbox status pipe closed during write")
        data = data[count:]


def _status(descriptor: int, result: dict[str, object]) -> None:
    _write(descriptor, json.dumps(result).encode("utf-8") + b"\n")


def _close_other_fds(keep: int) -> None:
    for entry in os.listdir("/proc/self/fd"):
        descriptor = int(entry)
        if descriptor > 2 and descriptor != keep:
            try:
                os.close(descriptor)
            except OSError as exc:
                if exc.errno != errno.EBADF:
                    raise


def _child(profile: dict[str, Any], library: ctypes.CDLL, policy: _Seccomp, error_fd: int) -> None:
    try:
        _close_other_fds(error_fd)
        _limits(profile)
        _landlock(profile, library)
        _seccomp(profile, policy)
        policy.close()
        # Python ignores these at startup, and an ignored signal stays ignored across
        # exec; restore the defaults so a closed pipe ends its writer quietly.
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
        signal.signal(signal.SIGXFSZ, signal.SIG_DFL)
        # The explicit marker distinguishes an early setup crash from a
        # successful exec closing the CLOEXEC pipe. An exec failure appends its
        # error to the marker and is still reported as a setup failure.
        _write(error_fd, _CHILD_READY)
        os.execv("/bin/sh", ["/bin/sh", "-c", profile["command"], profile["process_label"]])
    except BaseException as exc:
        try:
            _write(error_fd, f"{type(exc).__name__}: {exc}".encode("utf-8", errors="replace")[:8192])
        finally:
            os._exit(125)


def main() -> None:
    status_fd: int | None = None
    try:
        if sys.platform != "linux" or os.getpid() != 1:
            raise RuntimeError("sandbox runner must run as Linux namespace PID 1")
        status_fd = os.open(_STATUS_PATH, os.O_WRONLY | os.O_TRUNC | os.O_CLOEXEC)
        if len(sys.argv) != 2:
            raise RuntimeError("sandbox runner requires one profile path")
        with open(sys.argv[1], encoding="utf-8") as stream:
            profile: dict[str, Any] = json.load(stream)
        library = _libc()
        # Load libseccomp and all Python extensions before making their mounts noexec.
        policy = _Seccomp()
        _check_entry_identities(profile)
        _lock_mounts(profile, library)
        _drop_capabilities(library)
        read_fd, error_fd = os.pipe2(os.O_CLOEXEC)
        child = os.fork()
        if child == 0:
            os.close(read_fd)
            os.close(status_fd)
            _child(profile, library, policy, error_fd)
            os._exit(125)
        os.close(error_fd)
        policy.close()
        try:
            setup_error = bytearray()
            while chunk := os.read(read_fd, 8192):
                setup_error.extend(chunk)
                if len(setup_error) > 8192:
                    raise RuntimeError("sandbox child sent an invalid setup result")
        finally:
            os.close(read_fd)
        _, wait_status = os.waitpid(child, 0)
        if setup_error != _CHILD_READY:
            if setup_error.startswith(_CHILD_READY):
                del setup_error[: len(_CHILD_READY)]
            message = setup_error.decode("utf-8", errors="replace") or "sandbox child did not confirm setup"
            _status(status_fd, {"error": message})
            os._exit(1)
        _status(status_fd, {"exit_code": os.waitstatus_to_exitcode(wait_status)})
        # Namespace teardown kills detached children too, before the outer bwrap exits.
        os._exit(0)
    except BaseException as exc:
        message = f"{type(exc).__name__}: {exc}"
        if status_fd is not None:
            try:
                _status(status_fd, {"error": message})
            except OSError:
                pass
        else:
            _write(2, f"sandbox setup failed: {message}\n".encode("utf-8", errors="replace"))
        os._exit(1)


if __name__ == "__main__":
    main()
