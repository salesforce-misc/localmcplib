# Linux sandbox support

The Linux backend uses bubblewrap for filesystem and process isolation,
Landlock for executable and filesystem permissions, and libseccomp for syscall
restrictions. It fails with `SandboxError` when required enforcement cannot be
established; it never falls back to running a command outside the sandbox.

## Installation and usage

On Debian or Ubuntu, install the system dependencies:

```sh
sudo apt-get install bubblewrap python3 libseccomp2 libc-bin file
```

The host must permit unprivileged user namespaces and enable Landlock ABI 3
or newer. Container environments must also permit the namespace and mount
operations bubblewrap needs. Presence of the executable alone does not prove
the host supports sandbox execution: setup is checked on every invocation.
The initial implementation targets x86-64 and AArch64 glibc systems with a
merged `/usr`, where `/bin`, `/sbin`, `/lib`, and `/lib64` are links into
`/usr`, as on current Debian, Ubuntu, and Fedora releases. Hosts where any of
them is a separate directory are rejected. System tools are found under
`/usr/bin`, `/bin`, `/usr/sbin`, and `/sbin`.

```python
from pathlib import Path

from localmcp.sandbox import RootAccess, SandboxProfile, SandboxRoot, create_sandbox

profile = SandboxProfile(
    roots=(SandboxRoot(Path("/path/to/workspace"), RootAccess.READ_WRITE),),
)
sandbox = create_sandbox(profile)
result = await sandbox.run("find . -type f | head")
```

`localmcp.sandbox.sandbox_tools` exposes the same budgeted `Bash` tool on both
platforms. Code that needs a specific backend can import `LinuxSandbox` from
`localmcp.sandbox.bubblewrap` or `MacOSSandbox` from `localmcp.sandbox.seatbelt`.

## Enforcement and shared behavior

The first root is the command's working directory. Root normalization,
conflicting grants, nested read-only protection, denied-path alias checks,
command validation, environment construction, and bounded output collection
share code with the macOS backend. Denied and protected paths are checked
again before each command, including host-created hard links.

Writable roots expose host writes directly. Read-only mounts also prevent
metadata changes such as `chmod`. Protected and denied ancestors are pinned
as mount points so commands cannot rename them to evade the policy. Denied
paths that exist are covered by inaccessible, read-only mounts; a placeholder
file covers any non-directory, such as a FIFO or socket.

A missing denied path cannot be covered by a mount, so in a read-only
directory its nearest existing ancestor is rebuilt for the command: an empty,
read-only tmpfs holding a read-only mount of every other entry, with symlinks
recreated rather than followed. A denied path the host creates while the
command runs therefore stays hidden, and the next command covers it. Beneath a
file, the file itself is pinned instead.

Every invocation receives a private temporary directory outside the roots.
It is writable only when at least one root is writable, and is removed after
execution. Synthetic directories and other filesystem surfaces do not provide
an additional writable temporary directory.

Only the shell, selected tools, their own helper directories, and required
ELF loaders receive execution grants. Shared-library mappings are resolved
with the system `ldd` using a clean environment, exclusively for trusted system
programs. Other mounts prohibit execution and executable file mappings,
including programs copied into a workspace. Tool resolution ignores the
caller's `PATH` and environment.

The trusted setup launcher temporarily holds namespace-local mount capabilities
to apply restrictions. It drops capabilities before starting the workload.
The workload cannot create or join namespaces, change mounts, trace its
supervisor, or use anonymous executable files or io_uring to bypass policy.

The launcher supervises the shell as PID 1 in a private PID namespace. It
returns the shell's status and exits when the shell finishes, causing the
kernel to terminate remaining descendants. Timeouts, output overflow, and
cancellation also terminate the namespace before the call returns. Concurrent
commands have independent namespaces and temporary directories.

## Current differences from macOS

This is an initial Linux implementation, with explicit restrictions where
Seatbelt semantics cannot yet be reproduced:

- Denied paths are planned from the host as each command starts. A missing
  denied path in a directory that commands can write is not hidden: as on
  macOS, a command can create it, but unlike Seatbelt that command, and any
  command already running, can read what it holds until it exits. The next
  command covers it. Linux denied mounts also block writes; Seatbelt's current
  denied-path rules specifically block reads and unlink operations.
- A read-only directory rebuilt to hide a missing denied path is fixed for the
  duration of the command. Entries the host adds while it runs stay hidden, a
  file the host replaces with a new one keeps its earlier contents, and the
  directory itself reports mode `0755`. Each such directory may hold at most
  1,024 other entries; beyond that, the command is rejected. An entry the host
  replaces while the command is being set up also rejects it, rather than
  mounting what the replacement names.
- A script tool requires its shebang interpreter to be explicitly listed in
  the required `tools` list (the shell is always available). Without that grant,
  a required script tool is rejected and an optional one is omitted. This avoids
  silently making an unlisted interpreter
  runnable. Multicall providers such as BusyBox are rejected because a grant
  to their executable would authorize additional commands.
- Separate roots and pinned ancestors are separate mount points. Renames and
  hard links across those boundaries may fail with `EXDEV` even when the host
  paths share a filesystem. Operations inside one unprotected mounted tree
  retain normal filesystem behavior.
- The private user namespace reports UID and GID 0. These map to the caller's
  host identity; files written on the host still belong to that caller. The
  workload has no capabilities, including inside its namespace.
- `network=True` permits outbound IPv4/IPv6 TCP and UDP, with DNS and system
  trust-store reads. It does not permit listeners, raw sockets, or Unix-domain
  sockets. Name resolution uses glibc's built-in `files` and `dns` sources,
  which requires glibc 2.34 or newer. Other NSS modules named in
  `nsswitch.conf`, such as `resolve` or `mdns`, are not loaded; lookups
  continue with the next source as `nsswitch.conf` directs. With networking
  disabled, socket creation is denied in addition to network namespace
  isolation, so private loopback is not an escape hatch.
- `ipc=True` enables POSIX and SysV shared memory in a private IPC namespace;
  it does not expose host shared-memory objects or enable semaphores, message
  queues, or sockets. POSIX shared memory is an explicit writable memory-backed
  surface, even for otherwise read-only profiles.
- Tools requiring additional runtime data or dynamically loaded modules may
  need dedicated grants. User-local tool installations are not discovered.
  Utility flags, diagnostics, and GNU/BSD output differences remain platform
  specific.

The CPU, wall-clock, output, file-descriptor, and process-count limits follow
the macOS defaults. Linux counts processes within each command's private user
namespace, so a command may run up to 256 processes, including its shell,
however many the caller already runs; on macOS the allowance is added to the
caller's current process count. The caller's own per-user limit still applies.
Neither is a cgroup quota. Filesystem writes are not size-limited.

## Validation

`make test-bubblewrap` requires real Linux enforcement and fails instead of
skipping when prerequisites are missing. The ordinary suite skips Linux
enforcement tests on other platforms. CI and release workflows run Linux
enforcement separately from macOS Seatbelt enforcement.
