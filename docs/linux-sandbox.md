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
The initial implementation targets x86-64 and AArch64 glibc systems with system tools under
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
files and directories are covered by inaccessible, read-only mounts.

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

- A denied path must already exist as a regular file or directory. Missing
  denied paths are rejected before execution. Linux denied mounts also block
  writes; Seatbelt's current denied-path rules specifically block reads and
  unlink operations.
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
  sockets. With networking disabled, socket creation is denied in addition to
  network namespace isolation, so private loopback is not an escape hatch.
- `ipc=True` enables POSIX and SysV shared memory in a private IPC namespace;
  it does not expose host shared-memory objects or enable semaphores, message
  queues, or sockets. POSIX shared memory is an explicit writable memory-backed
  surface, even for otherwise read-only profiles.
- Tools requiring additional runtime data or dynamically loaded modules may
  need dedicated grants. User-local tool installations are not discovered.
  Utility flags, diagnostics, and GNU/BSD output differences remain platform
  specific.

The CPU, wall-clock, output, and file-descriptor limits follow the macOS
defaults. The process-count allowance is best effort and per user, as with
the macOS implementation; it is not a cgroup quota. Filesystem writes are
not size-limited.

## Validation

`make test-bubblewrap` requires real Linux enforcement and fails instead of
skipping when prerequisites are missing. The ordinary suite skips Linux
enforcement tests on other platforms. CI and release workflows run Linux
enforcement separately from macOS Seatbelt enforcement.
