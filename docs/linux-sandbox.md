# Linux sandbox support

The Linux backend uses bubblewrap for filesystem and process isolation and
libseccomp for syscall restrictions. A shared host-side Bash check validates
submitted command names and syntax on both Linux and macOS. Setup failures
raise `SandboxError`; commands never fall back to unsandboxed execution.

## Requirements and discovery

Install bubblewrap 0.6.3 or newer, Bash, and libseccomp. For Debian or Ubuntu:

```sh
sudo apt-get install bubblewrap bash libseccomp2
```

Install the commands your profiles require too, for example `git`, `file`, or
`ripgrep`. These are workload tools, not sandbox setup dependencies. The host
must permit unprivileged user namespaces and the mount, PID, IPC, UTS, cgroup, and
network namespaces bubblewrap needs. Containers must permit those operations
too. No Python, Landlock, `ldd`, `file`, or merged `/usr` is required inside
the sandbox. Python still runs localmcplib on
the host, where `/proc` supplies task counts for resource-limit preparation.

Discovery happens once per sandbox. Bubblewrap is found in system directories
first, then Nix profiles. Tools and Bash are found on a fixed sandbox PATH:
resolved Nix profile `bin` directories, `/usr/bin`, `/bin`, `/usr/sbin`, `/sbin`.
The caller's PATH is ignored. Supported Nix profiles are
`/nix/var/nix/profiles/default`,
`/nix/var/nix/profiles/per-user/<account>/profile`, `/run/current-system/sw`,
and the caller's `.nix-profile` and `.local/state/nix/profiles/profile` beneath their account's
home directory. Account-specific profiles are skipped when the caller's UID has
no account entry, as in some containers. Only profiles resolving into `/nix/store` are used. Profile
links are recreated to point directly into the store, so absolute shebangs
resolve without exposing Nix's mutable profile-management directories.
Bubblewrap itself runs on the host, so its resolved executable need not be
inside a workload mount. Writable roots cannot include that host launcher.

Libseccomp is loaded by absolute path, preferring standard system library
directories and then the resolved profiles' `lib` directories. It must know
every syscall in the policy; an unknown name rejects construction. Known
architecture-specific pseudo syscall numbers are handled by libseccomp.
The native BPF filter is compiled once per sandbox and supplied to bubblewrap
through a separate descriptor per invocation. The library is never loaded
inside the sandbox. Both x86-64 and AArch64 are supported.

## Usage

```python
from pathlib import Path

from localmcp.sandbox import RootAccess, SandboxProfile, SandboxRoot, create_sandbox

profile = SandboxProfile(
    roots=(SandboxRoot(Path("/path/to/workspace"), RootAccess.READ_WRITE),),
)
sandbox = create_sandbox(profile)
result = await sandbox.run("find . -type f | head")
```

`sandbox_tools` exposes the same budgeted `Bash` tool on both platforms.
`LinuxSandbox` can also be imported from `localmcp.sandbox.bubblewrap`.
Required tools must be installed; missing optional tools are omitted from
`.tools` and the generated Bash tool description.
Executables are resolved during discovery. A required tool whose symlink
target lies outside the mounted runtime directories rejects construction;
an optional one with such a target is omitted.

## Shared command check

The host parses commands with tree-sitter and tree-sitter-bash. Parse errors,
missing nodes, unsupported syntax, and uncheckable command names fail closed.
Every command name must be a bare literal word without a slash and must name
a granted tool or a supported builtin. Nested `$(...)` substitutions are
checked recursively, including those in arguments and redirection targets.

Supported syntax includes lists, pipelines, ordinary quoting, redirections
and here-strings, substitutions, scalar assignments, subshells, groups,
`for`, `while`, `until`, `if`, and `case`. Builtins include `:`, `true`,
`false`, `echo`, `printf`, `pwd`, `cd`, basic `test`/`[ ... ]`, `read`,
`exit`, `break`, `continue`, `wait`, and `kill`. Builtins that assign variables
or accept expressions have additional argument checks: `printf` requires a
literal format and excludes `-v` and `%n`; `read` accepts ordinary variable
names, optionally after `-r`; test operands that expand must be quoted.
`command` supports only literal `-v`/`-V` lookups. Assignments and loop
variables cannot alter shell settings such as `PATH`, `IFS`, or `BASH_ENV`.
They also cannot change loader variables (`LD_*`/`DYLD_*`) or the fixed
environment, including `HOME`, `TMPDIR`, and Git configuration. Numeric shell
variables that evaluate assignments, such as `RANDOM` and `OPTIND`, are blocked.

Functions, `eval`, `source`, `exec`, dispatch through `command` or `builtin`,
shell-option changes, traps, arithmetic, indirect or evaluating parameter
expansions, here-documents, backticks, `[[ ... ]]`, and process substitutions
are unsupported. Paths such as `/bin/cat` and `./script`, quoted command names,
and dynamic names such as `$tool` are rejected even if the program is granted.

Rejection returns `CommandResult(exit_code=126, stdout="", stderr="localmcp: ...")`
before any part of the submitted command runs. Existing blank, NUL, and size
validation still raises `SandboxError`.

The check does not inspect arguments interpreted as code by a granted tool.
On Linux, granted tools may execute other programs available inside the
sandbox. For example, `awk`, `find -exec`, `xargs`, an interpreter, or Git
helpers may run unlisted tools or repository code. Tool names are a shell
policy, not an OS execution allowlist. macOS retains Seatbelt's additional
executable restrictions.

## Filesystem and process isolation

The first root is the working directory. Roots are bound read-only or
read-write as declared. Nested protected roots are rebound read-only, with
ancestors pinned as mount points to prevent renames from evading protection.
Separate mounts can make renames and hard links fail with `EXDEV` across
boundaries, even when their host paths share a filesystem.

Runtime reads extend beyond roots to read-only `/usr`, `/nix/store` when it
exists, and a curated `/etc`: loader cache, alternatives, passwd/group,
localtime, and OS release information. `/bin`, `/lib`, `/lib64`, and `/sbin`
are symlinks or read-only binds matching the host. Network mode adds resolver
configuration and system trust stores. Roots overlapping runtime mounts or
recreated Nix profile links are rejected.

Existing denied paths are covered by empty, inaccessible read-only
placeholders; one file placeholder can cover a FIFO or socket too. Missing
denied paths are not covered during the current command: if the host or a
command creates one, it becomes hidden from the next command onward. This
also applies in read-only roots when the host creates a path. Paths are
planned afresh each run; directories are not rebuilt or snapshotted. Denied
and protected paths are rechecked for symlinks and hard links before launch.
Concurrent changes by another host process are outside this path-based
policy's guarantees; callers must keep policy paths stable during setup.

Every command receives a private tmpfs at its unique scratch path, writable
only if a root is writable. `/dev` provides standard devices, with directory
writes disabled. Synthetic ancestors and the sandbox root filesystem are
read-only. `/proc` is not mounted, so `ps`, `/proc/self`, `/dev/fd`-dependent
operations, and similar facilities are unavailable. With `ipc=True`, a
separate writable tmpfs at `/dev/shm` enables POSIX shared memory even for
otherwise read-only profiles.

Managed Git worktrees need both the worktree root and a root granting their
common Git metadata directory. The backend does not discover or grant Git
metadata automatically; the trusted application constructing the profile
must grant it explicitly.

Commands run as the caller's UID/GID in private user, mount, PID, IPC, UTS and cgroup
namespaces, with no capabilities, a new session, and parent-death teardown.
The environment comes only from `command_environment`, supplied through
`--clearenv` and `--setenv`. Bash is invoked explicitly without startup files.

## Network, IPC, limits, and teardown

Networking defaults off: a private network namespace and seccomp deny socket
operations. `network=True` permits outbound IPv4/IPv6 TCP and UDP, DNS, and
system trust-store reads. It still denies listeners, raw sockets and Unix
sockets. `ipc=True` permits POSIX and SysV shared memory in a private IPC
namespace, not semaphores, message queues, or sockets. Seccomp also blocks
namespace/mount changes, tracing, io_uring, and the other restricted syscalls.

CPU, file-descriptor and process limits are applied by a small isolated
host-side Python launcher before it execs bubblewrap. Its source is fixed in
memory; it loads no site packages, and no Python callback runs between fork
and exec in the multithreaded parent. Linux starts the current interpreter
through host `/proc/self/exe`, so replacing a virtualenv executable alias
cannot replace the launcher. Writable roots overlapping its base Python
installation are rejected. This launcher does not run inside the sandbox.
The process allowance is recalculated for every command and includes host
tasks already owned by the caller, counting threads, plus up to 256 additional
tasks, capped by the caller's inherited soft limit. This is an inherited
per-user rlimit, not a per-command cgroup quota; concurrent host activity and
kernel user-namespace accounting affect the effective allowance. Writes have
no size quota. Wall-clock time and combined output limits are also enforced.
Tools that select workers from the host CPU count can need explicit bounds,
for example `git -c grep.threads=2 grep ...` on machines with many CPUs.

Bubblewrap's namespace init supervises the workload; namespace teardown kills
remaining descendants when it exits. On timeout, output overflow, or
cancellation, the host kills the init PID reported by `--info-fd`, then the
process group. Each invocation has independent namespaces, scratch and FDs.
Both backends report the shell convention `128 + signal` for signal
termination and preserve explicit exit statuses in that range. Timeout and
output-limit aborts return `-1`, with `timed_out` or `truncated` set.
Linux obtains the workload status from bubblewrap. `--json-status-fd`,
supported by both tested versions (0.6.3 and 0.12.0), distinguishes successful
workload execution from sandbox setup/exec failures,
which raise `SandboxError`.

## Validation

`make test-bubblewrap` requires real enforcement and fails instead of skipping
when Linux prerequisites are missing. `LOCALMCP_REQUIRE_BUBBLEWRAP=1 uv run
pytest` runs the full suite with enforcement required. Tests include filesystem,
denied-path, network, IPC, resource-limit and descendant teardown checks,
managed Git worktrees, ordinary review commands, and a startup timing check.
Portable unit tests cover Bash acceptance/rejection, system/Nix discovery,
argv construction, concurrent FDs, and seccomp failures.
