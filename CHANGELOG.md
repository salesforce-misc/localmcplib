# Changelog

## 0.7.0 (unreleased)

### Sandbox compatibility changes

- **Signal exit statuses change on both platforms.** In 0.6.1, a command
  terminated by SIGTERM returned `exit_code=-15`. In 0.7.0 it returns `143`
  (`128 + signal`). Explicit exit statuses remain unchanged, including
  `exit 143`, so a high exit status alone does not prove signal termination.
  Update callers that use negative exit codes to detect killed workloads.
  Timeout and output-limit aborts still return `-1`; use `timed_out` and
  `truncated` to identify those cases.
- Both backends check a conservative Bash syntax subset before launch.
  Command names must be bare literal names for granted tools or supported
  builtins, including inside substitutions. Unsupported syntax, executable
  paths, dynamic command names, and changes to fixed shell, loader, or
  sandbox environment settings reject the entire command with exit code 126.
- Linux now uses bubblewrap directly with a host-built seccomp filter.
  **Tools are checked at the shell level:** granted tools can launch other
  programs or interpret repository code within the sandbox. Linux no longer
  enforces an OS executable allowlist. macOS retains its Seatbelt restrictions.
- Linux exposes read-only `/usr`, curated `/etc` entries, and `/nix/store`
  when present. It does not mount `/proc`. Missing denied paths created during
  a command are covered from the next command onward. Scratch uses tmpfs.

### Requirements and implementation

- Linux requires bubblewrap 0.6.3 or newer, Bash, libseccomp, and permission
  to create the required namespaces, including user and cgroup namespaces.
  System and supported Nix profile installations are discovered. Landlock,
  merged `/usr`, `ldd`, and an in-sandbox Python runner are no longer required.
- Added `tree-sitter` and `tree-sitter-bash` dependencies for the shared check.
- Host task counts are refreshed for every command. A separate isolated host
  launcher applies resource limits without a Python `preexec_fn` callback.
- See [Linux sandbox support](docs/linux-sandbox.md) for policy details,
  supported syntax, platform differences, and validation commands.
