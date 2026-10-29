"""Host-side limits applied before launching the Linux sandbox."""

from __future__ import annotations

import json
import resource
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from localmcp.sandbox import MAX_ADDITIONAL_PROCESSES, MAX_OPEN_FILES, RootAccess, SandboxError, SandboxRoot, runtime

_READ_LIMITS = (
    "import json, resource, sys; "
    "print(json.dumps({'cpu': resource.getrlimit(resource.RLIMIT_CPU), "
    "'files': resource.getrlimit(resource.RLIMIT_NOFILE), "
    "'processes': resource.getrlimit(resource.RLIMIT_NPROC), 'arguments': sys.argv[1:]}))"
)


@pytest.mark.parametrize("process_limit", [None, "inherited"])
def test_limited_command_applies_limits_in_a_fresh_child_and_preserves_arguments(process_limit: str | None) -> None:
    limits_before = {
        kind: resource.getrlimit(kind) for kind in (resource.RLIMIT_CPU, resource.RLIMIT_NOFILE, resource.RLIMIT_NPROC)
    }
    inherited_processes = limits_before[resource.RLIMIT_NPROC]
    maximum_processes = None
    if process_limit is not None:
        maximum_processes = inherited_processes[0] if inherited_processes[0] != resource.RLIM_INFINITY else 65536
    arguments = ["literal;$(printf injected)", "space and newline\nargument", "", "--option"]
    command = runtime.limited_command(
        [sys.executable, "-I", "-S", "-c", _READ_LIMITS, *arguments], 5.5, maximum_processes
    )

    result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=True)
    limits_after = {kind: resource.getrlimit(kind) for kind in limits_before}
    observed = json.loads(result.stdout)

    assert limits_after == limits_before
    cpu_hard = limits_before[resource.RLIMIT_CPU][1]
    cpu_maximum = 5 if cpu_hard == resource.RLIM_INFINITY else min(5, cpu_hard)
    file_hard = limits_before[resource.RLIMIT_NOFILE][1]
    file_maximum = MAX_OPEN_FILES if file_hard == resource.RLIM_INFINITY else min(MAX_OPEN_FILES, file_hard)
    assert observed == {
        "cpu": [cpu_maximum, cpu_maximum],
        "files": [file_maximum, file_maximum],
        "processes": list(inherited_processes) if maximum_processes is None else [maximum_processes] * 2,
        "arguments": arguments,
    }


def test_limited_command_clamps_to_inherited_hard_limits_before_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    inherited = {resource.RLIMIT_CPU: (1, 1), resource.RLIMIT_NOFILE: (16, 32)}
    monkeypatch.setattr(resource, "getrlimit", inherited.__getitem__)
    monkeypatch.setattr(resource, "setrlimit", lambda *_args: pytest.fail("the parent must not apply child limits"))
    command = runtime.limited_command([sys.executable, "-I", "-S", "-c", _READ_LIMITS], 10, None)

    result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=True)
    observed = json.loads(result.stdout)

    assert observed["cpu"] == [1, 1]
    assert observed["files"] == [32, 32]


def test_limited_command_never_executes_target_if_a_limit_cannot_be_applied(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    marker = tmp_path / "target-started"
    # An invalid resource exercises the fresh interpreter's actual setup-failure path.
    monkeypatch.setattr(resource, "RLIMIT_CPU", 999999999)
    monkeypatch.setattr(resource, "getrlimit", lambda _kind: (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    command = runtime.limited_command(
        [sys.executable, "-I", "-S", "-c", "import pathlib, sys; pathlib.Path(sys.argv[1]).touch()", str(marker)],
        1,
        None,
    )

    result = subprocess.run(command, capture_output=True, text=True, timeout=10)

    assert result.returncode != 0
    assert result.stderr
    assert not marker.exists()


def test_limited_command_preserves_default_sigpipe_disposition() -> None:
    command = runtime.limited_command(["/bin/sh", "-c", "kill -PIPE $$"], 5, None)

    result = subprocess.run(command, capture_output=True, timeout=10)

    assert result.returncode == -signal.SIGPIPE


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux's running-interpreter handle")
def test_limited_command_cannot_be_redirected_through_a_mutable_interpreter_alias(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    alias = tmp_path / "mutable-python"
    alias.symlink_to(sys.executable)
    marker = alias.with_name(f"{alias.name}.marker")
    attacker = tmp_path / "attacker"
    attacker.write_text('#!/bin/sh\nprintf attacker > "$0.marker"\n')
    attacker.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(alias))

    # A writable environment can retarget the launcher's original executable
    # alias between calls; the next host launch must still use its own interpreter.
    alias.unlink()
    alias.symlink_to(attacker)
    command = runtime.limited_command(["/bin/echo", "safe"], 5, None)
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)

    assert result.returncode == 0, result.stderr
    assert result.stdout == "safe\n"
    assert not marker.exists()


@pytest.mark.parametrize("runtime_index", [0, 1])
@pytest.mark.parametrize("overlap", ["same", "child", "ancestor"])
def test_host_runtime_rejects_writable_roots_overlapping_either_library_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, runtime_index: int, overlap: str
) -> None:
    trees = (tmp_path / "python-base", tmp_path / "python-exec-base")
    monkeypatch.setattr(runtime, "_HOST_RUNTIME_TREES", trees)
    tree = trees[runtime_index]
    path = {"same": tree, "child": tree / "lib" / "python", "ancestor": tree.parent}[overlap]
    roots = (SandboxRoot(tmp_path / "workspace", RootAccess.READ_WRITE), SandboxRoot(path, RootAccess.READ_WRITE))

    with pytest.raises(SandboxError, match="overlaps the host Python runtime"):
        runtime.check_host_runtime(roots)


@pytest.mark.parametrize("overlap", ["same", "child", "ancestor"])
def test_host_runtime_allows_read_only_overlap_and_unrelated_writable_roots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, overlap: str
) -> None:
    tree = tmp_path / "python-runtime"
    monkeypatch.setattr(runtime, "_HOST_RUNTIME_TREES", (tree,))
    path = {"same": tree, "child": tree / "lib" / "python", "ancestor": tree.parent}[overlap]
    roots = (SandboxRoot(tmp_path / "workspace", RootAccess.READ_WRITE), SandboxRoot(path, RootAccess.READ_ONLY))

    runtime.check_host_runtime(roots)


@pytest.mark.parametrize("returncode, expected", [(0, 0), (7, 7), (143, 143), (255, 255), (-15, 143), (-9, 137)])
def test_command_exit_code_uses_shell_signal_convention(returncode: int, expected: int) -> None:
    assert runtime.command_exit_code(returncode) == expected


def test_developer_environment_pins_the_toolchain_and_scratch_cache_over_extra_values(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    developer = tmp_path / "selected-developer"
    extra = {"DEVELOPER_DIR": "/unselected", "xcrun_db": "/outside/cache", "CUSTOM": "retained"}

    environment = runtime.command_environment(scratch, developer_directory=developer, extra=extra)

    assert environment["DEVELOPER_DIR"] == str(developer)
    assert environment["xcrun_db"] == str(scratch / "xcrun_db")
    assert environment["CUSTOM"] == "retained"
    assert extra == {"DEVELOPER_DIR": "/unselected", "xcrun_db": "/outside/cache", "CUSTOM": "retained"}


def test_developer_environment_is_absent_without_a_selected_toolchain(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DEVELOPER_DIR", "/caller/developer")
    monkeypatch.setenv("xcrun_db", "/caller/cache")

    environment = runtime.command_environment(tmp_path)

    assert "DEVELOPER_DIR" not in environment
    assert "xcrun_db" not in environment


@pytest.fixture
def host_proc(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    proc = tmp_path / "proc"
    proc.mkdir()
    monkeypatch.setattr(runtime, "Path", lambda _path: proc)
    monkeypatch.setattr(runtime.os, "getuid", lambda: 1000)
    monkeypatch.setattr(resource, "getrlimit", lambda _kind: (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    return proc


def _task(proc: Path, pid: str, uid: int, threads: int) -> None:
    directory = proc / pid
    directory.mkdir()
    (directory / "status").write_text(f"Name:\ttask\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\nThreads:\t{threads}\n")


def test_owned_process_limit_counts_real_uid_threads_and_tolerates_exits(host_proc: Path) -> None:
    _task(host_proc, "1", 1000, 3)
    _task(host_proc, "2", 1000, 5)
    _task(host_proc, "3", 2000, 99)
    (host_proc / "4").mkdir()  # This process exited before its status could be read.
    (host_proc / "self").mkdir()

    assert runtime.owned_process_limit() == 8 + MAX_ADDITIONAL_PROCESSES


def test_owned_process_limit_respects_the_callers_soft_limit(host_proc: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _task(host_proc, "1", 1000, 8)
    monkeypatch.setattr(resource, "getrlimit", lambda _kind: (20, 100))

    assert runtime.owned_process_limit() == 20


@pytest.mark.parametrize("contents", ["Uid:\t1000\n", "Uid:\t1000\nThreads:\tinvalid\n"])
def test_owned_process_limit_fails_closed_for_unreadable_task_counts(host_proc: Path, contents: str) -> None:
    directory = host_proc / "1"
    directory.mkdir()
    (directory / "status").write_text(contents)

    with pytest.raises(SandboxError, match="count host tasks"):
        runtime.owned_process_limit()


def test_owned_process_limit_fails_closed_without_an_owned_task(host_proc: Path) -> None:
    _task(host_proc, "1", 2000, 3)

    with pytest.raises(SandboxError, match="count host tasks"):
        runtime.owned_process_limit()


def test_owned_process_limit_fails_closed_when_proc_is_unavailable(host_proc: Path) -> None:
    host_proc.rmdir()

    with pytest.raises(SandboxError, match="count host tasks"):
        runtime.owned_process_limit()
