"""Public platform selection and shared model-tool behavior."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from localmcp.sandbox import CommandResult, SandboxError, SandboxProfile, SandboxRoot, create_sandbox, sandbox_tools
from localmcp.sandbox import tools as sandbox_api
from localmcp.sandbox.runtime import command_environment


class _Sandbox:
    tools = ("cat", "rg")

    def __init__(self, profile: SandboxProfile, **options: Any) -> None:
        self.profile = profile
        self.options = options
        self.commands: list[str] = []

    async def run(self, command: str) -> CommandResult:
        self.commands.append(command)
        if command == "fail":
            raise SandboxError("unavailable")
        if command == "cancel":
            raise asyncio.CancelledError
        return CommandResult(exit_code=0, stdout=command, stderr="")


class _Budget:
    def __init__(self, *claims: tuple[bool, str | None]) -> None:
        self.claims = iter(claims)

    async def claim(self) -> tuple[bool, str | None]:
        return next(self.claims)

    def metadata(self, exhausted_scope: str | None = None) -> dict[str, object]:
        return {"scope": exhausted_scope or "available"}


@pytest.mark.parametrize(
    ("platform", "module", "backend"), [("darwin", "seatbelt", "MacOSSandbox"), ("linux", "bubblewrap", "LinuxSandbox")]
)
async def test_factory_selects_backend_and_forwards_policy_and_limits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, platform: str, module: str, backend: str
) -> None:
    selected = ModuleType(f"localmcp.sandbox.{module}")
    setattr(selected, backend, _Sandbox)
    monkeypatch.setitem(sys.modules, selected.__name__, selected)
    monkeypatch.setattr(sandbox_api.sys, "platform", platform)
    profile = SandboxProfile((SandboxRoot(tmp_path),), tools=("cat",), optional_tools=("rg", "missing"))

    sandbox = create_sandbox(profile, timeout_seconds=5, max_output_bytes=123, process_label="test-command")

    assert isinstance(sandbox, _Sandbox)
    assert sandbox.profile is profile
    assert sandbox.options == {"timeout_seconds": 5, "max_output_bytes": 123, "process_label": "test-command"}
    assert (await sandbox.run("hello")).stdout == "hello"


def test_factory_rejects_unsupported_platform(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sandbox_api.sys, "platform", "freebsd")

    with pytest.raises(SandboxError, match="unsupported on platform: freebsd"):
        create_sandbox(SandboxProfile((SandboxRoot(tmp_path),)))


def test_factory_preserves_backend_setup_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def unavailable(*_args: Any, **_kwargs: Any) -> None:
        raise SandboxError("required isolation unavailable")

    module = ModuleType("localmcp.sandbox.bubblewrap")
    module.LinuxSandbox = unavailable  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(sandbox_api.sys, "platform", "linux")

    with pytest.raises(SandboxError, match="required isolation unavailable"):
        create_sandbox(SandboxProfile((SandboxRoot(tmp_path),)))


async def test_generic_tool_uses_resolved_grants_and_preserves_budget_results(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    profile = SandboxProfile((SandboxRoot(tmp_path),), tools=("cat",), optional_tools=("rg", "missing"))
    backend = _Sandbox(profile)
    calls: list[tuple[SandboxProfile, dict[str, Any]]] = []

    def construct(profile: SandboxProfile, **options: Any) -> _Sandbox:
        calls.append((profile, options))
        return backend

    monkeypatch.setattr(sandbox_api, "create_sandbox", construct)
    tool = sandbox_tools(
        profile,
        budget=_Budget((False, "operation"), (True, None), (True, None)),
        timeout_seconds=7,
        max_output_bytes=321,
        process_label="portable-command",
        exhausted_message="done",
    )[0]

    assert tool.name == "Bash"
    assert "cat, rg" in tool.description
    assert "missing" not in tool.description
    assert calls == [(profile, {"timeout_seconds": 7, "max_output_bytes": 321, "process_label": "portable-command"})]
    assert await tool.ainvoke({"command": "blocked"}) == {"error": "done", "tool_budget": {"scope": "operation"}}
    assert await tool.ainvoke({"command": "hello"}) == {
        "result": CommandResult(exit_code=0, stdout="hello", stderr="").model_dump(),
        "tool_budget": {"scope": "available"},
    }
    assert await tool.ainvoke({"command": "fail"}) == {
        "error": "Bash failed with SandboxError: unavailable",
        "tool_budget": {"scope": "available"},
    }
    assert backend.commands == ["hello", "fail"]


async def test_generic_tool_preserves_cancellation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    profile = SandboxProfile((SandboxRoot(tmp_path),))
    monkeypatch.setattr(sandbox_api, "create_sandbox", lambda *_args, **_kwargs: _Sandbox(profile))
    tool = sandbox_tools(profile, budget=_Budget((True, None)))[0]

    with pytest.raises(asyncio.CancelledError):
        await tool.ainvoke({"command": "cancel"})


async def test_legacy_seatbelt_wrapper_keeps_explicit_mac_binding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from localmcp.sandbox import seatbelt

    monkeypatch.setattr(seatbelt, "MacOSSandbox", _Sandbox)
    monkeypatch.setattr(
        sandbox_api, "create_sandbox", lambda *_args, **_kwargs: pytest.fail("legacy wrapper must retain Mac binding")
    )
    tool = seatbelt.sandbox_tools(
        SandboxProfile((SandboxRoot(tmp_path),)), budget=_Budget((True, None)), description="legacy description"
    )[0]

    assert tool.description == "legacy description"
    assert (await tool.ainvoke({"command": "hello"}))["result"]["stdout"] == "hello"
    assert seatbelt.BashInput is sandbox_api.BashInput


def test_shared_environment_does_not_inherit_or_override_authority(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SSH_AUTH_SOCK", "/caller/agent")
    monkeypatch.setenv("PATH", "/caller/bin")

    environment = command_environment(
        tmp_path,
        bin_directories=(Path("/trusted/bin"),),
        extra={"HOME": "/caller/home", "PATH": "/caller/bin", "DEVELOPER_DIR": "/trusted/developer"},
    )

    assert "SSH_AUTH_SOCK" not in environment
    assert environment["HOME"] == "/var/empty"
    assert environment["PATH"] == "/trusted/bin:/usr/bin:/bin:/usr/sbin:/sbin"
    assert environment["TMPDIR"] == str(tmp_path)
    assert environment["DEVELOPER_DIR"] == "/trusted/developer"
