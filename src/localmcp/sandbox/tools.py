"""Platform selection and budgeted model tools for command sandboxes."""

from __future__ import annotations

import sys
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, cast

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from localmcp.sandbox.base import (
    DEFAULT_COMMAND_TIMEOUT_SECONDS,
    DEFAULT_MAX_OUTPUT_BYTES,
    INSPECTION_TOOLS,
    MAX_COMMAND_CHARACTERS,
    Sandbox,
    SandboxError,
    SandboxProfile,
)

DEFAULT_EXHAUSTED_MESSAGE = "Source-tool budget exhausted. Produce the final structured response now."


def _tool_description(tools: tuple[str, ...]) -> str:
    commands = "Only supported shell builtins can be named as commands."
    if tools:
        commands = f"Use supported shell builtins or these literal command names: {', '.join(tools)}."
    return (
        f"Run a command in the policy-bound filesystem roots. {commands} "
        "Use bare command names; paths, dynamic command names, and unsupported Bash syntax are rejected. "
        "No credentials or caller environment are inherited."
    )


DEFAULT_TOOL_DESCRIPTION = _tool_description(INSPECTION_TOOLS)


class BashInput(BaseModel):
    command: str = Field(min_length=1, max_length=MAX_COMMAND_CHARACTERS)


class ToolCallBudget(Protocol):
    async def claim(self) -> tuple[bool, str | None]: ...

    def metadata(self, exhausted_scope: str | None = None) -> dict[str, object]: ...


def create_sandbox(
    profile: SandboxProfile,
    *,
    timeout_seconds: float = DEFAULT_COMMAND_TIMEOUT_SECONDS,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    process_label: str = "localmcp-workspace-sandbox",
) -> Sandbox:
    """Construct the current platform's sandbox, without an unsandboxed fallback."""
    if sys.platform == "darwin":
        from localmcp.sandbox.seatbelt import MacOSSandbox

        return MacOSSandbox(
            profile,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
            process_label=process_label,
        )
    if sys.platform == "linux":
        from localmcp.sandbox.bubblewrap import LinuxSandbox

        return LinuxSandbox(
            profile,
            timeout_seconds=timeout_seconds,
            max_output_bytes=max_output_bytes,
            process_label=process_label,
        )
    raise SandboxError(f"sandbox execution is unsupported on platform: {sys.platform}")


class _GrantedSandbox(Sandbox, Protocol):
    tools: tuple[str, ...]


def sandbox_tools(
    profile: SandboxProfile,
    *,
    budget: ToolCallBudget,
    timeout_seconds: float = DEFAULT_COMMAND_TIMEOUT_SECONDS,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    exhausted_message: str = DEFAULT_EXHAUSTED_MESSAGE,
    description: str | None = None,
    process_label: str = "localmcp-workspace-sandbox",
) -> list[StructuredTool]:
    """Expose Bash under the current platform's immutable sandbox authority."""
    sandbox = create_sandbox(
        profile,
        timeout_seconds=timeout_seconds,
        max_output_bytes=max_output_bytes,
        process_label=process_label,
    )
    return _sandbox_tools(
        sandbox,
        granted_tools=cast(_GrantedSandbox, sandbox).tools,
        budget=budget,
        exhausted_message=exhausted_message,
        description=description,
    )


def _sandbox_tools(
    sandbox: Sandbox,
    *,
    granted_tools: tuple[str, ...],
    budget: ToolCallBudget,
    exhausted_message: str = DEFAULT_EXHAUSTED_MESSAGE,
    description: str | None = None,
) -> list[StructuredTool]:
    """Wrap an already constructed backend without changing its platform binding."""
    if description is None:
        description = _tool_description(granted_tools)

    async def invoke(name: str, operation: Callable[[], Awaitable[BaseModel]]) -> dict[str, Any]:
        allowed, exhausted_scope = await budget.claim()
        if not allowed:
            return {"error": exhausted_message, "tool_budget": budget.metadata(exhausted_scope)}
        try:
            result = await operation()
            return {"result": result.model_dump(), "tool_budget": budget.metadata()}
        except Exception as exc:
            return {"error": f"{name} failed with {type(exc).__name__}: {exc}", "tool_budget": budget.metadata()}

    async def bash(command: str) -> dict[str, Any]:
        return await invoke("Bash", lambda: sandbox.run(command))

    return [
        StructuredTool.from_function(
            coroutine=bash,
            name="Bash",
            description=description,
            args_schema=BashInput,
        )
    ]
