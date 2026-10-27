"""LangChain agents whose model calls retry transient errors."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain.agents import create_agent as _create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from localmcp.retry import ModelCallRetryMiddleware


def create_agent(
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable[..., Any] | dict[str, Any]] | None = None,
    *,
    middleware: Sequence[AgentMiddleware[Any, Any, Any]] = (),
    retry: ModelCallRetryMiddleware | None = None,
    **kwargs: Any,
) -> CompiledStateGraph[Any, Any, Any, Any]:
    """Return LangChain's ``create_agent`` with each model call retried on transient errors.

    Every response is streamed and bounded by a deadline, so a stalled or
    dropped stream fails one model call. ``retry``, a default
    :class:`~localmcp.retry.ModelCallRetryMiddleware` when omitted, is placed
    after ``middleware`` so it wraps the model call directly: only the failed
    request is repeated, and completed turns and tool calls are kept. Pass
    ``ModelCallRetryMiddleware(max_attempts=1)`` to disable retries. Other
    arguments are passed to LangChain's ``create_agent`` unchanged.
    """
    return _create_agent(model, tools, middleware=[*middleware, retry or ModelCallRetryMiddleware()], **kwargs)
