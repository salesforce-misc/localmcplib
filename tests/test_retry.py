from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import anthropic
import httpx2 as httpx
import openai
import pytest
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel, PrivateAttr

from localmcp.retry import (
    _MAX_RESET_DELAY_SECONDS,
    ModelCallRetryMiddleware,
    _reset_after_seconds,
    call_with_rate_limit_retry,
    is_rate_limit_error,
    is_retryable_error,
    rate_limit_delay,
)
from localmcp.structured_output import SubmitResultError, submit_structured


def _response(status: int, headers: dict[str, str] | None = None) -> httpx.Response:
    request = httpx.Request("POST", "https://gateway.example/v1/messages")
    return httpx.Response(status, headers=headers or {}, request=request)


def _gateway_rate_limit(reset_at: str) -> openai.RateLimitError:
    """A 429 shaped like LiteLLM's: reset instant in the message, no Retry-After."""
    message = (
        "Error code: 429 - {'error': {'message': 'litellm.RateLimitError: Rate limit exceeded. "
        f"Limit type: requests. Current limit: 50, Remaining: 0. Limit resets at: {reset_at} UTC'}}}}"
    )
    return openai.RateLimitError(message, response=_response(429), body=None)


class StatusError(Exception):
    """A stand-in exposing the ``status_code`` that SDK errors carry."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


def test_rate_limits_are_detected_from_sdk_errors_and_status_codes() -> None:
    assert is_rate_limit_error(openai.RateLimitError("rl", response=_response(429), body=None))
    assert is_rate_limit_error(anthropic.RateLimitError("rl", response=_response(429), body=None))
    assert is_rate_limit_error(StatusError(429))
    assert not is_rate_limit_error(StatusError(400))
    assert not is_rate_limit_error(ValueError("nope"))


def test_only_transient_faults_are_retryable() -> None:
    request = httpx.Request("POST", "https://gateway.example/v1/messages")
    assert is_retryable_error(StatusError(429))
    assert is_retryable_error(StatusError(503))
    assert is_retryable_error(openai.APIConnectionError(message="boom", request=request))
    assert is_retryable_error(openai.APITimeoutError(request=request))
    assert is_retryable_error(anthropic.APITimeoutError(request=request))
    assert not is_retryable_error(StatusError(400))
    assert not is_retryable_error(ValueError("validation"))


class _CutOffStream(httpx.AsyncByteStream):
    """A response body that drops the connection after its first event."""

    def __init__(self, first: bytes) -> None:
        self._first = first

    async def __aiter__(self) -> Any:
        yield self._first
        raise httpx.RemoteProtocolError("peer closed connection without sending complete message body")


def _event_stream(body: bytes | httpx.AsyncByteStream) -> httpx.AsyncClient:
    def respond(request: httpx.Request) -> httpx.Response:
        headers = {"content-type": "text/event-stream"}
        if isinstance(body, bytes):
            return httpx.Response(200, headers=headers, content=body)
        return httpx.Response(200, headers=headers, stream=body)

    return httpx.AsyncClient(transport=httpx.MockTransport(respond))


async def _anthropic_stream_failure(body: bytes | httpx.AsyncByteStream) -> BaseException:
    client = anthropic.AsyncAnthropic(api_key="token", http_client=_event_stream(body), max_retries=0)
    with pytest.raises(Exception) as caught:
        stream = await client.messages.create(
            model="claude-example",
            max_tokens=16,
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        async for _ in stream:
            pass
    return caught.value


async def _openai_stream_failure(body: bytes) -> BaseException:
    client = openai.AsyncOpenAI(api_key="token", http_client=_event_stream(body), max_retries=0)
    with pytest.raises(Exception) as caught:
        stream = await client.chat.completions.create(
            model="gpt-example",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        async for _ in stream:
            pass
    return caught.value


def _anthropic_error_event(error_type: str) -> bytes:
    return (
        f'event: error\ndata: {{"type": "error", "error": {{"type": "{error_type}", "message": "failed"}}}}\n\n'
    ).encode()


def _openai_error_event(error: str) -> bytes:
    return f'data: {{"error": {{"message": "failed", {error}}}}}\n\n'.encode()


@pytest.mark.parametrize("error_type", ["overloaded_error", "api_error", "rate_limit_error"])
async def test_a_transient_anthropic_error_inside_a_stream_is_retryable(error_type: str) -> None:
    exc = await _anthropic_stream_failure(_anthropic_error_event(error_type))

    assert getattr(exc, "status_code", None) == 200
    assert is_retryable_error(exc)
    assert is_rate_limit_error(exc) == (error_type == "rate_limit_error")


@pytest.mark.parametrize(
    "error",
    ['"type": "server_error", "code": null', '"type": "None", "code": "500"', '"code": "rate_limit_exceeded"'],
)
async def test_a_transient_openai_dialect_error_inside_a_stream_is_retryable(error: str) -> None:
    exc = await _openai_stream_failure(_openai_error_event(error))

    assert isinstance(exc, openai.APIError)
    assert is_retryable_error(exc)
    assert is_rate_limit_error(exc) == ("rate_limit" in error)


async def test_a_deterministic_error_inside_a_stream_is_not_retryable() -> None:
    anthropic_exc = await _anthropic_stream_failure(_anthropic_error_event("invalid_request_error"))
    openai_exc = await _openai_stream_failure(_openai_error_event('"type": "invalid_request_error", "code": "400"'))

    assert not is_retryable_error(anthropic_exc)
    assert not is_retryable_error(openai_exc)


@pytest.mark.parametrize("code", ["\u00b2", "9" * 5_000, "5000"], ids=["superscript", "oversized", "four-digit"])
async def test_a_malformed_error_code_inside_a_stream_is_not_retryable(code: str) -> None:
    # Neither a non-ASCII digit nor a code too long for int() is an HTTP status.
    exc = await _openai_stream_failure(_openai_error_event(f'"type": "None", "code": "{code}"'))

    assert not is_retryable_error(exc)


async def test_a_stream_cut_off_mid_response_is_retryable() -> None:
    start = b'event: ping\ndata: {"type": "ping"}\n\n'
    exc = await _anthropic_stream_failure(_CutOffStream(start))

    assert isinstance(exc, httpx.RemoteProtocolError)
    assert is_retryable_error(exc)
    assert not is_rate_limit_error(exc)


def test_a_status_error_keeps_its_http_classification() -> None:
    body = {"type": "error", "error": {"type": "api_error", "message": "bad"}}
    rejected = anthropic.BadRequestError("bad", response=_response(400), body=body)

    assert not is_retryable_error(rejected)


def test_a_retry_after_header_sets_the_delay_up_to_the_maximum() -> None:
    short = openai.RateLimitError("rl", response=_response(429, {"retry-after": "7"}), body=None)
    long = anthropic.RateLimitError("rl", response=_response(429, {"retry-after": "900"}), body=None)

    assert rate_limit_delay(short, 1, base_delay=2.0, max_delay=60.0) == 7.0
    assert rate_limit_delay(long, 1, base_delay=2.0, max_delay=60.0) == 60.0


def test_other_errors_use_jittered_exponential_backoff() -> None:
    first = rate_limit_delay(StatusError(503), 1, base_delay=2.0, max_delay=60.0)
    second = rate_limit_delay(StatusError(503), 2, base_delay=2.0, max_delay=60.0)

    assert 2.0 <= first <= 2.5
    assert 4.0 <= second <= 5.0


def test_a_retry_after_header_on_a_server_error_is_ignored() -> None:
    error = StatusError(503)
    error.response = _response(503, {"retry-after": "900"})  # type: ignore[attr-defined]

    assert rate_limit_delay(error, 1, base_delay=2.0, max_delay=60.0) <= 2.5


def test_an_embedded_reset_instant_is_parsed_case_insensitively_and_bounded() -> None:
    now = datetime(2026, 9, 3, 1, 11, 16, tzinfo=UTC)
    shouted = StatusError(429)
    shouted.args = ("RATE LIMIT EXCEEDED. LIMIT RESETS AT: 2026-09-03 01:11:46 UTC",)

    assert _reset_after_seconds(_gateway_rate_limit("2026-09-03 01:11:46"), now=now) == 30.0
    assert _reset_after_seconds(shouted, now=now) == 30.0
    assert _reset_after_seconds(_gateway_rate_limit("2026-09-03 01:00:00"), now=now) == 0.0
    assert _reset_after_seconds(StatusError(429)) is None
    # The message is untrusted, so a far-future instant cannot become a multi-year wait.
    assert _reset_after_seconds(_gateway_rate_limit("2099-01-01 00:00:00"), now=now) == _MAX_RESET_DELAY_SECONDS


def test_a_rate_limit_without_retry_after_waits_for_the_reset_window() -> None:
    elapsed = rate_limit_delay(_gateway_rate_limit("2000-01-01 00:00:00"), 1, base_delay=2.0, max_delay=90.0)
    far = _gateway_rate_limit("2099-01-01 00:00:00")

    # Only the jittered buffer remains once the window has reopened.
    assert 1.0 <= elapsed < 2.0
    assert rate_limit_delay(far, 1, base_delay=2.0, max_delay=90.0) == 90.0
    generous = rate_limit_delay(far, 1, base_delay=2.0, max_delay=100_000.0)
    assert _MAX_RESET_DELAY_SECONDS < generous <= _MAX_RESET_DELAY_SECONDS + 2.0


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr("localmcp.retry.asyncio.sleep", fake_sleep)
    monkeypatch.setattr("localmcp.retry.time.sleep", delays.append)
    return delays


async def test_a_transient_failure_is_retried_until_the_operation_succeeds(sleeps: list[float]) -> None:
    attempts = 0
    retries: list[int] = []

    async def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise StatusError(429)
        return "ok"

    result = await call_with_rate_limit_retry(
        operation,
        max_attempts=3,
        base_delay=2.0,
        max_delay=60.0,
        on_retry=lambda attempt, delay, exc: retries.append(attempt),
    )

    assert result == "ok"
    assert attempts == 3
    assert len(sleeps) == 2
    assert retries == [1, 2]


async def test_a_deterministic_failure_is_raised_without_retrying(sleeps: list[float]) -> None:
    attempts = 0

    async def operation() -> str:
        nonlocal attempts
        attempts += 1
        raise ValueError("deterministic")

    with pytest.raises(ValueError, match="deterministic"):
        await call_with_rate_limit_retry(operation, max_attempts=3, base_delay=2.0, max_delay=60.0)
    assert attempts == 1
    assert sleeps == []


async def test_the_last_transient_failure_is_raised_once_attempts_run_out(sleeps: list[float]) -> None:
    attempts = 0

    async def operation() -> str:
        nonlocal attempts
        attempts += 1
        raise StatusError(503)

    with pytest.raises(StatusError):
        await call_with_rate_limit_retry(operation, max_attempts=3, base_delay=2.0, max_delay=60.0)
    assert attempts == 3
    assert len(sleeps) == 2


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_attempts": 0}, "at least 1"),
        ({"base_delay": -1.0}, "must not be negative"),
        ({"base_delay": 10.0, "max_delay": 5.0}, "must not exceed"),
    ],
)
def test_an_invalid_retry_policy_is_rejected(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ModelCallRetryMiddleware(**kwargs)


def _submission(args: dict[str, Any]) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": "submit_result", "args": args, "id": "s1", "type": "tool_call"}])


class _ScriptedChatModel(BaseChatModel):
    """Replays scripted turns, raising scripted exceptions, and refuses forced tool use."""

    responses: list[Any]
    _requests: list[list[BaseMessage]] = PrivateAttr(default_factory=list)
    _bound_tools: list[str] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    @property
    def requests(self) -> list[list[BaseMessage]]:
        return self._requests

    @property
    def bound_tools(self) -> list[str]:
        return self._bound_tools

    def bind_tools(self, tools: Sequence[Any], *, tool_choice: Any = None, **kwargs: Any) -> _ScriptedChatModel:
        if tool_choice is not None:
            raise AssertionError(f"forced tool_choice is not supported: {tool_choice!r}")
        self._bound_tools[:] = [getattr(tool, "name", None) or tool["name"] for tool in tools]
        return self

    def _generate(self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any) -> Any:
        self._requests.append(list(messages))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        # Copy so a reused scripted turn never shares a message ID with an earlier one.
        return ChatResult(generations=[ChatGeneration(message=response.model_copy())])


class _Answer(BaseModel):
    city: str


async def _submit(model: _ScriptedChatModel, **kwargs: Any) -> _Answer:
    return await submit_structured(
        model, _Answer, system_prompt="Answer.", messages=[HumanMessage(content="Capital of France?")], **kwargs
    )


async def test_submit_structured_offers_only_an_unforced_submission_tool(sleeps: list[float]) -> None:
    model = _ScriptedChatModel(responses=[_submission({"city": "Paris"})])

    assert await _submit(model) == _Answer(city="Paris")
    assert model.bound_tools == ["submit_result"]
    assert "Answer." in str(model.requests[0][0].content)


async def test_submit_structured_corrects_an_invalid_submission_in_the_same_run(sleeps: list[float]) -> None:
    model = _ScriptedChatModel(responses=[_submission({"town": "Paris"}), _submission({"city": "Paris"})])

    assert await _submit(model) == _Answer(city="Paris")
    assert len(model.requests) == 2


async def test_submit_structured_raises_when_the_model_never_submits(sleeps: list[float]) -> None:
    model = _ScriptedChatModel(responses=[AIMessage(content="Paris")] * 2)

    with pytest.raises(SubmitResultError):
        await _submit(model, max_attempts=2)


async def test_a_transient_model_error_retries_only_the_failed_call(sleeps: list[float]) -> None:
    model = _ScriptedChatModel(responses=[StatusError(503), _submission({"city": "Paris"})])

    assert await _submit(model) == _Answer(city="Paris")
    # The retried request is identical: no progress is lost or replayed.
    assert len(model.requests) == 2
    assert model.requests[0] == model.requests[1]
    assert len(sleeps) == 1


async def test_a_rate_limited_model_call_waits_for_retry_after(sleeps: list[float]) -> None:
    limited = openai.RateLimitError("limited", response=_response(429, {"retry-after": "7"}), body=None)
    model = _ScriptedChatModel(responses=[limited, _submission({"city": "Paris"})])

    assert await _submit(model) == _Answer(city="Paris")
    assert sleeps == [7.0]


async def test_a_deterministic_model_error_is_not_retried(sleeps: list[float]) -> None:
    model = _ScriptedChatModel(responses=[StatusError(400), _submission({"city": "Paris"})])

    with pytest.raises(StatusError):
        await _submit(model)
    assert len(model.requests) == 1
    assert sleeps == []


async def test_a_custom_retry_policy_and_callback_are_used(sleeps: list[float]) -> None:
    retried: list[int] = []
    retry = ModelCallRetryMiddleware(max_attempts=2, on_retry=lambda attempt, delay, exc: retried.append(attempt))
    model = _ScriptedChatModel(responses=[StatusError(503)] * 2)

    with pytest.raises(StatusError):
        await _submit(model, retry=retry)
    assert len(model.requests) == 2
    assert retried == [1]


def test_synchronous_agents_retry_a_transient_model_error(sleeps: list[float]) -> None:
    model = _ScriptedChatModel(responses=[StatusError(503), AIMessage(content="Paris")])
    agent = create_agent(model, [], middleware=[ModelCallRetryMiddleware()])

    result = agent.invoke({"messages": [HumanMessage(content="Capital of France?")]})

    assert result["messages"][-1].content == "Paris"
    assert len(sleeps) == 1
