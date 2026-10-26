import asyncio
import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx2
import pytest
from langchain_core.messages import AIMessage

from localmcp.llm import (
    DEFAULT_MODEL_REGISTRY,
    ConfiguredModelFactory,
    GatewayEndpoint,
    GatewayModelFactory,
    LLMBackendConfig,
    ModelConfigurationError,
    ModelFactory,
    ModelRegistry,
    ModelSpec,
    NativeModelFactory,
    NativeTransport,
    OpenAICompatibleEndpoint,
    OpenAICompatibleModelFactory,
    OwnedChatModel,
    _response_deadline_client_type,
    infer_model_spec,
    response_deadline_http_client,
)
from localmcp.retry import is_retryable_error


class FakeClient:
    def __init__(self) -> None:
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


class FakeChatOpenAI:
    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs


class AlternateModelFactory:
    def __init__(self, owner: FakeClient) -> None:
        self.owner = owner
        self.created_with: tuple[str, int | None, str | None] | None = None

    def validate(self, model_id: str, *, reasoning_effort: str | None = None) -> ModelSpec:
        del model_id, reasoning_effort
        return ModelSpec()

    def create(
        self,
        model_id: str,
        *,
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> OwnedChatModel:
        self.created_with = (model_id, max_output_tokens, reasoning_effort)
        return OwnedChatModel(object(), self.owner)


def create_with(factory: ModelFactory) -> OwnedChatModel:
    return factory.create("native-model", max_output_tokens=321, reasoning_effort="high")


def test_default_catalog_separates_capabilities_from_invocation_policy() -> None:
    gpt = DEFAULT_MODEL_REGISTRY.resolve("gpt-5.6")
    claude = DEFAULT_MODEL_REGISTRY.resolve("claude-opus-4-8")

    assert gpt.supports_reasoning_effort
    assert gpt.default_reasoning_effort is None
    assert claude.supports_reasoning_effort
    assert claude.default_reasoning_effort is None
    assert claude.native_provider == "anthropic"
    assert claude.temperature == "omit"
    assert not DEFAULT_MODEL_REGISTRY.resolve("claude-haiku-4-5-20251001").supports_reasoning_effort


async def test_model_factory_protocol_accepts_alternate_backends() -> None:
    client = FakeClient()
    factory = AlternateModelFactory(client)

    async with create_with(factory):
        pass

    assert factory.created_with == ("native-model", 321, "high")
    assert client.closed


async def test_factory_routes_every_model_through_openai_compatible_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    import langchain_openai

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeChatOpenAI)
    client = FakeClient()
    registry = ModelRegistry(
        {
            "claude-example": ModelSpec(temperature="omit"),
            "gpt-example": ModelSpec(
                temperature="omit",
                supports_reasoning_effort=True,
                default_reasoning_effort="medium",
            ),
        }
    )
    factory = OpenAICompatibleModelFactory(
        registry,
        OpenAICompatibleEndpoint(
            "https://gateway.example/v1",
            default_headers={"x-service-name": "test"},
            keepalive_expiry=1.0,
        ),
        "token",
        http_client_factory=lambda _: client,
    )

    owner = factory.create("claude-example", max_output_tokens=123)
    async with owner as model:
        assert isinstance(model, FakeChatOpenAI)
        assert model.kwargs["model"] == "claude-example"
        assert model.kwargs["max_completion_tokens"] == 123
        assert model.kwargs["default_headers"] == {"x-service-name": "test"}
        assert "temperature" not in model.kwargs
    assert client.closed


async def test_factory_owns_default_httpx2_transport() -> None:
    import httpx2
    from langchain_openai import ChatOpenAI

    factory = OpenAICompatibleModelFactory(
        ModelRegistry({"model": ModelSpec()}),
        OpenAICompatibleEndpoint("https://gateway.invalid/v1"),
        "token",
    )

    owner = factory.create("model")
    client = owner._owner
    assert isinstance(client, httpx2.AsyncClient)
    async with owner as model:
        assert isinstance(model, ChatOpenAI)
    assert client.is_closed


def test_registry_overlay_and_reasoning_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    import langchain_openai

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeChatOpenAI)
    registry = ModelRegistry({"base": ModelSpec()}).overlay({"other": ModelSpec()})
    factory = OpenAICompatibleModelFactory(
        registry,
        OpenAICompatibleEndpoint("https://gateway.example/v1"),
        "token",
        http_client_factory=lambda _: FakeClient(),
    )

    assert registry.resolve("base") == ModelSpec()
    with pytest.raises(ModelConfigurationError, match="does not support"):
        factory.create("other", reasoning_effort="high")


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        ("claude-opus-9", ModelSpec(temperature="omit", supports_reasoning_effort=True, native_provider="anthropic")),
        ("gpt-7", ModelSpec(temperature="omit", supports_reasoning_effort=True, native_provider="openai")),
        ("o5-mini", ModelSpec(temperature="omit", supports_reasoning_effort=True, native_provider="openai")),
        ("gemini-9-pro", ModelSpec(temperature="omit", supports_reasoning_effort=True)),
    ],
)
def test_unregistered_models_resolve_to_inferred_specs(model_id: str, expected: ModelSpec) -> None:
    assert ModelRegistry().resolve(model_id) == expected
    assert infer_model_spec(model_id) == expected


def test_unregistered_model_warns_once_and_blank_ids_are_rejected(caplog: pytest.LogCaptureFixture) -> None:
    registry = ModelRegistry()

    with caplog.at_level(logging.WARNING, logger="localmcp.llm"):
        registry.resolve("warn-once-model")
        registry.resolve("warn-once-model")

    assert [record.getMessage() for record in caplog.records] == [
        'model "warn-once-model" is not in the catalog; using inferred capabilities'
    ]
    with pytest.raises(ModelConfigurationError, match="must not be blank"):
        registry.resolve(" ")


async def test_unregistered_model_passes_through_to_the_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    import langchain_openai

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeChatOpenAI)
    factory = OpenAICompatibleModelFactory(
        ModelRegistry(),
        OpenAICompatibleEndpoint("https://gateway.example/v1"),
        "token",
        http_client_factory=lambda _: FakeClient(),
    )

    async with factory.create("gpt-7", reasoning_effort="high") as model:
        assert model.kwargs["model"] == "gpt-7"
        assert model.kwargs["reasoning_effort"] == "high"
        assert "temperature" not in model.kwargs


def test_unregistered_models_route_by_inferred_provider() -> None:
    gateway = ConfiguredModelFactory(
        ModelRegistry(),
        LLMBackendConfig(backend="openai_compatible", base_url="https://gateway.example/v1"),
        gateway_api_key="gateway-token",
    )
    native = ConfiguredModelFactory(
        ModelRegistry(),
        LLMBackendConfig(backend="native"),
        anthropic_api_key="anthropic-token",
    )

    assert gateway.registry.resolve("claude-opus-9") == gateway.validate("claude-opus-9")
    assert gateway.validate("claude-opus-9").adapter_dialect == "anthropic"
    assert native.validate("claude-opus-9").native_provider == "anthropic"
    assert gateway.validate("claude-opus-9", reasoning_effort="high").supports_reasoning_effort
    with pytest.raises(ModelConfigurationError, match='"gemini-9-pro" is available only through a gateway'):
        native.validate("gemini-9-pro")


def test_backend_selection_is_explicit_and_validates_gateway_state() -> None:
    with pytest.raises(ModelConfigurationError, match="requires base_url"):
        LLMBackendConfig(backend="openai_compatible")
    with pytest.raises(ModelConfigurationError, match="does not accept"):
        LLMBackendConfig(backend="native", base_url="https://stale-gateway.example/v1")


async def test_configured_factory_switches_one_model_between_gateway_and_native(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx2
    from langchain_anthropic import ChatAnthropic
    from langchain_openai import ChatOpenAI

    # The Anthropic client reads ANTHROPIC_BASE_URL and friends from the caller's shell.
    for name in [name for name in os.environ if name.startswith("ANTHROPIC_")]:
        monkeypatch.delenv(name)

    registry = ModelRegistry(
        {
            # The gateway dialect is deliberately independent from the native
            # provider: stock LiteLLM can expose Claude through its OpenAI API.
            "claude-over-openai": ModelSpec(
                temperature="omit",
                native_provider="anthropic",
                dialect="openai",
            ),
            "claude-over-anthropic": ModelSpec(native_provider="anthropic"),
        }
    )
    gateway = ConfiguredModelFactory(
        registry,
        LLMBackendConfig(backend="openai_compatible", base_url="https://litellm.example/v1"),
        gateway_api_key="gateway-token",
    )

    openai_gateway_owner = gateway.create("claude-over-openai")
    assert isinstance(openai_gateway_owner.model, ChatOpenAI)
    assert str(openai_gateway_owner.model.openai_api_base) == "https://litellm.example/v1"
    assert isinstance(openai_gateway_owner._owner, httpx2.AsyncClient)
    async with openai_gateway_owner:
        pass
    assert openai_gateway_owner._owner.is_closed

    anthropic_gateway_owner = gateway.create("claude-over-anthropic")
    assert isinstance(anthropic_gateway_owner.model, ChatAnthropic)
    assert str(anthropic_gateway_owner.model.anthropic_api_url) == "https://litellm.example/v1"
    async with anthropic_gateway_owner:
        pass
    assert anthropic_gateway_owner._owner.is_closed

    native = ConfiguredModelFactory(
        registry,
        LLMBackendConfig(backend="native"),
        anthropic_api_key="anthropic-token",
    )
    native_owner = native.create("claude-over-openai")
    assert isinstance(native_owner.model, ChatAnthropic)
    assert str(native_owner.model.anthropic_api_url) == "https://api.anthropic.com"
    async with native_owner:
        pass
    assert native_owner._owner.is_closed


async def test_native_factory_requires_only_the_selected_provider_key() -> None:
    registry = ModelRegistry(
        {
            "claude": ModelSpec(native_provider="anthropic"),
            "gpt": ModelSpec(native_provider="openai"),
        }
    )
    factory = NativeModelFactory(registry, anthropic_api_key="anthropic-token")

    owner = factory.create("claude")
    with pytest.raises(ModelConfigurationError, match='native openai API key.*"gpt"'):
        factory.create("gpt")

    # Construction is lazy: selecting Claude did not require an OpenAI key.
    assert owner.model is not None
    await owner.__aexit__()


def test_gateway_only_model_fails_before_transport_construction() -> None:
    constructed = False

    def client_factory(*_: object) -> FakeClient:
        nonlocal constructed
        constructed = True
        return FakeClient()

    factory = NativeModelFactory(
        ModelRegistry({"gemini": ModelSpec()}),
        http_client_factory=client_factory,
    )

    with pytest.raises(ModelConfigurationError, match='"gemini" is available only through a gateway'):
        factory.validate("gemini")
    assert not constructed


def test_configured_factory_validates_backend_compatibility_without_constructing_transport() -> None:
    registry = ModelRegistry({"gemini": ModelSpec()})
    gateway = ConfiguredModelFactory(
        registry,
        LLMBackendConfig(backend="openai_compatible", base_url="https://gateway.example/v1"),
        gateway_api_key="gateway-token",
    )
    native = ConfiguredModelFactory(registry, LLMBackendConfig(backend="native"))

    assert gateway.validate("gemini") == ModelSpec()
    with pytest.raises(ModelConfigurationError, match='"gemini" is available only through a gateway'):
        native.validate("gemini")


def test_model_specs_reject_inconsistent_capabilities() -> None:
    with pytest.raises(ModelConfigurationError, match="requires supports_reasoning_effort"):
        ModelSpec(default_reasoning_effort="medium")
    with pytest.raises(ModelConfigurationError, match="max_output_tokens must be a positive integer"):
        ModelSpec(max_output_tokens=0)
    with pytest.raises(ModelConfigurationError, match="max_output_tokens must be a positive integer"):
        ModelSpec(max_output_tokens=True)

    assert ModelSpec(native_provider="anthropic").adapter_dialect == "anthropic"
    assert ModelSpec().adapter_dialect == "openai"


def test_registry_rejects_blank_ids_and_returns_a_defensive_mapping() -> None:
    with pytest.raises(ModelConfigurationError, match="must not be blank"):
        ModelRegistry({" ": ModelSpec()})

    registry = ModelRegistry({"model": ModelSpec()})
    copied = registry.as_mapping()
    assert copied == {"model": ModelSpec()}
    copied.clear()
    assert registry.resolve("model") == ModelSpec()


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: GatewayEndpoint(" "),
        lambda: GatewayEndpoint("https://example.test", keepalive_expiry=-1),
        lambda: NativeTransport(keepalive_expiry=-1),
        lambda: GatewayEndpoint("https://example.test", response_timeout=0),
        lambda: GatewayEndpoint("https://example.test", response_timeout=float("inf")),
        lambda: GatewayEndpoint("https://example.test", response_timeout=True),
        lambda: NativeTransport(response_timeout=-1),
        lambda: NativeTransport(response_timeout=float("nan")),
        lambda: LLMBackendConfig(backend="native", response_timeout=0),
        lambda: response_deadline_http_client(response_timeout=float("nan")),
        lambda: response_deadline_http_client(response_timeout=True),
        lambda: LLMBackendConfig(backend="unsupported"),  # type: ignore[arg-type]
        lambda: LLMBackendConfig(backend="openai_compatible", base_url="https://example.test", keepalive_expiry=-1),
    ],
)
def test_transport_configuration_rejects_invalid_values(constructor: Any) -> None:
    with pytest.raises(ModelConfigurationError):
        constructor()


def test_backend_config_exposes_only_the_selected_transport() -> None:
    gateway = LLMBackendConfig(
        backend="openai_compatible",
        base_url="https://gateway.example/v1",
        default_headers={"x-test": "value"},
        verify=False,
        keepalive_expiry=2,
        response_timeout=30,
        stream_usage=False,
    )
    assert gateway.gateway_endpoint() == GatewayEndpoint(
        "https://gateway.example/v1",
        default_headers={"x-test": "value"},
        verify=False,
        keepalive_expiry=2,
        response_timeout=30,
        stream_usage=False,
    )
    with pytest.raises(ModelConfigurationError, match="non-native"):
        gateway.native_transport()

    native = LLMBackendConfig(backend="native", verify=False, keepalive_expiry=3, response_timeout=40)
    assert native.native_transport() == NativeTransport(verify=False, keepalive_expiry=3, response_timeout=40)
    with pytest.raises(ModelConfigurationError, match="non-openai_compatible"):
        native.gateway_endpoint()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: OpenAICompatibleModelFactory(
            ModelRegistry(), OpenAICompatibleEndpoint("https://example.test"), "token", default_max_output_tokens=0
        ),
        lambda: NativeModelFactory(ModelRegistry(), default_max_output_tokens=0),
        lambda: NativeModelFactory(ModelRegistry(), default_max_output_tokens=True),
        lambda: ConfiguredModelFactory(
            ModelRegistry(),
            LLMBackendConfig(backend="openai_compatible", base_url="https://example.test"),
            gateway_api_key="token",
            default_max_output_tokens=0,
        ),
    ],
)
def test_factories_reject_non_positive_default_output_limits(factory: Any) -> None:
    with pytest.raises(ModelConfigurationError, match="must be a positive integer"):
        factory()


def test_create_rejects_non_positive_per_call_output_limit_before_transport() -> None:
    constructed = False

    def client_factory(_: GatewayEndpoint) -> FakeClient:
        nonlocal constructed
        constructed = True
        return FakeClient()

    factory = OpenAICompatibleModelFactory(
        ModelRegistry({"model": ModelSpec()}),
        GatewayEndpoint("https://example.test"),
        "token",
        http_client_factory=client_factory,
    )
    for invalid in (0, True):
        with pytest.raises(ModelConfigurationError, match="must be a positive integer"):
            factory.create("model", max_output_tokens=invalid)
    assert not constructed


_ANTHROPIC_STREAM = [
    (
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-example",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 1},
            },
        },
    ),
    ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
    (
        "content_block_delta",
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hel"}},
    ),
    ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "lo"}}),
    ("content_block_stop", {"type": "content_block_stop", "index": 0}),
    (
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": "max_tokens", "stop_sequence": None},
            "usage": {"output_tokens": 5},
        },
    ),
    ("message_stop", {"type": "message_stop"}),
]


def _openai_chunk(delta: dict[str, Any], finish_reason: str | None = None) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-example",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


_OPENAI_STREAM = [
    _openai_chunk({"role": "assistant", "content": "Hel"}),
    _openai_chunk({"content": "lo"}, finish_reason="length"),
    {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-example",
        "choices": [],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    },
]


def _anthropic_tool_use_events() -> list[tuple[str, dict[str, Any]]]:
    events: list[tuple[str, dict[str, Any]]] = [_ANTHROPIC_STREAM[0]]
    for index, (call_id, finding) in enumerate([("toolu_1", "F1"), ("toolu_2", "F2")]):
        events += [
            (
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {"type": "tool_use", "id": call_id, "name": "record", "input": {}},
                },
            ),
            *(
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {"type": "input_json_delta", "partial_json": part},
                    },
                )
                for part in (f'{{"finding_id": "{finding}",', ' "decision": "keep"}')
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": index}),
        ]
    return [
        *events,
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"output_tokens": 30},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]


def _openai_tool_call_chunks() -> list[dict[str, Any]]:
    chunks = [_openai_chunk({"role": "assistant", "content": None})]
    for index, (call_id, finding) in enumerate([("call_1", "F1"), ("call_2", "F2")]):
        start = {"index": index, "id": call_id, "type": "function", "function": {"name": "record", "arguments": ""}}
        chunks.append(_openai_chunk({"tool_calls": [start]}))
        for part in (f'{{"finding_id": "{finding}",', ' "decision": "keep"}'):
            chunks.append(_openai_chunk({"tool_calls": [{"index": index, "function": {"arguments": part}}]}))
    return [*chunks, _openai_chunk({}, finish_reason="tool_calls"), _OPENAI_STREAM[-1]]


_RECORD_TOOL = {
    "type": "function",
    "function": {
        "name": "record",
        "description": "Record one decision.",
        "parameters": {
            "type": "object",
            "properties": {"finding_id": {"type": "string"}, "decision": {"type": "string"}},
            "required": ["finding_id", "decision"],
        },
    },
}


class RecordingGateway:
    """Serve one canned event stream per request and record each request body."""

    def __init__(self, dialect: str, stream: list[Any] | None = None) -> None:
        self.dialect = dialect
        self.stream = stream
        self.requests: list[dict[str, Any]] = []

    def __call__(self, _: GatewayEndpoint) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.MockTransport(self._respond))

    def _respond(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content))
        if self.dialect == "anthropic":
            events = self.stream or _ANTHROPIC_STREAM
            body = "".join(f"event: {event}\ndata: {json.dumps(data)}\n\n" for event, data in events)
        else:
            chunks = self.stream or _OPENAI_STREAM
            body = "".join(f"data: {json.dumps(data)}\n\n" for data in chunks) + "data: [DONE]\n\n"
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())


@pytest.fixture
def _no_provider_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    # The SDK clients read ANTHROPIC_* and OPENAI_* settings from the caller's shell.
    for name in [name for name in os.environ if name.startswith(("ANTHROPIC_", "OPENAI_"))]:
        monkeypatch.delenv(name)


async def _invoke(
    registry: ModelRegistry,
    dialect: str,
    model_id: str,
    *,
    endpoint: GatewayEndpoint | None = None,
    default_max_output_tokens: int | None = None,
    stream: list[Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
    **create: Any,
) -> tuple[dict[str, Any], AIMessage]:
    gateway = RecordingGateway(dialect, stream)
    factory = GatewayModelFactory(
        registry,
        endpoint or GatewayEndpoint("https://gateway.example/v1"),
        "token",
        http_client_factory=gateway,
        default_max_output_tokens=default_max_output_tokens,
    )
    async with factory.create(model_id, **create) as model:
        message = await (model.bind_tools(tools) if tools else model).ainvoke("hi")
    [request] = gateway.requests
    return request, message


@pytest.mark.usefixtures("_no_provider_environment")
async def test_anthropic_models_stream_and_return_one_reassembled_message() -> None:
    registry = ModelRegistry({"claude-opus-5": ModelSpec(supports_reasoning_effort=True, native_provider="anthropic")})

    request, message = await _invoke(registry, "anthropic", "claude-opus-5", reasoning_effort="xhigh")

    assert request["stream"] is True
    assert request["output_config"] == {"effort": "xhigh"}
    # An effort enables thinking, which rejects the catalogued temperature.
    assert request["thinking"]["type"] == "adaptive"
    assert "temperature" not in request
    assert message.text == "Hello"
    assert message.response_metadata["stop_reason"] == "max_tokens"
    assert message.usage_metadata is not None
    assert message.usage_metadata["output_tokens"] == 5


_RECORDED_CALLS = [{"finding_id": "F1", "decision": "keep"}, {"finding_id": "F2", "decision": "keep"}]


@pytest.mark.usefixtures("_no_provider_environment")
@pytest.mark.parametrize(
    ("dialect", "spec", "stream", "call_ids"),
    [
        ("anthropic", ModelSpec(native_provider="anthropic"), _anthropic_tool_use_events(), ["toolu_1", "toolu_2"]),
        ("openai", ModelSpec(native_provider="openai"), _openai_tool_call_chunks(), ["call_1", "call_2"]),
    ],
)
async def test_streamed_tool_calls_reassemble_into_complete_calls(
    dialect: str, spec: ModelSpec, stream: list[Any], call_ids: list[str]
) -> None:
    registry = ModelRegistry({"model": spec})

    request, message = await _invoke(registry, dialect, "model", stream=stream, tools=[_RECORD_TOOL])

    assert request["stream"] is True
    assert [tool.get("name") or tool["function"]["name"] for tool in request["tools"]] == ["record"]
    assert [(call["id"], call["name"], call["args"]) for call in message.tool_calls] == [
        (call_id, "record", args) for call_id, args in zip(call_ids, _RECORDED_CALLS, strict=True)
    ]
    assert not message.invalid_tool_calls


@pytest.mark.usefixtures("_no_provider_environment")
async def test_any_effort_on_an_anthropic_model_omits_the_catalogued_temperature() -> None:
    registry = ModelRegistry(
        {
            "claude-default": ModelSpec(
                supports_reasoning_effort=True, default_reasoning_effort="medium", native_provider="anthropic"
            ),
            "claude-translated": ModelSpec(
                supports_reasoning_effort=True, native_provider="anthropic", dialect="openai"
            ),
        }
    )

    default, _ = await _invoke(registry, "anthropic", "claude-default")
    translated, _ = await _invoke(registry, "openai", "claude-translated", reasoning_effort="high")
    without_effort, _ = await _invoke(registry, "openai", "claude-translated")

    assert default["output_config"] == {"effort": "medium"}
    assert "temperature" not in default
    assert translated["reasoning_effort"] == "high"
    assert "temperature" not in translated
    assert without_effort["temperature"] == 0


@pytest.mark.usefixtures("_no_provider_environment")
async def test_anthropic_models_without_a_limit_use_the_model_maximum() -> None:
    from langchain_anthropic import ChatAnthropic

    registry = ModelRegistry(
        {
            "claude-opus-5": ModelSpec(native_provider="anthropic"),
            "claude-house": ModelSpec(native_provider="anthropic"),
            "claude-small": ModelSpec(native_provider="anthropic", max_output_tokens=8_192),
        }
    )
    profile = ChatAnthropic(model="claude-opus-5", api_key="token").profile or {}

    profiled, _ = await _invoke(registry, "anthropic", "claude-opus-5")
    unprofiled, _ = await _invoke(registry, "anthropic", "claude-house")
    declared, _ = await _invoke(registry, "anthropic", "claude-small")
    requested, _ = await _invoke(registry, "anthropic", "claude-small", max_output_tokens=1_000)

    assert profiled["max_tokens"] == profile["max_output_tokens"] > 64_000
    assert profiled["temperature"] == 0
    assert unprofiled["max_tokens"] == 64_000
    assert declared["max_tokens"] == 8_192
    assert requested["max_tokens"] == 1_000


@pytest.mark.usefixtures("_no_provider_environment")
async def test_openai_dialect_models_stream_usage_and_omit_an_unset_limit() -> None:
    registry = ModelRegistry(
        {
            "gpt-example": ModelSpec(temperature="omit", supports_reasoning_effort=True, native_provider="openai"),
            "gpt-small": ModelSpec(temperature="omit", native_provider="openai", max_output_tokens=4_096),
        }
    )

    request, message = await _invoke(registry, "openai", "gpt-example", reasoning_effort="high")
    declared, _ = await _invoke(registry, "openai", "gpt-small")

    assert request["stream"] is True
    assert request["stream_options"] == {"include_usage": True}
    assert request["reasoning_effort"] == "high"
    assert "max_completion_tokens" not in request
    assert "max_tokens" not in request
    assert declared["max_completion_tokens"] == 4_096
    assert message.text == "Hello"
    assert message.response_metadata["finish_reason"] == "length"
    assert message.usage_metadata is not None
    assert message.usage_metadata["output_tokens"] == 5


@pytest.mark.usefixtures("_no_provider_environment")
async def test_a_gateway_can_stop_requesting_streamed_usage() -> None:
    registry = ModelRegistry({"gpt-example": ModelSpec(native_provider="openai")})
    endpoint = GatewayEndpoint("https://gateway.example/v1", stream_usage=False)

    request, message = await _invoke(registry, "openai", "gpt-example", endpoint=endpoint)

    assert request["stream"] is True
    assert "stream_options" not in request
    assert message.text == "Hello"


@pytest.mark.usefixtures("_no_provider_environment")
async def test_a_factory_default_limit_never_exceeds_a_catalogued_maximum() -> None:
    registry = ModelRegistry(
        {
            "gpt-example": ModelSpec(native_provider="openai"),
            "gpt-small": ModelSpec(native_provider="openai", max_output_tokens=4_096),
            "claude-small": ModelSpec(native_provider="anthropic", max_output_tokens=8_192),
        }
    )

    uncatalogued, _ = await _invoke(registry, "openai", "gpt-example", default_max_output_tokens=10_000)
    capped, _ = await _invoke(registry, "openai", "gpt-small", default_max_output_tokens=10_000)
    lower, _ = await _invoke(registry, "openai", "gpt-small", default_max_output_tokens=2_000)
    requested, _ = await _invoke(
        registry, "openai", "gpt-small", default_max_output_tokens=2_000, max_output_tokens=6_000
    )
    anthropic, _ = await _invoke(registry, "anthropic", "claude-small", default_max_output_tokens=10_000)

    assert uncatalogued["max_completion_tokens"] == 10_000
    assert capped["max_completion_tokens"] == 4_096
    assert lower["max_completion_tokens"] == 2_000
    assert requested["max_completion_tokens"] == 6_000
    assert anthropic["max_tokens"] == 8_192


def test_configured_gateway_requires_its_explicit_credential() -> None:
    with pytest.raises(ModelConfigurationError, match="gateway API key is required"):
        ConfiguredModelFactory(
            ModelRegistry(),
            LLMBackendConfig(backend="openai_compatible", base_url="https://example.test"),
        )


def test_anthropic_async_key_provider_is_rejected_before_transport_use() -> None:
    class AwaitableValue:
        def __await__(self) -> Any:
            yield
            return "token"

    def key() -> Any:
        return AwaitableValue()

    factory = NativeModelFactory(
        ModelRegistry({"claude": ModelSpec(native_provider="anthropic")}),
        anthropic_api_key=key,
        http_client_factory=lambda *_: FakeClient(),
    )
    with pytest.raises(ModelConfigurationError, match="synchronous callables"):
        factory.create("claude")


_PING = b'event: ping\ndata: {"type": "ping"}\n\n'


class _KeepAliveStream(httpx2.AsyncByteStream):
    """A response body that sends ``first`` and then only keep-alive pings, forever."""

    def __init__(self, first: bytes) -> None:
        self._first = first

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._first
        while True:
            await asyncio.sleep(0.01)
            yield _PING


def _deadline_client(handler: Any, response_timeout: float) -> httpx2.AsyncClient:
    # A custom client factory keeps the deadline by building its clients here.
    client = response_deadline_http_client(response_timeout=response_timeout, transport=httpx2.MockTransport(handler))
    assert isinstance(client, httpx2.AsyncClient)
    return client


async def test_factory_clients_bound_every_response() -> None:
    factory = GatewayModelFactory(
        ModelRegistry({"model": ModelSpec()}),
        GatewayEndpoint("https://gateway.invalid/v1", response_timeout=45),
        "token",
    )

    owner = factory.create("model")
    async with owner:
        assert isinstance(owner._owner, _response_deadline_client_type())
        assert owner._owner._response_timeout == 45


async def test_a_response_streaming_only_keep_alives_fails_at_its_deadline() -> None:
    def respond(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=_KeepAliveStream(_PING))

    async with _deadline_client(respond, 0.2) as client:
        async with client.stream("POST", "https://gateway.example/v1/messages") as response:
            with pytest.raises(httpx2.ReadTimeout, match="exceeded the 0.2 s response timeout") as raised:
                async for _ in response.aiter_bytes():
                    pass
    assert is_retryable_error(raised.value)


async def test_a_response_whose_headers_never_arrive_fails_at_its_deadline() -> None:
    async def respond(_: httpx2.Request) -> httpx2.Response:
        await asyncio.sleep(10)
        return httpx2.Response(200)

    async with _deadline_client(respond, 0.1) as client:
        with pytest.raises(httpx2.ReadTimeout, match="response timeout"):
            await client.post("https://gateway.example/v1/messages")


class _ChunkedStream(httpx2.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            await asyncio.sleep(0)
            yield chunk


async def test_a_response_within_its_deadline_is_read_in_full() -> None:
    chunks = [f"chunk {index}\n".encode() for index in range(50)]

    def respond(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, stream=_ChunkedStream(chunks))

    async with _deadline_client(respond, 5) as client:
        async with client.stream("POST", "https://gateway.example/v1/messages") as response:
            assert await response.aread() == b"".join(chunks)
        assert (await client.post("https://gateway.example/v1/messages")).content == b"".join(chunks)


@pytest.mark.usefixtures("_no_provider_environment")
async def test_a_model_stalled_mid_stream_raises_a_retryable_timeout() -> None:
    event, data = _ANTHROPIC_STREAM[0]
    started = f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()

    def respond(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, stream=_KeepAliveStream(started))

    factory = GatewayModelFactory(
        ModelRegistry({"claude-opus-5": ModelSpec(native_provider="anthropic")}),
        GatewayEndpoint("https://gateway.example/v1"),
        "token",
        http_client_factory=lambda _: _deadline_client(respond, 0.2),
    )

    async with factory.create("claude-opus-5") as model:
        with pytest.raises(Exception) as raised:
            await asyncio.wait_for(model.ainvoke("hi"), timeout=10)
    assert not isinstance(raised.value, TimeoutError)
    assert is_retryable_error(raised.value)
