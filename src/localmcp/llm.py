"""Extensible model catalogs and config-selected LangChain construction."""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import re
import ssl
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Any, Literal, Protocol

TemperaturePolicy = Literal["zero", "omit"]
NativeProvider = Literal["anthropic", "openai"]
ModelDialect = Literal["anthropic", "openai"]
LLMBackend = Literal["openai_compatible", "native"]

_log = logging.getLogger(__name__)
_warned_unregistered: set[str] = set()


class ModelConfigurationError(ValueError):
    """Raised for invalid model specs or model/backend settings."""


@dataclass(frozen=True)
class ModelSpec:
    """Transport-neutral capabilities for one logical model ID.

    ``dialect`` selects the LangChain adapter used through a gateway and is
    independent of the native provider. When it is omitted,
    ``native_provider`` is used, then the OpenAI dialect. A missing
    ``native_provider`` deliberately marks a gateway-only model.
    ``max_output_tokens`` is the most the model can generate in one response;
    when it is omitted, the adapter's model profile or the provider decides.
    """

    temperature: TemperaturePolicy = "zero"
    supports_reasoning_effort: bool = False
    default_reasoning_effort: str | None = None
    native_provider: NativeProvider | None = None
    dialect: ModelDialect | None = None
    max_output_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.default_reasoning_effort is not None and not self.supports_reasoning_effort:
            raise ModelConfigurationError("default_reasoning_effort requires supports_reasoning_effort=True")
        if self.max_output_tokens is not None and self.max_output_tokens <= 0:
            raise ModelConfigurationError("max_output_tokens must be positive")

    @property
    def adapter_dialect(self) -> ModelDialect:
        """Return the gateway protocol dialect for this model."""
        return self.dialect or self.native_provider or "openai"


def infer_model_spec(model_id: str) -> ModelSpec:
    """Return conservative capabilities for a model the catalog does not know.

    The catalog is advisory: an unregistered ID is passed through to the
    provider, which remains the authority on whether it exists. Temperature is
    omitted because newer models increasingly reject it, and the provider is
    inferred from well-known ID prefixes. A caller-requested reasoning effort
    is passed through for the provider to accept or reject.
    """
    normalized = model_id.lower()
    if normalized.startswith("claude-"):
        return ModelSpec(temperature="omit", supports_reasoning_effort=True, native_provider="anthropic")
    if normalized.startswith("gpt-") or re.match(r"o\d", normalized):
        return ModelSpec(temperature="omit", supports_reasoning_effort=True, native_provider="openai")
    return ModelSpec(temperature="omit", supports_reasoning_effort=True)


class ModelRegistry:
    """Immutable, advisory model catalog with explicit application overlays.

    Registered specs describe known capabilities. Unregistered IDs resolve to
    :func:`infer_model_spec` so new models work without a library release.
    """

    def __init__(self, models: Mapping[str, ModelSpec] | None = None):
        self._models = dict(models or {})
        if any(not model_id.strip() for model_id in self._models):
            raise ModelConfigurationError("model IDs must not be blank")

    def resolve(self, model_id: str) -> ModelSpec:
        if not model_id.strip():
            raise ModelConfigurationError("model IDs must not be blank")
        spec = self._models.get(model_id)
        if spec is None:
            if model_id not in _warned_unregistered:
                _warned_unregistered.add(model_id)
                _log.warning('model "%s" is not in the catalog; using inferred capabilities', model_id)
            return infer_model_spec(model_id)
        return spec

    def overlay(self, models: Mapping[str, ModelSpec]) -> ModelRegistry:
        return ModelRegistry({**self._models, **models})

    def as_mapping(self) -> Mapping[str, ModelSpec]:
        return dict(self._models)


# Conservative cross-gateway capabilities. Applications and localmcp.toml
# [llm.models] may overlay this catalog when an endpoint has stricter or newer
# behavior; unlisted IDs fall back to infer_model_spec. Invocation policy
# such as the chosen reasoning effort intentionally remains caller-owned.
DEFAULT_MODEL_SPECS: Mapping[str, ModelSpec] = {
    "claude-opus-5": ModelSpec(supports_reasoning_effort=True, native_provider="anthropic"),
    "claude-opus-4-8": ModelSpec(temperature="omit", supports_reasoning_effort=True, native_provider="anthropic"),
    "claude-sonnet-5": ModelSpec(temperature="omit", supports_reasoning_effort=True, native_provider="anthropic"),
    "claude-sonnet-4-6": ModelSpec(supports_reasoning_effort=True, native_provider="anthropic"),
    "claude-haiku-4-5-20251001": ModelSpec(native_provider="anthropic"),
    "gpt-5.6": ModelSpec(
        temperature="omit",
        supports_reasoning_effort=True,
        native_provider="openai",
    ),
    "gemini-3.1-pro-preview": ModelSpec(),
    "gemini-3.5-flash": ModelSpec(),
    "grok-4.6": ModelSpec(temperature="omit"),
}
DEFAULT_MODEL_REGISTRY = ModelRegistry(DEFAULT_MODEL_SPECS)


# Seconds one model response may take, from sending its request to its last
# streamed byte. Streaming leaves the SDKs' read timeout unable to bound a
# response: it limits only the wait between bytes, and providers keep sending
# keep-alive events while a model thinks or stalls. The deadline cannot tell a
# stalled response from a long one, so a model whose uncapped responses
# routinely take longer needs a larger timeout or an output limit.
DEFAULT_RESPONSE_TIMEOUT = 600.0


def _validate_response_timeout(response_timeout: float) -> None:
    # inf would disable the deadline and nan would expire it immediately.
    if not (math.isfinite(response_timeout) and response_timeout > 0):
        raise ModelConfigurationError("response_timeout must be a finite positive number")


def _validate_transport(keepalive_expiry: float, response_timeout: float) -> None:
    if keepalive_expiry < 0:
        raise ModelConfigurationError("keepalive_expiry must be non-negative")
    _validate_response_timeout(response_timeout)


@dataclass(frozen=True)
class GatewayEndpoint:
    """Shared endpoint and transport settings for a model gateway.

    ``stream_usage`` asks OpenAI-dialect responses to report token usage, which
    a streamed response otherwise omits. Disable it only for a gateway that
    rejects ``stream_options``.
    """

    base_url: str
    default_headers: Mapping[str, str] | None = None
    verify: bool | str | ssl.SSLContext = True
    keepalive_expiry: float = 5.0
    response_timeout: float = DEFAULT_RESPONSE_TIMEOUT
    stream_usage: bool = True

    def __post_init__(self) -> None:
        if not self.base_url.strip():
            raise ModelConfigurationError("gateway base URL must not be blank")
        _validate_transport(self.keepalive_expiry, self.response_timeout)


@dataclass(frozen=True)
class NativeTransport:
    """HTTP transport settings shared by native provider connections."""

    verify: bool | str | ssl.SSLContext = True
    keepalive_expiry: float = 5.0
    response_timeout: float = DEFAULT_RESPONSE_TIMEOUT

    def __post_init__(self) -> None:
        _validate_transport(self.keepalive_expiry, self.response_timeout)


@dataclass(frozen=True)
class LLMBackendConfig:
    """Explicit deployment routing selected by application-owned config.

    The library deliberately does not infer a backend from available keys.
    ``base_url`` is required only for ``openai_compatible`` and rejected for
    ``native`` so stale gateway routing cannot leak into native provider
    construction. ``stream_usage`` applies only to the gateway; native OpenAI
    always reports streamed usage.
    """

    backend: LLMBackend
    base_url: str | None = None
    default_headers: Mapping[str, str] | None = None
    verify: bool | str | ssl.SSLContext = True
    keepalive_expiry: float = 5.0
    response_timeout: float = DEFAULT_RESPONSE_TIMEOUT
    stream_usage: bool = True

    def __post_init__(self) -> None:
        if self.backend not in ("openai_compatible", "native"):
            raise ModelConfigurationError(f"unsupported LLM backend: {self.backend!r}")
        if self.backend == "openai_compatible":
            if self.base_url is None or not self.base_url.strip():
                raise ModelConfigurationError("openai_compatible LLM backend requires base_url")
        elif self.base_url is not None:
            raise ModelConfigurationError("native LLM backend does not accept a gateway base_url")
        _validate_transport(self.keepalive_expiry, self.response_timeout)

    def gateway_endpoint(self) -> GatewayEndpoint:
        if self.backend != "openai_compatible" or self.base_url is None:
            raise ModelConfigurationError("gateway endpoint requested for a non-openai_compatible backend")
        return GatewayEndpoint(
            self.base_url,
            default_headers=self.default_headers,
            verify=self.verify,
            keepalive_expiry=self.keepalive_expiry,
            response_timeout=self.response_timeout,
            stream_usage=self.stream_usage,
        )

    def native_transport(self) -> NativeTransport:
        if self.backend != "native":
            raise ModelConfigurationError("native transport requested for a non-native backend")
        return NativeTransport(
            verify=self.verify,
            keepalive_expiry=self.keepalive_expiry,
            response_timeout=self.response_timeout,
        )


class AsyncHTTPClient(Protocol):
    async def aclose(self) -> None: ...


# Backward-compatible name for callers that used the original all-OpenAI
# factory before dialect-aware gateway routing was added.
OpenAICompatibleEndpoint = GatewayEndpoint

HTTPClientFactory = Callable[[GatewayEndpoint], AsyncHTTPClient]
NativeHTTPClientFactory = Callable[[NativeProvider, NativeTransport], AsyncHTTPClient]
APIKey = str | Callable[[], str] | Callable[[], Awaitable[str]]


class OwnedChatModel:
    """Async context manager owning a model and its HTTP transport."""

    def __init__(self, model: Any, owner: AsyncHTTPClient):
        self.model = model
        self._owner = owner

    async def __aenter__(self) -> Any:
        return self.model

    async def __aexit__(self, *exc: object) -> None:
        await self._owner.aclose()


class ModelFactory(Protocol):
    """Structural interface for constructing owned chat models."""

    def validate(self, model_id: str, *, reasoning_effort: str | None = None) -> ModelSpec:
        """Validate model/backend compatibility without allocating a transport."""
        ...

    def create(
        self,
        model_id: str,
        *,
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> OwnedChatModel: ...


@lru_cache(maxsize=1)
def _response_deadline_client_type() -> type[Any]:
    import httpx2

    def timeout_error(timeout: float, request: httpx2.Request) -> httpx2.ReadTimeout:
        # A transport timeout, so callers retry it like any other stalled read.
        return httpx2.ReadTimeout(f"model response exceeded the {timeout:g} s response timeout", request=request)

    class DeadlineStream(httpx2.AsyncByteStream):
        """Response body that fails once its response's deadline passes."""

        def __init__(self, stream: Any, *, deadline: float, timeout: float, request: httpx2.Request) -> None:
            self._stream = stream
            self._deadline = deadline
            self._timeout = timeout
            self._request = request

        async def __aiter__(self) -> AsyncIterator[bytes]:
            chunks = aiter(self._stream)
            while True:
                try:
                    async with asyncio.timeout_at(self._deadline):
                        chunk = await anext(chunks)
                except StopAsyncIteration:
                    return
                except TimeoutError:
                    raise timeout_error(self._timeout, self._request) from None
                yield chunk

        async def aclose(self) -> None:
            await self._stream.aclose()

    class ResponseDeadlineClient(httpx2.AsyncClient):
        """Client bounding each response from request to last byte, however steadily bytes arrive."""

        def __init__(self, *, response_timeout: float, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            self._response_timeout = response_timeout

        async def send(self, request: httpx2.Request, *, stream: bool = False, **kwargs: Any) -> httpx2.Response:
            deadline = asyncio.get_running_loop().time() + self._response_timeout
            try:
                async with asyncio.timeout_at(deadline):
                    response = await super().send(request, stream=stream, **kwargs)
            except TimeoutError:
                raise timeout_error(self._response_timeout, request) from None
            if stream:
                response.stream = DeadlineStream(
                    response.stream, deadline=deadline, timeout=self._response_timeout, request=request
                )
            return response

    return ResponseDeadlineClient


def response_deadline_http_client(*, response_timeout: float, **client_kwargs: Any) -> AsyncHTTPClient:
    """Return an ``httpx2.AsyncClient`` that bounds each response to ``response_timeout`` seconds.

    The default transports use this client. A custom ``http_client_factory``
    replaces them, so its clients enforce the response timeout only when they
    are built here; ``client_kwargs`` reach ``httpx2.AsyncClient`` unchanged.
    """
    _validate_response_timeout(response_timeout)
    client: AsyncHTTPClient = _response_deadline_client_type()(response_timeout=response_timeout, **client_kwargs)
    return client


def _httpx2_client(
    *, verify: bool | str | ssl.SSLContext, keepalive_expiry: float, response_timeout: float
) -> AsyncHTTPClient:
    import httpx2

    return response_deadline_http_client(
        response_timeout=response_timeout,
        verify=verify,
        limits=httpx2.Limits(keepalive_expiry=keepalive_expiry),
    )


def _default_http_client(endpoint: GatewayEndpoint) -> AsyncHTTPClient:
    return _httpx2_client(
        verify=endpoint.verify,
        keepalive_expiry=endpoint.keepalive_expiry,
        response_timeout=endpoint.response_timeout,
    )


def _default_native_http_client(provider: NativeProvider, transport: NativeTransport) -> AsyncHTTPClient:
    del provider
    return _httpx2_client(
        verify=transport.verify,
        keepalive_expiry=transport.keepalive_expiry,
        response_timeout=transport.response_timeout,
    )


# ChatAnthropic falls back to 4,096 output tokens for a model it has no profile
# for, which a single reasoning turn can exhaust. Every current Claude model
# accepts at least this many; an older Claude model or another model behind an
# Anthropic-dialect gateway with a smaller limit needs its max_output_tokens
# catalogued.
_UNPROFILED_ANTHROPIC_OUTPUT_TOKENS = 64_000


def _validate_default_output_tokens(value: int | None) -> None:
    if value is not None and value <= 0:
        raise ModelConfigurationError("default_max_output_tokens must be positive")


def _output_tokens(value: int | None, default: int | None, spec: ModelSpec) -> int | None:
    """Return the per-call output limit, or ``None`` for the model's maximum.

    A per-call limit is used as given. The factory-wide default applies to
    every model, so it never raises a model above its catalogued maximum.
    """
    if value is not None:
        if value <= 0:
            raise ModelConfigurationError("max_output_tokens must be positive")
        return value
    if default is not None and spec.max_output_tokens is not None:
        return min(default, spec.max_output_tokens)
    return default if default is not None else spec.max_output_tokens


def _invocation_kwargs(model_id: str, spec: ModelSpec, reasoning_effort: str | None) -> dict[str, Any]:
    selected_effort = reasoning_effort if reasoning_effort is not None else spec.default_reasoning_effort
    if selected_effort is not None and not spec.supports_reasoning_effort:
        raise ModelConfigurationError(f'model "{model_id}" does not support reasoning_effort')
    kwargs: dict[str, Any] = {}
    # An effort turns on Anthropic thinking, which rejects a temperature.
    anthropic = "anthropic" in (spec.native_provider, spec.adapter_dialect)
    if spec.temperature == "zero" and not (anthropic and selected_effort is not None):
        kwargs["temperature"] = 0
    if selected_effort is not None:
        kwargs["reasoning_effort"] = selected_effort
    return kwargs


def _openai_api_key(api_key: APIKey) -> APIKey:
    if not isinstance(api_key, str):
        return api_key
    raw_api_key = api_key

    def provide_api_key() -> str:
        return raw_api_key

    return provide_api_key


def _anthropic_api_key(api_key: APIKey) -> str:
    if isinstance(api_key, str):
        return api_key
    resolved = api_key()
    if inspect.isawaitable(resolved):
        raise ModelConfigurationError("Anthropic API keys must be strings or synchronous callables")
    return resolved


@lru_cache(maxsize=1)
def _owned_chat_anthropic_type() -> type[Any]:
    from langchain_anthropic import ChatAnthropic
    from pydantic import PrivateAttr

    class OwnedTransportChatAnthropic(ChatAnthropic):
        """Async-only ChatAnthropic using the transport owned by this library."""

        _localmcp_async_client: Any = PrivateAttr()

        @cached_property
        def _client(self) -> Any:
            raise RuntimeError("localmcp-owned Anthropic models support async invocation only")

        @cached_property
        def _async_client(self) -> Any:
            return self._localmcp_async_client

    return OwnedTransportChatAnthropic


def _create_openai_model(
    *,
    model_id: str,
    api_key: APIKey,
    client: AsyncHTTPClient,
    max_output_tokens: int | None,
    kwargs: Mapping[str, Any],
    endpoint: GatewayEndpoint | None,
) -> Any:
    from langchain_openai import ChatOpenAI

    # Responses stream so a long generation never waits on one idle read, and
    # LangChain reassembles them into one message for the caller.
    constructor: dict[str, Any] = {
        "model": model_id,
        "api_key": _openai_api_key(api_key),
        "http_async_client": client,
        "streaming": True,
        # LangChain reports a streamed response's usage only when it is asked for.
        "stream_usage": endpoint is None or endpoint.stream_usage,
        **kwargs,
    }
    # Without a limit the provider allows the model's maximum.
    if max_output_tokens is not None:
        constructor["max_completion_tokens"] = max_output_tokens
    if endpoint is not None:
        constructor.update(base_url=endpoint.base_url, default_headers=dict(endpoint.default_headers or {}))
    return ChatOpenAI(**constructor)


def _create_anthropic_model(
    *,
    model_id: str,
    api_key: APIKey,
    client: AsyncHTTPClient,
    max_output_tokens: int | None,
    kwargs: Mapping[str, Any],
    endpoint: GatewayEndpoint | None,
) -> Any:
    import anthropic

    raw_api_key = _anthropic_api_key(api_key)
    client_kwargs: dict[str, Any] = {"api_key": raw_api_key, "http_client": client}
    # The Anthropic SDK refuses a non-streaming request whose output limit could
    # outlast its timeout, so responses stream and LangChain reassembles them.
    model_kwargs: dict[str, Any] = {
        "model": model_id,
        "api_key": raw_api_key,
        "max_tokens": max_output_tokens,
        "streaming": True,
        **kwargs,
    }
    if endpoint is not None:
        headers = dict(endpoint.default_headers or {})
        client_kwargs.update(base_url=endpoint.base_url, default_headers=headers)
        model_kwargs.update(base_url=endpoint.base_url, default_headers=headers)
    async_client = anthropic.AsyncAnthropic(**client_kwargs)
    # The Messages API requires a limit. Without one, ChatAnthropic uses the
    # maximum from the model's profile.
    model = _owned_chat_anthropic_type()(**model_kwargs)
    if max_output_tokens is None and not (model.profile or {}).get("max_output_tokens"):
        model.max_tokens = _UNPROFILED_ANTHROPIC_OUTPUT_TOKENS
    model._localmcp_async_client = async_client
    return model


class OpenAICompatibleModelFactory:
    """Construct all registered models through one OpenAI-compatible adapter.

    This compatibility factory intentionally keeps its historical behavior:
    every model uses ``ChatOpenAI``. New deployments that need per-model
    dialect routing should use :class:`GatewayModelFactory`.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        endpoint: GatewayEndpoint,
        api_key: APIKey,
        *,
        http_client_factory: HTTPClientFactory | None = None,
        default_max_output_tokens: int | None = None,
    ) -> None:
        _validate_default_output_tokens(default_max_output_tokens)
        self.registry = registry
        self.endpoint = endpoint
        self.api_key = api_key
        self.http_client_factory = http_client_factory or _default_http_client
        self.default_max_output_tokens = default_max_output_tokens

    def validate(self, model_id: str, *, reasoning_effort: str | None = None) -> ModelSpec:
        spec = self.registry.resolve(model_id)
        _invocation_kwargs(model_id, spec, reasoning_effort)
        return spec

    def create(
        self,
        model_id: str,
        *,
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> OwnedChatModel:
        """Create an owned ``ChatOpenAI`` without performing network I/O."""
        spec = self.validate(model_id, reasoning_effort=reasoning_effort)
        output_tokens = _output_tokens(max_output_tokens, self.default_max_output_tokens, spec)
        kwargs = _invocation_kwargs(model_id, spec, reasoning_effort)
        client = self.http_client_factory(self.endpoint)
        model = _create_openai_model(
            model_id=model_id,
            api_key=self.api_key,
            client=client,
            max_output_tokens=output_tokens,
            kwargs=kwargs,
            endpoint=self.endpoint,
        )
        return OwnedChatModel(model, client)


class GatewayModelFactory:
    """Route registered models through one gateway using their API dialect."""

    def __init__(
        self,
        registry: ModelRegistry,
        endpoint: GatewayEndpoint,
        api_key: APIKey,
        *,
        http_client_factory: HTTPClientFactory | None = None,
        default_max_output_tokens: int | None = None,
    ) -> None:
        _validate_default_output_tokens(default_max_output_tokens)
        self.registry = registry
        self.endpoint = endpoint
        self.api_key = api_key
        self.http_client_factory = http_client_factory or _default_http_client
        self.default_max_output_tokens = default_max_output_tokens

    def validate(self, model_id: str, *, reasoning_effort: str | None = None) -> ModelSpec:
        spec = self.registry.resolve(model_id)
        _invocation_kwargs(model_id, spec, reasoning_effort)
        return spec

    def create(
        self,
        model_id: str,
        *,
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> OwnedChatModel:
        spec = self.validate(model_id, reasoning_effort=reasoning_effort)
        output_tokens = _output_tokens(max_output_tokens, self.default_max_output_tokens, spec)
        kwargs = _invocation_kwargs(model_id, spec, reasoning_effort)
        client = self.http_client_factory(self.endpoint)
        if spec.adapter_dialect == "anthropic":
            model = _create_anthropic_model(
                model_id=model_id,
                api_key=self.api_key,
                client=client,
                max_output_tokens=output_tokens,
                kwargs=kwargs,
                endpoint=self.endpoint,
            )
        else:
            model = _create_openai_model(
                model_id=model_id,
                api_key=self.api_key,
                client=client,
                max_output_tokens=output_tokens,
                kwargs=kwargs,
                endpoint=self.endpoint,
            )
        return OwnedChatModel(model, client)


class NativeModelFactory:
    """Route each registered model directly to its declared native provider."""

    def __init__(
        self,
        registry: ModelRegistry,
        *,
        openai_api_key: APIKey | None = None,
        anthropic_api_key: APIKey | None = None,
        transport: NativeTransport | None = None,
        http_client_factory: NativeHTTPClientFactory | None = None,
        default_max_output_tokens: int | None = None,
    ) -> None:
        _validate_default_output_tokens(default_max_output_tokens)
        self.registry = registry
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key
        self.transport = transport or NativeTransport()
        self.http_client_factory = http_client_factory or _default_native_http_client
        self.default_max_output_tokens = default_max_output_tokens

    def validate(self, model_id: str, *, reasoning_effort: str | None = None) -> ModelSpec:
        spec = self.registry.resolve(model_id)
        provider = spec.native_provider
        if provider is None:
            raise ModelConfigurationError(f'model "{model_id}" is available only through a gateway')
        api_key = self.anthropic_api_key if provider == "anthropic" else self.openai_api_key
        if api_key is None:
            raise ModelConfigurationError(f'native {provider} API key is required for model "{model_id}"')
        _invocation_kwargs(model_id, spec, reasoning_effort)
        return spec

    def create(
        self,
        model_id: str,
        *,
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> OwnedChatModel:
        spec = self.validate(model_id, reasoning_effort=reasoning_effort)
        provider = spec.native_provider
        assert provider is not None
        api_key = self.anthropic_api_key if provider == "anthropic" else self.openai_api_key
        assert api_key is not None
        output_tokens = _output_tokens(max_output_tokens, self.default_max_output_tokens, spec)
        kwargs = _invocation_kwargs(model_id, spec, reasoning_effort)
        client = self.http_client_factory(provider, self.transport)
        if provider == "anthropic":
            model = _create_anthropic_model(
                model_id=model_id,
                api_key=api_key,
                client=client,
                max_output_tokens=output_tokens,
                kwargs=kwargs,
                endpoint=None,
            )
        else:
            model = _create_openai_model(
                model_id=model_id,
                api_key=api_key,
                client=client,
                max_output_tokens=output_tokens,
                kwargs=kwargs,
                endpoint=None,
            )
        return OwnedChatModel(model, client)


class ConfiguredModelFactory:
    """Delegate the stable ``ModelFactory`` API to an explicit backend."""

    def __init__(
        self,
        registry: ModelRegistry,
        config: LLMBackendConfig,
        *,
        gateway_api_key: APIKey | None = None,
        openai_api_key: APIKey | None = None,
        anthropic_api_key: APIKey | None = None,
        gateway_http_client_factory: HTTPClientFactory | None = None,
        native_http_client_factory: NativeHTTPClientFactory | None = None,
        default_max_output_tokens: int | None = None,
    ) -> None:
        self.registry = registry
        self.config = config
        if config.backend == "openai_compatible":
            if gateway_api_key is None:
                raise ModelConfigurationError("gateway API key is required for the openai_compatible backend")
            delegate: ModelFactory = GatewayModelFactory(
                registry,
                config.gateway_endpoint(),
                gateway_api_key,
                http_client_factory=gateway_http_client_factory,
                default_max_output_tokens=default_max_output_tokens,
            )
        else:
            delegate = NativeModelFactory(
                registry,
                openai_api_key=openai_api_key,
                anthropic_api_key=anthropic_api_key,
                transport=config.native_transport(),
                http_client_factory=native_http_client_factory,
                default_max_output_tokens=default_max_output_tokens,
            )
        self._delegate = delegate

    def validate(self, model_id: str, *, reasoning_effort: str | None = None) -> ModelSpec:
        return self._delegate.validate(model_id, reasoning_effort=reasoning_effort)

    def create(
        self,
        model_id: str,
        *,
        max_output_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> OwnedChatModel:
        return self._delegate.create(
            model_id,
            max_output_tokens=max_output_tokens,
            reasoning_effort=reasoning_effort,
        )
