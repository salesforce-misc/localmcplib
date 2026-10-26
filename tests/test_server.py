from __future__ import annotations

import contextlib
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client
from fastmcp.tools import tool

import localmcp
import localmcp.server as server_module
from localmcp.config import ConfigError, ServerConfig
from localmcp.llm import DEFAULT_RESPONSE_TIMEOUT, ConfiguredModelFactory, ModelFactory, ModelSpec
from localmcp.observability.telemetry import LangfuseTelemetry
from localmcp.server import RuntimeUnavailableError, current_runtime, main


@dataclass(frozen=True)
class _Config:
    greeting: str = "hello"


class _Runtime:
    instances: list[_Runtime] = []

    def __init__(
        self,
        config: _Config,
        state_root: Path,
        model_factory: ModelFactory,
        *,
        telemetry: LangfuseTelemetry,
    ) -> None:
        self.config = config
        self.state_root = state_root
        self.model_factory = model_factory
        self.telemetry = telemetry
        self.started = False
        self.closed = False
        self.__class__.instances.append(self)

    async def start(self) -> None:
        self.started = True

    async def close(self) -> None:
        self.closed = True


@tool
def example_tool() -> str:
    return "ok"


def _parse_config(config: ServerConfig) -> _Config:
    application = config.values["application"]
    assert isinstance(application, dict)
    greeting = application["greeting"]
    assert isinstance(greeting, str)
    return _Config(greeting=greeting)


def _write_document(home: Path) -> dict[str, str]:
    config_home = home / "config"
    state_home = home / "state"
    path = config_home / "localmcp/localmcp.toml"
    path.parent.mkdir(parents=True)
    path.write_text(
        """
schema_version = 1

[observability]
log_level = "INFO"

[llm]
backend = "openai_compatible"
base_url = "https://llm.example.test/v1"

[application]
greeting = "hello"

[secrets.llm_api_key]
env_vars = ["SHARED_LLM_API_KEY"]

[server.example-mcp.observability]
log_level = "DEBUG"

[server.example-mcp.application]
greeting = "hello from overlay"
""",
        encoding="utf-8",
    )
    return {
        "XDG_CONFIG_HOME": str(config_home),
        "XDG_STATE_HOME": str(state_home),
        "SHARED_LLM_API_KEY": "test-token",
    }


class _SharedRuntime:
    instances: list[_SharedRuntime] = []

    def __init__(
        self,
        config: ServerConfig,
        state_root: Path,
        model_factory: ModelFactory,
        *,
        telemetry: LangfuseTelemetry,
    ) -> None:
        self.config = config
        self.state_root = state_root
        self.model_factory = model_factory
        self.telemetry = telemetry
        self.started = False
        self.closed = False
        self.__class__.instances.append(self)

    async def start(self) -> None:
        self.started = True

    async def close(self) -> None:
        self.closed = True


def _server(tools: Sequence[Any] = ()) -> localmcp.STDIOServer[_Config, _Runtime]:
    return localmcp.STDIOServer(
        name="example-mcp",
        tools=tools,
        config_parser=_parse_config,
        runtime_factory=_Runtime,
    )


@pytest.fixture(autouse=True)
async def _reset_runtime() -> Iterator[None]:
    _Runtime.instances = []
    yield
    server = _server()
    with contextlib.suppress(BaseException):
        await server.stop_runtime()


def test_current_runtime_raises_before_start() -> None:
    with pytest.raises(RuntimeUnavailableError, match="not initialized"):
        current_runtime(_Runtime)


async def test_start_composes_config_secrets_models_telemetry_and_state(tmp_path: Path) -> None:
    env = _write_document(tmp_path)
    server = _server()

    config = await server.start_runtime(env=env, home=tmp_path)
    runtime = current_runtime(_Runtime)

    assert config == _Config(greeting="hello from overlay")
    assert server.load_config(env=env, home=tmp_path) == _Config(greeting="hello from overlay")
    assert runtime.started
    assert runtime.state_root == tmp_path / "state/localmcp/example-mcp"
    assert isinstance(runtime.model_factory, ConfiguredModelFactory)
    assert runtime.model_factory.config.backend == "openai_compatible"
    assert runtime.model_factory.config.base_url == "https://llm.example.test/v1"
    assert runtime.model_factory.config.response_timeout == DEFAULT_RESPONSE_TIMEOUT
    async with runtime.model_factory.create("gpt-5.6") as model:
        assert model is not None


async def test_llm_transport_settings_reach_the_backend_config(tmp_path: Path) -> None:
    env = _write_document(tmp_path)
    path = tmp_path / "config/localmcp/localmcp.toml"
    path.write_text(
        path.read_text(encoding="utf-8").replace("[llm]\n", "[llm]\nresponse_timeout = 90\nstream_usage = false\n", 1),
        encoding="utf-8",
    )

    await _server().start_runtime(env=env, home=tmp_path)

    model_factory = current_runtime(_Runtime).model_factory
    assert isinstance(model_factory, ConfiguredModelFactory)
    assert model_factory.config.response_timeout == 90.0
    assert model_factory.config.gateway_endpoint().response_timeout == 90.0
    assert model_factory.config.gateway_endpoint().stream_usage is False


async def test_start_without_application_parser_uses_effective_application_config(tmp_path: Path) -> None:
    env = _write_document(tmp_path)
    server: localmcp.STDIOServer[ServerConfig, _SharedRuntime] = localmcp.STDIOServer(
        name="example-mcp",
        tools=(),
        runtime_factory=_SharedRuntime,
    )

    config = await server.start_runtime(env=env, home=tmp_path)
    runtime = current_runtime(_SharedRuntime)

    assert config.name == "example-mcp"
    assert config.values == {"application": {"greeting": "hello from overlay"}}
    assert runtime.config is config
    assert runtime.started


async def test_native_backend_requires_only_declared_provider_secret(tmp_path: Path) -> None:
    config_home = tmp_path / "config"
    path = config_home / "localmcp/localmcp.toml"
    path.parent.mkdir(parents=True)
    path.write_text(
        """
schema_version = 1

[llm]
backend = "native"

[secrets.anthropic_api_key]
env_vars = ["ANTHROPIC_API_KEY"]
""",
        encoding="utf-8",
    )
    env = {
        "XDG_CONFIG_HOME": str(config_home),
        "XDG_STATE_HOME": str(tmp_path / "state"),
        "ANTHROPIC_API_KEY": "test-token",
    }
    server: localmcp.STDIOServer[ServerConfig, _SharedRuntime] = localmcp.STDIOServer(
        name="example-mcp",
        tools=(),
        runtime_factory=_SharedRuntime,
    )

    config = await server.start_runtime(env=env, home=tmp_path)
    runtime = current_runtime(_SharedRuntime)

    assert config.values == {}
    async with runtime.model_factory.create("claude-sonnet-4-6") as model:
        assert model is not None


async def test_restart_closes_previous_runtime(tmp_path: Path) -> None:
    env = _write_document(tmp_path)
    server = _server()

    await server.start_runtime(env=env, home=tmp_path)
    first = current_runtime(_Runtime)
    await server.start_runtime(env=env, home=tmp_path)

    assert first.closed
    assert current_runtime(_Runtime) is not first


async def test_stop_closes_and_clears_runtime(tmp_path: Path) -> None:
    env = _write_document(tmp_path)
    server = _server()
    await server.start_runtime(env=env, home=tmp_path)
    runtime = current_runtime(_Runtime)

    await server.stop_runtime()

    assert runtime.closed
    with pytest.raises(RuntimeUnavailableError, match="not initialized"):
        current_runtime(_Runtime)


async def test_lifespan_registers_tools_and_owns_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _write_document(tmp_path)
    server = _server([example_tool])
    monkeypatch.setattr("localmcp.server.os.environ", env)
    monkeypatch.setattr("localmcp.server.Path.home", lambda: tmp_path)

    async with Client(server.mcp) as client:
        result = await client.call_tool("example_tool", {})
        assert result.data == "ok"
        assert current_runtime(_Runtime).started

    assert _Runtime.instances[-1].closed


def test_run_configures_stdio_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    env = _write_document(tmp_path)
    server = _server([example_tool])
    calls: list[Mapping[str, object]] = []
    monkeypatch.setattr("localmcp.server.os.environ", env)
    monkeypatch.setattr("localmcp.server.Path.home", lambda: tmp_path)
    monkeypatch.setattr(server.mcp, "run", lambda **kwargs: calls.append(kwargs))

    server.run()

    assert calls == [{"transport": "stdio", "show_banner": False}]
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_run_converts_serve_failure_to_silent_system_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    env = _write_document(tmp_path)
    server = _server()
    monkeypatch.setattr("localmcp.server.os.environ", env)
    monkeypatch.setattr("localmcp.server.Path.home", lambda: tmp_path)

    def fail(**_kwargs: object) -> None:
        raise RuntimeError("sensitive detail")

    monkeypatch.setattr(server.mcp, "run", fail)

    with pytest.raises(SystemExit) as exc_info:
        server.run()

    assert exc_info.value.code == 1
    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == ""


@pytest.mark.parametrize(
    ("config_text", "parser_error", "message"),
    [
        ("[llm]\nbackend = 'other'", None, "unsupported llm.backend: 'other'"),
        (
            "[llm]\nbackend = 'native'",
            "pr_review.allowed_hosts: extra inputs are not permitted",
            "pr_review.allowed_hosts: extra inputs are not permitted",
        ),
    ],
)
def test_run_reports_config_error_to_stderr(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    config_text: str,
    parser_error: str | None,
    message: str,
) -> None:
    config_home = tmp_path / "config"
    path = config_home / "localmcp/localmcp.toml"
    path.parent.mkdir(parents=True)
    path.write_text(config_text)
    env = {"XDG_CONFIG_HOME": str(config_home)}
    monkeypatch.setattr("localmcp.server.os.environ", env)
    monkeypatch.setattr("localmcp.server.Path.home", lambda: tmp_path)
    server = _server()
    if parser_error is not None:

        def reject_config(_config: ServerConfig) -> _Config:
            raise ConfigError(parser_error)

        server.config_parser = reject_config
    calls: list[None] = []
    monkeypatch.setattr(server.mcp, "run", lambda **_kwargs: calls.append(None))

    with pytest.raises(SystemExit) as exc_info:
        server.run()

    assert exc_info.value.code == 1
    assert calls == []
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"example-mcp: configuration error: {message}\n"


@pytest.mark.parametrize(
    ("shared_config", "message"),
    [
        ('llm = "invalid"', "llm must be a TOML table"),
        ("[llm]\nbackend = 'native'\nunexpected = true", "llm has unsupported fields"),
        ("[llm]\nbackend = 1", "llm.backend must be a string"),
        ("[llm]\nbackend = 'other'", "unsupported llm.backend"),
        ("[llm]\nbackend = 'native'\nbase_url = 1", "llm.base_url must be a string"),
        (
            "[llm]\nbackend = 'native'\ndefault_headers = { valid = 1 }",
            "llm.default_headers must be a table of strings",
        ),
        ("[llm]\nbackend = 'native'\nverify = 1", "llm.verify must be a boolean or string"),
        ("[llm]\nbackend = 'native'\nkeepalive_expiry = true", "llm.keepalive_expiry must be a number"),
        ("[llm]\nbackend = 'native'\nkeepalive_expiry = -1", "llm: keepalive_expiry must be non-negative"),
        ("[llm]\nbackend = 'native'\nresponse_timeout = true", "llm.response_timeout must be a number"),
        ("[llm]\nbackend = 'native'\nresponse_timeout = '600'", "llm.response_timeout must be a number"),
        ("[llm]\nbackend = 'native'\nresponse_timeout = 0", "llm: response_timeout must be a finite positive"),
        ("[llm]\nbackend = 'native'\nresponse_timeout = inf", "llm: response_timeout must be a finite positive"),
        ("[llm]\nbackend = 'native'\nresponse_timeout = nan", "llm: response_timeout must be a finite positive"),
        ("[llm]\nbackend = 'openai_compatible'", "llm: openai_compatible LLM backend requires base_url"),
        ("[llm]\nbackend = 'native'\nstream_usage = 'no'", "llm.stream_usage must be a boolean"),
        ("observability = 'invalid'\n[llm]\nbackend = 'native'", "observability must be a TOML table"),
        (
            "[llm]\nbackend = 'native'\n[observability]\nunexpected = true",
            "observability has unsupported fields",
        ),
        (
            "[llm]\nbackend = 'native'\n[observability]\nlog_level = 1",
            "observability.log_level must be a string",
        ),
    ],
)
def test_load_config_rejects_invalid_shared_settings(
    tmp_path: Path,
    shared_config: str,
    message: str,
) -> None:
    config_home = tmp_path / "config"
    path = config_home / "localmcp/localmcp.toml"
    path.parent.mkdir(parents=True)
    path.write_text(shared_config)
    env = {"XDG_CONFIG_HOME": str(config_home)}

    with pytest.raises(ConfigError, match=re.escape(message)):
        _server().load_config(env=env, home=tmp_path)


async def test_llm_models_overlay_catalog_and_inferred_specs(tmp_path: Path) -> None:
    env = _write_document(tmp_path)
    path = tmp_path / "config/localmcp/localmcp.toml"
    path.write_text(
        path.read_text(encoding="utf-8")
        + """
[llm.models."claude-opus-5"]
temperature = "omit"

[llm.models."gpt-7"]
default_reasoning_effort = "medium"

[llm.models."house-model"]
dialect = "anthropic"
temperature = "zero"

[llm.models."claude-via-openai"]
dialect = "openai"

[llm.models."claude-small"]
max_output_tokens = 8192
""",
        encoding="utf-8",
    )
    server = _server()

    await server.start_runtime(env=env, home=tmp_path)
    factory = current_runtime(_Runtime).model_factory

    assert factory.validate("claude-opus-5") == ModelSpec(
        temperature="omit", supports_reasoning_effort=True, native_provider="anthropic"
    )
    assert factory.validate("gpt-7").default_reasoning_effort == "medium"
    assert factory.validate("house-model") == ModelSpec(
        temperature="zero", supports_reasoning_effort=True, dialect="anthropic"
    )
    assert factory.validate("claude-via-openai") == ModelSpec(
        temperature="omit", supports_reasoning_effort=True, native_provider="anthropic", dialect="openai"
    )
    assert factory.validate("claude-small").max_output_tokens == 8192
    assert factory.validate("claude-sonnet-4-6") == ModelSpec(
        supports_reasoning_effort=True, native_provider="anthropic"
    )


@pytest.mark.parametrize(
    ("models", "message"),
    [
        ('[llm.models]\nm = "x"', "llm.models.m must be a TOML table"),
        ("[llm.models.m]\nunexpected = 1", "llm.models.m has unsupported fields: unexpected"),
        ("[llm.models.m]\ntemperature = 'hot'", "llm.models.m.temperature must be one of: zero, omit"),
        ("[llm.models.m]\ndialect = 'gemini'", "llm.models.m.dialect must be one of: anthropic, openai"),
        ("[llm.models.m]\nsupports_reasoning_effort = 'yes'", "supports_reasoning_effort must be a boolean"),
        ("[llm.models.m]\ndefault_reasoning_effort = 1", "default_reasoning_effort must be a string"),
        ("[llm.models.m]\nmax_output_tokens = 0", "llm.models.m: max_output_tokens must be a positive integer"),
        ("[llm.models.m]\nmax_output_tokens = true", "llm.models.m.max_output_tokens must be an integer"),
        ("[llm.models.m]\nmax_output_tokens = '8k'", "llm.models.m.max_output_tokens must be an integer"),
        (
            "[llm.models.m]\nsupports_reasoning_effort = false\ndefault_reasoning_effort = 'high'",
            "llm.models.m: default_reasoning_effort requires supports_reasoning_effort=True",
        ),
        ('[llm.models." "]\ntemperature = "omit"', "llm.models: model IDs must not be blank"),
    ],
)
def test_load_config_rejects_invalid_model_specs(tmp_path: Path, models: str, message: str) -> None:
    config_home = tmp_path / "config"
    path = config_home / "localmcp/localmcp.toml"
    path.parent.mkdir(parents=True)
    path.write_text(f"[llm]\nbackend = 'native'\n{models}\n")
    env = {"XDG_CONFIG_HOME": str(config_home)}

    with pytest.raises(ConfigError, match=re.escape(message)):
        _server().load_config(env=env, home=tmp_path)


def test_current_runtime_rejects_a_different_runtime_type(monkeypatch: pytest.MonkeyPatch) -> None:
    class OtherRuntime:
        async def start(self) -> None:
            return None

        async def close(self) -> None:
            return None

    monkeypatch.setattr(server_module, "_runtime", OtherRuntime())

    with pytest.raises(RuntimeUnavailableError, match="different application runtime"):
        current_runtime(_Runtime)


async def test_start_failure_closes_telemetry_and_leaves_no_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _write_document(tmp_path)

    async def fail_start(_self: _Runtime) -> None:
        raise RuntimeError("start failed")

    monkeypatch.setattr(_Runtime, "start", fail_start)
    with pytest.raises(RuntimeError, match="start failed"):
        await _server().start_runtime(env=env, home=tmp_path)

    assert server_module.telemetry().status.status == "disabled"
    with pytest.raises(RuntimeUnavailableError):
        current_runtime(_Runtime)


async def test_stop_propagates_runtime_close_failure_after_resetting_services(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = _server()
    await server.start_runtime(env=_write_document(tmp_path), home=tmp_path)
    runtime = current_runtime(_Runtime)

    async def fail_close() -> None:
        raise RuntimeError("close failed")

    monkeypatch.setattr(runtime, "close", fail_close)
    with pytest.raises(RuntimeError, match="close failed"):
        await server.stop_runtime()

    assert server_module.telemetry().status.status == "disabled"
    with pytest.raises(RuntimeUnavailableError):
        current_runtime(_Runtime)


async def test_gateway_backend_requires_declared_secret(tmp_path: Path) -> None:
    config_home = tmp_path / "config"
    path = config_home / "localmcp/localmcp.toml"
    path.parent.mkdir(parents=True)
    path.write_text(
        "[llm]\nbackend = 'openai_compatible'\nbase_url = 'https://example.test'\n[application]\ngreeting = 'hello'\n"
    )

    with pytest.raises(RuntimeError, match="secret declaration not found"):
        await _server().start_runtime(env={"XDG_CONFIG_HOME": str(config_home)}, home=tmp_path)


def test_main_delegates_to_server_run(monkeypatch: pytest.MonkeyPatch) -> None:
    server = _server()
    calls: list[None] = []
    monkeypatch.setattr(server, "run", lambda: calls.append(None))

    main(server)

    assert calls == [None]
