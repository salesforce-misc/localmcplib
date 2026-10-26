# Configuration contract

The default document is `$XDG_CONFIG_HOME/localmcp/localmcp.toml`, falling
back to `~/.config/localmcp/localmcp.toml`. It contains no resolved secret
values. Relative XDG directory values are ignored as required by the XDG Base
Directory specification.

`schema_version` is currently `1`. Every root value other than `server` is a
global default. Selecting MCP server `name` recursively merges `[server.name]`
over those defaults. Tables merge recursively; arrays and scalar values replace
the global value. The former root `tool` spelling is rejected; use `server`.

```toml
schema_version = 1

[observability]
log_level = "WARNING"

[llm]
backend = "openai_compatible"
base_url = "https://gateway.example/v1"

[server.server-a.llm]
backend = "openai_compatible"
```

The effective `server-a` view retains the global `llm.base_url` and inherits
`observability` unchanged. Unknown keys are retained. `STDIOServer` validates
the library-owned `llm`, `observability`, and `secrets` fields. When an
application supplies `config_parser`, it receives the effective `ServerConfig`
with those fields and `schema_version` removed, leaving only application-owned
keys to validate. Without a parser, that filtered `ServerConfig` is passed to
the runtime directly.

## LLM backend selection

Backend selection is explicit and never inferred from the credentials present.
`STDIOServer` validates the effective `[llm]` table and constructs its
`LLMBackendConfig`. The model IDs assigned to application-owned roles do not
change when deployment routing changes.

A gateway deployment uses one endpoint and logical secret:

```toml
[llm]
backend = "openai_compatible"
base_url = "https://gateway.example/v1"

[secrets.llm_api_key]
env_vars = ["GATEWAY_API_KEY"]
```

A native-provider deployment can replace those lines with native routing and
provider credentials:

```toml
[llm]
backend = "native"

[secrets.openai_api_key]
env_vars = ["OPENAI_API_KEY"]

[secrets.anthropic_api_key]
env_vars = ["ANTHROPIC_API_KEY"]
```

`GatewayModelFactory` selects `ChatOpenAI` or `ChatAnthropic` from the model's
independent gateway dialect while sharing the configured endpoint and key. This
allows, for example, an Anthropic-native model to use an OpenAI-dialect LiteLLM
endpoint. `NativeModelFactory`
uses `ModelSpec.native_provider` and never receives a gateway URL. A model with
no native provider is gateway-only. `ConfiguredModelFactory` provides the
stable `ModelFactory.create(model_id, ...)` boundary over both modes. Models
and transports are created lazily for each call and are closed by the returned
async context manager. A custom `http_client_factory` replaces the library's
transport, so its clients enforce the response timeout described below only
when built with `localmcp.llm.response_deadline_http_client`.

Every model streams its responses, and LangChain reassembles each into one
message with its stop reason, tool calls, and usage when the provider reports
it, so callers see an ordinary response. A long generation therefore never
waits on one idle read, and the Anthropic SDK does not refuse a large output
limit for lack of streaming. Without a per-call or factory-wide
`max_output_tokens`, a model may generate up to its maximum: the catalogued
`max_output_tokens` when declared, otherwise the provider's own limit. A
factory-wide limit never raises a model above its catalogued maximum; a
per-call limit is sent as given. Anthropic models, which require an explicit
limit, take it from `ChatAnthropic`'s model profile, or 64,000 tokens for a
model it does not know; catalogue `max_output_tokens` for an Anthropic-dialect
model that accepts fewer.

A streamed OpenAI-dialect response reports token usage only when its request
sets `stream_options.include_usage`, which localmcp sends by default. For an
`openai_compatible` gateway that rejects that option, set
`[llm].stream_usage = false`; its OpenAI-dialect responses then carry no usage
metadata. Native OpenAI and Anthropic-dialect models are unaffected.

A streamed response is never idle for long: providers send keep-alive events
while a model thinks, so the SDKs' read timeout, which bounds only the wait
between bytes, cannot end a stalled one. `[llm].response_timeout` therefore
bounds each response as a whole, from sending its request to its last streamed
byte, in seconds (default 600; it must be finite). The deadline applies to
each HTTP attempt. When it passes before the response begins, the provider SDK
retries the request itself (twice by default) and then raises its own timeout
error; when it passes mid-stream, the response fails with
`httpx2.ReadTimeout`. `localmcp.retry` treats both as transient faults. The
deadline cannot distinguish a stalled response from a long one, and a retry
repeats the same request, so a model whose uncapped responses routinely
outlast it fails every attempt: raise the timeout or set an output limit for
such models.

```toml
[llm]
response_timeout = 900
```

### Model catalog

The built-in model catalog is advisory. Any model ID is accepted and sent to
the provider, which remains the authority on whether it exists; an unknown or
misspelled ID fails with the provider's own error. IDs outside the catalog use
inferred capabilities and log one warning:

- temperature is omitted;
- `claude-*` IDs use the Anthropic dialect and native provider;
- `gpt-*` and `o<digit>*` IDs use the OpenAI dialect and native provider;
- all other IDs use the OpenAI dialect and are gateway-only; and
- a caller-requested reasoning effort is passed through.

A reasoning effort reaches OpenAI-dialect models as `reasoning_effort` and
Anthropic models as `output_config.effort`; both accept the same level names,
and the provider rejects a level the model lacks. On Anthropic models an effort
also enables adaptive thinking, which rejects a temperature. Whenever such a
model receives an effort, whether requested or its `default_reasoning_effort`,
a catalogued zero temperature is therefore omitted; this includes an Anthropic
model that a gateway serves through the OpenAI dialect.

`[llm.models."<model-id>"]` declares or corrects a model without a library
release. Each entry updates only the fields it names, starting from the
catalogued spec or, for an unlisted ID, the inferred one:

```toml
[llm.models."claude-opus-5-5"]
temperature = "omit"

[llm.models."house-model"]
dialect = "anthropic"
```

Supported fields are `temperature` (`"zero"` or `"omit"`),
`supports_reasoning_effort`, `default_reasoning_effort`, `native_provider`,
`dialect` (`"anthropic"` or `"openai"`), and `max_output_tokens` (a positive
integer: the most the model may generate in one response when a caller sets no
limit).

Server authors may select a different document at the loading boundary. This
defines which configuration and secret declarations are visible to that
runtime. Secret keyring boundaries are derived from logical server names, not
filesystem paths.

```python
from pathlib import Path

from localmcp.config import ConfigDocument

document = ConfigDocument.load(config_path=Path("/etc/example/localmcp.toml"))
values = document.for_server("server-a").values
```

Omitting `config_path` retains the XDG default described above. The selected
path is available as `document.path`.

## Secret declarations

`[secrets.<logical-name>]` declares a localmcp-wide secret. A matching
`[server.<name>.secrets.<logical-name>]` table overrides lookup metadata while
retaining unspecified global fields. A differently named declaration is
private to that server. Set `disabled = true` in a server declaration to remove
an inherited secret from that server's catalog.

Supported fields are:

- `name`: keyring account, defaulting to the logical name;
- `env_vars`: ordered environment aliases.

Global declarations use keyring service `localmcp`. A declaration introduced
under `[server.<server-name>.secrets]` uses `localmcp:<server-name>`; overriding
a global declaration does not change its global keyring location. Environment
aliases are always checked first; keyring is used only when no environment
alias provides a value.

## State paths

A selected document may configure one or more servers, while state remains
isolated:

- state: `$XDG_STATE_HOME/localmcp/<server-name>/`;
- cache: `$XDG_CACHE_HOME/localmcp/<server-name>/`; and
- log: `$XDG_STATE_HOME/localmcp/<server-name>/logs/<server-name>.log`.
