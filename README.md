# localmcplib

`localmcplib` is a Python foundation for local MCP servers that run over
stdio. It packages the infrastructure that tends to be repeated across serious
local servers while leaving tools, prompts, application schemas, and product
behavior to each application.

The published distribution is `localmcplib`; the Python import package is
`localmcp`.

## Introduction

A local MCP server can start as a few tool functions, but production use adds a
surprising amount of non-domain work: stdout must remain clean for JSON-RPC,
stderr should carry useful startup diagnostics, credentials need safe lookup
boundaries, configuration must compose across servers, model connections need
deployment-independent routing, and long-running operations need lifecycle and
recovery support. Model-controlled local commands also need an actual security
boundary.

`localmcplib` owns these reusable mechanics while consuming servers continue
to own their schemas, tools, service clients, prompts, and authorization
policy.

## Quickstart

Install the package:

```console
uv add localmcplib
```

The following complete `hello.py` registers one tool and runs it as a FastMCP
stdio server. `STDIOServer` loads and validates localmcplib's configuration;
applications only need a parser when they add their own configuration fields.

```python
from fastmcp.tools import tool

import localmcp

NAME = "hello-mcp"


@tool
def hello(name: str = "world") -> str:
    return f"Hello, {name}!"


# STDIOServer owns an application runtime so durable tasks and other resources
# share its startup/shutdown lifecycle. Hello has no such work, so this runtime
# intentionally does nothing.
class NoopRuntime:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    async def start(self) -> None:
        pass

    async def close(self) -> None:
        pass


server = localmcp.STDIOServer(
    name=NAME,
    tools=[hello],
    runtime_factory=NoopRuntime,
)

if __name__ == "__main__":
    localmcp.main(server)
```

Create `~/.config/localmcp/localmcp.toml`:

```toml
schema_version = 1

[llm]
backend = "openai_compatible"
base_url = "https://llm.example.test/v1"

[secrets.llm_api_key]
env_vars = ["HELLO_LLM_API_KEY"]
```

Then run the server from an MCP client with `uv run python hello.py`. The
integrated server loads configuration and secrets, creates the model and
telemetry boundaries, owns the application runtime, registers the tool, and
serves it over stdio without writing logs to the protocol streams.

## Features

### Stdio MCP server composition

`localmcp.STDIOServer` assembles a FastMCP application from a server name, tool
list, and runtime factory. It owns startup and shutdown ordering, shared config
validation, secrets, model and telemetry wiring, per-server state paths,
file-only logging, and config error messages on stderr while keeping stdout
clear for the MCP protocol. Applications with additional configuration fields
may supply an optional typed `config_parser`; its result is passed to the
runtime factory and lifespan.

### Shared configuration and paths

One schema-versioned, XDG-aware `localmcp.toml` document provides shared
defaults. `[server.<name>]` recursively overlays those defaults for one logical
server. Applications choose the config file boundary and validate their own
effective schema; state, cache, and logs remain isolated per server.

See [docs/configuration.md](docs/configuration.md) for the document contract.

### Secrets

Applications declare logical secret names in configuration without storing
secret values there. Resolution checks declared environment variables first,
then the operating-system keyring. Global secrets use the shared `localmcp`
keyring service; server-private secrets use `localmcp:<server-name>`. Resolved
values are redacted from string representations and errors.

### LLM routing

Applications select logical model IDs independently of deployment routing. An
`openai_compatible` deployment can send models through a shared LiteLLM-style
endpoint, while a `native` deployment routes the same IDs to OpenAI or
Anthropic credentials. Gateway protocol dialect and native provider are
separate model properties, so switching deployment does not require changing
application model roles.

Structured output from tool-using agents varies by model and gateway.
LangChain's `ToolStrategy` forces `tool_choice`, which some models reject (for
example with extended thinking), and `ProviderStrategy` relies on a native
schema constraint that gateways may drop or that lets a model answer without
using its other tools. `localmcp.structured_output.SubmitResultMiddleware`
works across tool-capable models: it registers an ordinary, unforced
`submit_result` tool whose arguments are the schema, returns validation errors
and plain-text answers to the model for correction, and raises
`SubmitResultError` after `max_attempts` (default 5) failed attempts in a run.
Tool calls whose arguments cannot be parsed are answered with an error rather
than replayed to the provider verbatim:

```python
agent = create_agent(model, tools, middleware=[SubmitResultMiddleware(Answer)])
answer = (await agent.ainvoke({"messages": [("user", "...")]}))["structured_response"]
```

For `ProviderStrategy` agents, `FencedJSONOutputMiddleware` accepts a response
wrapped in one Markdown JSON fence, still validated against the same schema.

### Sandboxing

The sandbox API provides bounded command execution with explicit filesystem
roots, a clean environment, a private temporary directory per command, output
and time limits, and network access disabled by default. When a command returns
or times out, every process it started is killed, including background and
detached ones. The current implementation uses macOS Seatbelt; the portable
interface leaves room for a future Linux backend.

Nothing limits how much a command writes. Its temporary directory is writable
only when the profile has a read-write root, so a read-only profile gets no
space to fill. The directory is created in the caller's temporary directory,
which must be outside every root, so what a command writes there counts against
that volume rather than against any quota on a root's.

`denied_paths` hide files and directories inside the roots. Seatbelt denies the
real path a file is opened by, so a denied path must not go through a symlink,
and a denied file must have no other hard link, whose name would stay readable.
The sandbox refuses to be created, or to run a command, while a denied path
breaks either rule or cannot be checked, such as when it holds a directory the
caller cannot list.

A profile's `tools` are the complete list of commands its commands may run
besides shell builtins; anything else fails with "Operation not permitted",
including when a listed tool such as `awk`, `find` or `xargs` tries to run it,
and so do programs copied or written into a root. The default,
`INSPECTION_TOOLS`, covers reading, searching and comparing files. Extend it
with what an agent needs, such as
`SandboxProfile(..., tools=(*INSPECTION_TOOLS, *WRITE_TOOLS, "git"))`, or pass
`tools=()` to allow builtins only. `WRITE_TOOLS` adds `cp`, `mv`, `rm` and other
file-management commands for read-write roots. It does not decide what can
write: root access does, and the shell and `sed -i` can write without it.

On macOS each tool is also granted read-only access to the code it runs from,
and nothing else:

- A command provided by a Homebrew formula brings that formula's install and the
  installs of its runtime dependencies. The rest of the Homebrew prefix stays
  unreadable, including other formulae and its `etc`, `share` and `var`.
- A command provided by the system-selected Xcode or Command Line Tools install,
  such as `git`, `python3`, `clang` or `make`, brings that install and Xcode's
  license record, and its `/usr/bin` shim may run too.
- A base-system command needs nothing more. A `/usr/bin` developer shim is not
  one, so a developer tool is not installed unless the selected install
  provides it. Any other command is an error when the sandbox is created.

A tool may also run its own helper programs, which its install keeps in a
`libexec` directory named after it, such as git's `git-core`. A script runs its
`#!` interpreter only for itself, so listing `shasum` does not allow `perl -e`.
A script that starts with `#!/usr/bin/env` also needs the command `env` runs
listed.

Commands the sandbox can do without belong in `optional_tools`, such as
`SandboxProfile(..., tools=(*INSPECTION_TOOLS, "git"), optional_tools=("rg",))`.
Each one that is installed is granted in the same way. One that is missing or
broken grants nothing and does not fail the sandbox, so a command that runs it
fails with "command not found". The default description of the Bash tool from
`sandbox_tools` lists only the tools that can run.

Homebrew is found only at its default prefixes, and the developer install only
through `xcode-select`. Neither `HOMEBREW_PREFIX`, `DEVELOPER_DIR` nor the
caller's `PATH` is consulted, so the caller's environment cannot widen these
grants, and tools installed elsewhere, such as under `~/.cargo/bin`, are not
available. Configuration a tool reads from outside its own install, such as
Homebrew OpenSSL's `etc/openssl@3`, is not granted either. The exception is a
profile with `network`, which also grants the system TLS configuration and
trust store in `/etc/ssl`, without which the system `curl` cannot start.

### Durable workflows

The workflow package supplies LangGraph lifecycle management and a generic
SQLite operation catalog with ownership checks, recovery, worker leases,
heartbeats, cancellation, and durable tool-call budgets. Applications retain
responsibility for domain state, idempotency, and consequential side effects.

### Observability

Structured logs are written to files only so they cannot corrupt stdio
JSON-RPC. Optional Langfuse integration adds failure-isolated tracing and
FastMCP middleware. Model and tool payload attributes are suppressed by
default. Set `LOCALMCP_LANGFUSE_CAPTURE_PAYLOADS=true` to capture model prompts,
model responses, tool arguments, and tool results. This opt-in can send secrets,
repository content, and other sensitive data to Langfuse; enable it only when
the configured Langfuse project is an approved destination for those payloads.

## Status

The API is alpha and may change as additional local MCP servers adopt the
library.

## Development

Install the complete development environment and run the repository checks:

```console
make install
make ci
make build
```

Run `make help` to list the individual formatting, linting, type-checking,
testing, coverage, build, and cleanup targets.
