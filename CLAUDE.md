# Agent Notes

## Architecture

`sdk-interface` is a deliberately thin OpenAI-compatible transport boundary.
It uses one `ProviderGateway` with direct OpenAI-compatible connections to each
configured provider; provider-specific SDKs must not be added to the request
path.

The service does not execute tools or maintain agent state. Callers such as
OpenCode, Claude Code, Hermes, and Open WebUI own the agent loop and send tool
definitions, tool calls, tool results, reasoning controls, and provider routing
preferences through `/v1/chat/completions`.

## Important Files

- `sdk-interface/app/provider_gateway.py`: direct provider discovery, routing, request forwarding, and SSE conversion
- `sdk-interface/app/models.py`: OpenAI-compatible and agentic request/response schemas
- `sdk-interface/app/main.py`: HTTP boundary, auth, CORS, and health endpoint
- `sdk-interface/app/config.py`: `SDK__` environment configuration
- `sdk-interface/tests/`: network-free gateway tests

## Development Rules

- Never hardcode a provider model list. `/v1/models` comes from configured providers.
- Preserve unknown agent fields where the OpenAI-compatible providers support them.
- Keep tool execution outside this service.
- Do not log credentials, prompts, tool arguments, or full provider responses by default.
- Run `make test` and `make lint` before submitting changes.
- Use direct provider keys and base URLs; OpenRouter is not required by the runtime contract.
