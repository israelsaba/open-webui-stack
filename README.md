# Open WebUI Stack

Self-hosted Open WebUI backed by a small OpenAI-compatible SDK interface. The
interface connects directly to configured provider APIs through their shared
OpenAI-compatible transport.

## What This Provides

- Live model listings from every configured provider. No provider or model allow-list is hardcoded.
- One gateway for OpenAI, Anthropic, Google, and xAI models.
- OpenAI-compatible chat completions, including streaming.
- Agent-ready requests: tools, tool calls, parallel tool calls, JSON response formats, multimodal message content, reasoning effort, and local provider selection are passed through.
- Open WebUI configured to use the SDK interface over the internal Compose network.
- Local bearer-token authentication and production CORS safeguards.

Model IDs come directly from each configured provider. The `owned_by` field in
`/v1/models` identifies which provider supplied each model.

## Quick Start

Requirements: Docker and Docker Compose.

```bash
cp .env.example .env
# Set at least one direct provider key in .env
docker volume create open-webui
docker compose up -d --build
```

Open WebUI is available at <http://localhost:8090>. The SDK is internal to the
Compose network by default. To expose it for local tooling, add a port mapping
to `sdk-interface` or run it directly with `make run`.

## Configuration

The important variables are:

```dotenv
SDK__OPENAI_API_KEY=your-key
SDK__ANTHROPIC_API_KEY=your-key
SDK__GOOGLE_API_KEY=your-key
SDK__GROK_API_KEY=your-key
SDK__MODELS_CACHE_TTL=60
SDK__API_KEYS=
SDK__CORS_ORIGINS=http://localhost:8090
OPENAI_API_BASE_URLS=http://sdk-interface:8060/v1
OPENAI_API_KEYS=
```

`SDK__API_KEYS` protects the SDK endpoint. When it is set, put the matching
token in `OPENAI_API_KEYS` for Open WebUI. Production mode requires explicit
CORS origins and at least one SDK token.

An agent can select a local provider explicitly, or select a model discovered
from `/v1/models`:

```json
{
  "model": "claude-sonnet-4-6",
  "messages": [{"role": "user", "content": "Inspect this repository."}],
  "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}],
  "provider": "anthropic"
}
```

This keeps tool execution in the calling agent, which is the expected model for
Claude Code, OpenCode, Hermes, and similar systems. The SDK is a direct transport
and model-discovery boundary, not an agent runtime with hidden state.

## Development

```bash
make setup
make test
make lint
make up
make logs-sdk
```

Tests use mocked provider responses and do not spend API credits. Use
`make test-real` only after configuring a real test environment.

Useful endpoints:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `GET /docs`

## Security

- Keep `.env` out of version control and never log API keys.
- Do not expose port `8060` publicly without authentication and TLS.
- Use provider spending limits and explicit provider configuration for production.

## Open-Source Project

This project is maintained in the open. Contributions are welcome through pull
requests. Before opening one, read [CONTRIBUTING.md](CONTRIBUTING.md) and run
the documented checks. Report reproducible bugs, installation problems,
compatibility reports, and focused proposals through GitHub Issues. Do not post
secrets, private prompts, provider responses, or vulnerability details publicly;
follow [SECURITY.md](SECURITY.md) for security reports.

Stable versions are published as immutable `vMAJOR.MINOR.PATCH` tags with
release notes on GitHub. The repository's provider APIs and OpenAI-compatible
request behavior are external contracts; their official documentation is the
source of truth when behavior changes.

## Sources

- [OpenAI API reference](https://platform.openai.com/docs/api-reference)
- [Anthropic OpenAI compatibility](https://docs.anthropic.com/en/api/openai-sdk)
- [Google Gemini OpenAI compatibility](https://ai.google.dev/gemini-api/docs/openai)
- [xAI API documentation](https://docs.x.ai/docs)
- Pin and review image and dependency updates before deploying them.
