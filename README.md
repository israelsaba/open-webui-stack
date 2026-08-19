# Open WebUI Stack

Self-hosted Open WebUI backed by a small OpenAI-compatible SDK interface. The
interface uses [OpenRouter](https://openrouter.ai) as the single gateway for
model discovery, provider routing, fallbacks, and chat completions.

## What This Provides

- Live model listings from OpenRouter. No provider or model allow-list is hardcoded.
- One gateway for Anthropic, Google, xAI, and other OpenRouter-supported models.
- OpenAI-compatible chat completions, including streaming.
- Agent-ready requests: tools, tool calls, parallel tool calls, JSON response formats, multimodal message content, reasoning effort, and OpenRouter provider preferences are passed through.
- Open WebUI configured to use the SDK interface over the internal Compose network.
- Local bearer-token authentication and production CORS safeguards.

OpenRouter model IDs use the `provider/model` form, for example
`anthropic/claude-sonnet-4.5` or `google/gemini-2.5-pro`.

## Quick Start

Requirements: Docker and Docker Compose.

```bash
cp .env.example .env
# Set SDK__OPENROUTER_API_KEY in .env
docker volume create open-webui
docker compose up -d --build
```

Open WebUI is available at <http://localhost:8090>. The SDK is internal to the
Compose network by default. To expose it for local tooling, add a port mapping
to `sdk-interface` or run it directly with `make run`.

## Configuration

The important variables are:

```dotenv
SDK__OPENROUTER_API_KEY=your-key
SDK__OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
SDK__OPENROUTER_SITE_URL=http://localhost:8090
SDK__OPENROUTER_SITE_NAME=Open WebUI Stack
SDK__MODELS_CACHE_TTL=60
SDK__API_KEYS=
SDK__CORS_ORIGINS=http://localhost:8090
OPENAI_API_BASE_URLS=http://sdk-interface:8060/v1
OPENAI_API_KEYS=
```

`SDK__API_KEYS` protects the SDK endpoint. When it is set, put the matching
token in `OPENAI_API_KEYS` for Open WebUI. Production mode requires explicit
CORS origins and at least one SDK token.

OpenRouter can route or fail over providers per request. For example, an agent
can send this OpenAI-compatible request:

```json
{
  "model": "anthropic/claude-sonnet-4.5",
  "messages": [{"role": "user", "content": "Inspect this repository."}],
  "tools": [{"type": "function", "function": {"name": "read_file", "parameters": {"type": "object"}}}],
  "provider": {"order": ["Anthropic", "Google"], "allow_fallbacks": true}
}
```

This keeps tool execution in the calling agent, which is the expected model for
Claude Code, OpenCode, Hermes, and similar systems. The SDK is a transport and
routing boundary, not an agent runtime with hidden state.

## Development

```bash
make setup
make test
make lint
make up
make logs-sdk
```

Tests use mocked OpenRouter responses and do not spend API credits. Use
`make test-real` only after configuring a real test environment.

Useful endpoints:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `GET /docs`

## Security

- Keep `.env` out of version control and never log API keys.
- Do not expose port `8060` publicly without authentication and TLS.
- Use OpenRouter spending limits and provider restrictions for production.
- Pin and review image and dependency updates before deploying them.
