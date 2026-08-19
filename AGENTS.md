# Autonomous Agent Guide

This repository deploys a local/self-hosted Open WebUI stack. The SDK
interface is an OpenAI-compatible boundary backed by OpenRouter.

## Runtime Contract

- Configure `SDK__OPENROUTER_API_KEY` in the root `.env`.
- Models are fetched automatically from `GET /v1/models` and use OpenRouter
  IDs such as `anthropic/claude-sonnet-4.5`.
- Chat requests and streams are served at `/v1/chat/completions`.
- Tools, tool calls, reasoning controls, JSON output, multimodal content, and
  OpenRouter provider preferences are forwarded for agent runtimes.
- This service does not run an agent loop or execute tools. The calling agent
  owns state, permissions, retries, and tool execution.

## Local Development

```bash
cp .env.example .env
# Set SDK__OPENROUTER_API_KEY without printing it
make setup
make test
make lint
docker compose up -d --build
```

Open WebUI runs at `http://localhost:8090`. The SDK remains internal to the
Compose network unless explicitly exposed. In production, set
`SDK__ENVIRONMENT=PROD`, configure explicit `SDK__CORS_ORIGINS`, and provide
`SDK__API_KEYS`.

## Safety

- Never commit `.env`, API keys, tokens, or provider responses.
- Do not expose port `8060` publicly without authentication and TLS.
- Do not reintroduce hardcoded model lists or direct provider clients.
- Preserve unrelated worktree changes and avoid destructive git commands.
- Run tests in mock mode; use real provider calls only when explicitly needed.
