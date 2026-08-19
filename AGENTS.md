# Autonomous Agent Guide

This repository deploys a local/self-hosted Open WebUI stack. The SDK
interface is an OpenAI-compatible boundary with direct provider connections.

## Runtime Contract

- Configure one or more direct provider keys in the root `.env`.
- Models are fetched automatically from every configured provider via
  `GET /v1/models`; no model list is hardcoded.
- Chat requests and streams are served at `/v1/chat/completions`.
- Tools, tool calls, reasoning controls, JSON output, multimodal content, and
  local provider selection are forwarded for agent runtimes.
- This service does not run an agent loop or execute tools. The calling agent
  owns state, permissions, retries, and tool execution.

## Local Development

```bash
cp .env.example .env
# Set one or more direct provider keys without printing them
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
- Do not reintroduce hardcoded model lists or hosted routing dependencies.
- Preserve unrelated worktree changes and avoid destructive git commands.
- Run tests in mock mode; use real provider calls only when explicitly needed.
