# Contributing

Contributions are welcome through pull requests against `main`.

## Before You Start

- Open an issue first for substantial behavior changes or compatibility work.
- Never include API keys, tokens, private prompts, provider responses, or user data.
- Keep provider and model discovery automatic. Do not add hardcoded model lists.
- Keep tool execution and agent state in the calling agent, not in `sdk-interface`.

## Development

```bash
cp .env.example .env
make setup
make test
make lint
docker compose config --quiet
```

Tests use mocked provider responses. Real provider calls are optional and must
never be required for CI.

## Pull Requests

A useful PR includes:

- A focused description of the behavior and compatibility impact.
- Tests for changed behavior and the exact commands used to run them.
- Documentation and environment example updates for new configuration.
- Security considerations, migration notes, or known provider limitations.

Keep commits focused and use the repository's conventional commit style where
practical. Maintainers may ask for changes before merge.
