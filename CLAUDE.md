# 🤖 Instructions for AI Assistants (Claude, GPT, and other LLMs)

> This document provides guidance for AI assistants (like Claude, GPT-4, Gemini, etc.) that are helping humans work on this codebase.

## 📋 Project Overview

**Project Name:** AI Research Hub / Open WebUI Stack  
**Primary Purpose:** OpenAI-compatible API bridge for Google Deep Research, Anthropic Claude, Google Gemini, and xAI Grok models, integrated with Open WebUI  
**Tech Stack:** Python 3.12, FastAPI, Docker, SQLite, pytest  
**Repository Structure:**
```
open-webui-stack/
├── sdk-interface/          # Main API bridge service
│   ├── app/               # Application code
│   │   ├── main.py        # FastAPI application entry
│   │   ├── *_client.py    # Provider-specific clients
│   │   ├── models.py      # Pydantic models
│   │   └── config.py      # Configuration management
│   ├── tests/             # Test suite
│   ├── migrations/        # Database migrations (yoyo)
│   ├── requirements*.txt  # Python dependencies
│   └── Makefile           # Development commands
├── docker-compose.yml     # Orchestration config
├── .github/workflows/     # CI/CD pipelines
└── README.md              # User documentation
```

## 🎯 Key Responsibilities of this Project

1. **OpenAI API Compatibility**: Translate OpenAI-format requests to provider-specific APIs
2. **Deep Research Persistence**: Store and resume Google Deep Research sessions via SQLite
3. **Multi-Provider Support**: Unified interface for Anthropic, Google, and xAI
4. **Streaming Responses**: Handle SSE (Server-Sent Events) with reasoning content
5. **Authentication**: Bearer token authentication with `op_wui_` prefix
6. **Integration**: Seamless connection with Open WebUI frontend

## 🔍 Critical Questions to Ask the User

When a human asks for help, use these questions to quickly understand context:

### For Bug Reports
1. "Can you share the error message or stack trace?"
2. "Which provider/model were you using (Claude, Gemini, Grok, Deep Research)?"
3. "Can you share the relevant logs from `docker compose logs sdk-interface`?"
4. "Is this happening consistently or intermittently?"
5. "What was the input/prompt that caused the issue?"

### For Feature Requests
1. "Which component does this affect (sdk-interface, deployment, testing)?"
2. "Do you need this for all providers or specific ones?"
3. "Is there a specific use case or user story driving this?"
4. "Are there any performance or rate limit considerations?"

### For Deployment Issues
1. "What's your deployment environment (AWS, local, Docker Desktop)?"
2. "Can you share your docker-compose.yml and .env structure (without secrets)?"
3. "What's the output of `docker compose ps` and `docker compose logs`?"
4. "Have you completed the database migrations?"

### For Testing/CI Issues
1. "Are tests failing locally or only in CI?"
2. "Which specific test(s) are failing?"
3. "Do you have the required API keys set in environment?"
4. "What's the output of `make test-cov`?"

## 🏗️ Architecture Deep Dive

### Request Flow

```
User → Open WebUI → SDK Interface → Provider API
                         ↓
                    SQLite DB (for Deep Research sessions)
```

### Key Components

**1. Provider Clients** (`app/*_client.py`):
- `anthropic_client.py`: Claude models with extended thinking support
- `gemini_client.py`: Gemini models + Deep Research with session persistence
- `grok_client.py`: xAI Grok models
- `connection_client.py`: Abstract base class for all clients

**2. Main Application** (`app/main.py`):
- FastAPI app with `/v1/models` and `/v1/chat/completions` endpoints
- Dependency injection for client selection
- Bearer token middleware

**3. Models** (`app/models.py`):
- Pydantic models for OpenAI-compatible request/response
- Custom models for reasoning content and deep research

**4. Deep Research Persistence**:
- Uses MD5 hash of message history to identify unique research sessions
- Stores `interaction_id` in SQLite `research_hashes` table
- Auto-resumes sessions without consuming API quota

### Testing Architecture

**Location:** `sdk-interface/tests/`

**Structure:**
- `conftest.py`: Pytest fixtures and configuration
- `test_health.py`: Basic health check tests
- `test_models.py`: Model listing tests
- `test_completions.py`: Completion tests for all providers (streaming and non-streaming)

**Running Tests:**
```bash
cd sdk-interface
make setup          # One-time: install dependencies
make test           # Run tests (skip slow tests)
make test-cov       # Run with coverage report
```

**CI/CD:** GitHub Actions workflow at `.github/workflows/test.yml`

## 💻 Development Workflow

### Setting Up Local Environment

```bash
cd sdk-interface

# Option 1: Using Make (recommended)
make setup          # Install all dependencies in .venv
source .venv/bin/activate

# Option 2: Manual setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-test.txt
```

### Common Development Tasks

| Task | Command | Purpose |
|------|---------|---------|
| Run locally | `make run` | Start dev server with hot reload |
| Run tests | `make test` | Run test suite (CI equivalent) |
| Coverage report | `make test-cov` | Generate HTML coverage report |
| Lint code | `make lint` | Run ruff linter |
| Format code | `make format` | Auto-format with ruff |
| Clean up | `make clean` | Remove .venv, cache, artifacts |

### Making Code Changes

**Before starting:**
1. Create a new branch: `git checkout -b feature/your-feature-name`
2. Ensure you understand the component you're modifying
3. Check existing tests for similar functionality

**While coding:**
1. Write tests first (TDD approach recommended)
2. Follow existing code patterns (especially in `*_client.py` files)
3. Update docstrings for any new functions
4. Run `make lint` and `make format` frequently

**Before committing:**
1. Run `make test-cov` to ensure tests pass and coverage doesn't drop
2. Check that your changes work with actual API calls (if possible)
3. Update CLAUDE.md or AGENTS.md if you change architecture

**Commit message format:**
```
type(scope): brief description

- Detail 1
- Detail 2

Closes #issue-number
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`

### Code Style Guidelines

1. **Type Hints**: Always use type hints for function parameters and return values
2. **Docstrings**: Use Google-style docstrings for public functions
3. **Error Handling**: Catch specific exceptions, log errors with context
4. **Async/Await**: Use async for I/O operations (API calls, database)
5. **Logging**: Use structured logging with appropriate levels
6. **Configuration**: Never hardcode values - use environment variables

**Example:**
```python
async def create_completion(
    self,
    request: ChatCompletionRequest
) -> ChatCompletionResponse:
    """
    Create a non-streaming chat completion.
    
    Args:
        request: OpenAI-compatible completion request
        
    Returns:
        OpenAI-compatible completion response
        
    Raises:
        HTTPException: If provider API returns an error
    """
    logger.info(f"Creating completion for model: {request.model}")
    try:
        # Implementation
        pass
    except ProviderAPIError as e:
        logger.error(f"Provider API error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
```

## 🧪 Testing Guidelines

### Test Coverage Expectations

- **Target**: 70%+ line coverage
- **Critical paths**: 90%+ coverage for `*_client.py` files
- **New features**: Must include tests

### Writing Tests

**Use fixtures from `conftest.py`:**
```python
@pytest.mark.asyncio
async def test_example(
    http_client: httpx.AsyncClient,
    test_models: dict[str, str],
    anthropic_api_key: str,
    skip_if_no_api_key
):
    skip_if_no_api_key("anthropic", anthropic_api_key)
    # Your test here
```

**Test organization:**
- `test_health.py`: Basic endpoint tests
- `test_models.py`: Model listing and validation
- `test_completions.py`: Actual API interactions (may be skipped if no API keys)

**Markers:**
- `@pytest.mark.asyncio`: For async tests
- `@pytest.mark.slow`: For tests >5 seconds (skipped in CI)
- `@pytest.mark.integration`: For tests requiring external APIs

### CI/CD Pipeline

**GitHub Actions** (`.github/workflows/test.yml`):
1. Sets up Python 3.12
2. Installs dependencies from `requirements-test.txt`
3. Runs linter (ruff)
4. Runs tests with coverage (skips `@pytest.mark.slow`)
5. Uploads coverage to Codecov
6. Comments coverage on PRs

**GitHub Secrets (Optional):**
- `CODECOV_TOKEN`: For coverage reporting (optional)

**No API keys in GitHub!** API keys should only be set as local environment variables for local testing. CI/CD runs without API keys and auto-skips provider integration tests.

**For local testing with real APIs:**
```bash
export GOOGLE_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GROK_API_KEY="your-key"
make test-cov
```

## 🐛 Debugging Tips

### Common Issues

**1. "Attempt to decode JSON with unexpected mimetype: text/plain"**
- **Cause**: Provider API returning text/plain instead of JSON
- **Check**: Response headers in `*_client.py`
- **Fix**: Ensure `Content-Type: application/json` for non-streaming, `text/event-stream` for streaming

**2. Deep Research not resuming sessions**
- **Check**: `research_hashes` table in SQLite (`data/db.sqlite3`)
- **Verify**: MD5 hash generation in `gemini_client.py`
- **Debug**: Enable debug logging: `LOG_LEVEL=debug` in .env

**3. Tests failing with "API key not configured"**
- **Cause**: Missing environment variables
- **Fix**: Set `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `GROK_API_KEY`
- **Skip**: Tests auto-skip if API keys not available

**4. Connection refused errors**
- **Check**: Is docker service running? `docker compose ps`
- **Verify**: Port mapping in docker-compose.yml
- **Network**: Ensure `open-webui-net` bridge exists

### Logging

**Enable debug logging:**
```bash
# In sdk-interface/.env
LOG_LEVEL=debug
```

**View logs:**
```bash
docker compose logs -f sdk-interface  # Follow logs
docker compose logs --tail=100 sdk-interface  # Last 100 lines
```

**Log locations:**
- Container logs: `docker compose logs`
- Test output: `pytest -v --log-cli-level=DEBUG`

## 📝 Documentation Standards

### When to Update Documentation

- **README.md**: User-facing changes, deployment steps, new features
- **CLAUDE.md** (this file): Architecture changes, new development patterns
- **AGENTS.md**: Autonomous deployment procedures, automation scripts
- **Code docstrings**: New functions, changed behavior
- **CHANGELOG.md**: Version releases, breaking changes

### Documentation Style

- **Be concise**: Assume technical audience
- **Use examples**: Show don't tell
- **Link to code**: Reference specific files and line numbers
- **Keep updated**: Documentation rots quickly - review quarterly

## 🚀 Deployment Notes

### Local Development

```bash
cd sdk-interface
make setup && make run
# Access at http://localhost:8060
```

### Docker Compose (Production-like)

```bash
docker compose up -d
# SDK Interface: http://localhost:8060 (internal)
# Open WebUI: http://localhost:8090
```

### Environment Variables

**Required:**
- `GOOGLE_API_KEY`: For Deep Research and Gemini
- At least one of: `ANTHROPIC_API_KEY`, `GROK_API_KEY`

**Optional:**
- `API_KEYS`: Format `username:token;username2:token2`
- `LOG_LEVEL`: `debug|info|warning|error`
- `INTERACTION_POLL_INTERVAL`: Deep Research polling (default: 30)

## 🤝 Best Practices for AI Assistants

### Do:
✅ Ask clarifying questions before making changes  
✅ Run tests after code modifications  
✅ Provide context for why a change is needed  
✅ Suggest multiple approaches when applicable  
✅ Reference existing code patterns  
✅ Update tests when changing functionality  
✅ Check logs for error context  

### Don't:
❌ Make breaking changes without discussing  
❌ Commit API keys or secrets  
❌ Skip running tests  
❌ Assume user's environment setup  
❌ Modify docker-compose.yml without asking  
❌ Change core architecture without consensus  
❌ Ignore failing tests or linter errors  

### Communication Style

- Be direct and technical
- Use code blocks liberally
- Provide commands the user can copy-paste
- Explain the "why" not just the "what"
- Anticipate follow-up questions
- Offer to dive deeper when appropriate

## 🔗 Quick Reference Links

- **Main README**: [README.md](./README.md)
- **Agent Instructions**: [AGENTS.md](./AGENTS.md)
- **API Documentation**: [sdk-interface/README.md](./sdk-interface/README.md)
- **GitHub Workflow**: [.github/workflows/test.yml](./.github/workflows/test.yml)
- **Test Suite**: [sdk-interface/tests/](./sdk-interface/tests/)
- **Docker Compose**: [docker-compose.yml](./docker-compose.yml)

---

**Remember**: You're here to help humans build and maintain this project. When in doubt, ask questions. Good luck! 🚀
