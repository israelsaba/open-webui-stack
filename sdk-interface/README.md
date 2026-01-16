# Anthropic to OpenAI API Bridge

A dockerized FastAPI application that exposes Anthropic's Claude models through an OpenAI-compatible API. This allows you to use Claude models with any application that supports the OpenAI API format, including Open WebUI.

## Features

- **OpenAI-Compatible API**: Implements `/v1/models` and `/v1/chat/completions` endpoints
- **Streaming Support**: Full support for streaming responses
- **Localhost Only**: Configured to only be accessible from localhost for security
- **Strongly Typed**: Built with Python 3.12 using modern type hints (no `Any` types)
- **Docker Ready**: Includes Dockerfile and docker-compose for easy deployment
- **Health Checks**: Built-in health monitoring

## Supported Models

### Latest Models (Claude 4.5)
- claude-sonnet-4-5-20250929 (Claude Sonnet 4.5)
- claude-haiku-4-5-20251001 (Claude Haiku 4.5)
- claude-opus-4-5-20251101 (Claude Opus 4.5)

### Legacy Models (Claude 3.5 & Claude 3)
- claude-3-5-sonnet-20241022
- claude-3-5-haiku-20241022
- claude-3-opus-20240229
- claude-3-sonnet-20240229
- claude-3-haiku-20240307

## Prerequisites

- Docker and Docker Compose
- Anthropic API key

## Setup

1. **Clone or navigate to the repository**

2. **Create your `.env` file**:
   ```bash
   cp .env.example .env
   ```

3. **Edit `.env` and add your Anthropic API key**:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   ```

4. **Build and run with Docker Compose**:
   ```bash
   docker-compose up -d
   ```

5. **Verify the service is running**:
   ```bash
   curl http://localhost:8000/health
   ```

## API Endpoints

### List Models
```bash
curl http://localhost:8000/v1/models
```

### Chat Completion (Non-streaming)
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-5-20250929",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

### Chat Completion (Streaming)
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-sonnet-4-5-20250929",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "stream": true
  }'
```

## Connecting to Open WebUI

1. Open Open WebUI in your browser
2. Go to **Admin Settings** → **Connections** → **OpenAI**
3. Click **Add Connection**
4. Select the **Standard / Compatible** tab
5. Configure:
   - **API URL**: `http://localhost:8000/v1`
   - **API Key**: Leave blank (not required for localhost)
6. Click **Save**

If running Open WebUI in Docker and this service on your host, use:
- **API URL**: `http://host.docker.internal:8000/v1`

## Development

### Running locally without Docker:

1. **Create a virtual environment**:
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Create `.env` file** with your API key

4. **Run the application**:
   ```bash
   python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
   ```

## Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── config.py           # Configuration with Pydantic Settings
│   ├── models.py           # Request/Response models
│   ├── anthropic_client.py # Anthropic API client
│   └── main.py            # FastAPI application
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Security Notes

- The service is configured to only bind to `127.0.0.1` (localhost) in docker-compose
- Never commit your `.env` file with real API keys
- The Anthropic API key is stored as a Pydantic `SecretStr` for additional security

## Logs

View logs:
```bash
docker-compose logs -f
```

## Stopping the Service

```bash
docker-compose down
```

## Troubleshooting

**Connection refused**: Ensure the service is running with `docker-compose ps`

**Model not found**: Check that you're using one of the supported model IDs

**API key error**: Verify your `.env` file contains a valid `ANTHROPIC_API_KEY`

## License

MIT
