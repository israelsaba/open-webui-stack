# Open WebUI + Multi-Provider AI Integration Stack

A complete Docker Compose stack that runs **Open WebUI** with **Anthropic Claude** and **xAI Grok** models via an OpenAI-compatible API bridge. This setup provides a self-hosted, feature-rich chat interface powered by advanced AI capabilities from multiple providers.

## 🚀 Features

### Core Functionality

- **OpenAI-Compatible API Bridge**: FastAPI-based service that translates OpenAI API requests to Anthropic and xAI API formats
- **Open WebUI Interface**: Modern, feature-rich web UI for interacting with AI models
- **Multi-Provider Support**: 
  - **Anthropic Claude**: Access to the latest Claude models, dynamically fetched from Anthropic's API
  - **xAI Grok**: Access to the latest Grok models, dynamically fetched from xAI's API
  - Models are automatically discovered and updated without code changes
- **Streaming Support**: Real-time streaming responses for a natural chat experience
- **Bearer Token Authentication**: Secure API access with custom token management
- **Docker-based Deployment**: Easy setup and management with Docker Compose
- **Auto-updates**: Integrated Watchtower for automatic container updates

### Open WebUI Capabilities

With this stack, you get all the powerful features of Open WebUI integrated with multiple AI providers:

- **RAG (Retrieval Augmented Generation)**: Upload documents and chat with your data using AI models
  - Uses OpenAI-compatible embedding engine for document vectorization
- **Speech-to-Text**: Audio transcription powered by OpenAI's STT engine
- **Persistent Storage**: Conversation history and uploaded documents are preserved
- **Multi-user Support**: Bearer token authentication allows multiple users with separate access
- **Health Monitoring**: Built-in health checks and monitoring for all services

## 📋 Prerequisites

- Docker and Docker Compose installed
- At least one API key from supported providers:
  - Anthropic API key ([get one here](https://console.anthropic.com/))
  - xAI API key ([get one here](https://console.x.ai/))

## 🔧 Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd open-webui-stack
   ```

2. **Create the required `.env` files**

   Main `.env` file (root directory):
   ```bash
   # Add any Open WebUI specific configurations here
   ```

   SDK interface `.env` file (`sdk-interface/.env`):
   ```bash
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   XAI_API_KEY=your_xai_api_key_here
   API_KEYS=username1:op_wui_token1;username2:op_wui_token2
   LOG_LEVEL=info
   ```
   
   Note: You can configure one or both API keys depending on which providers you want to use.

3. **Generate API tokens** (optional, for authentication)
   ```bash
   python sdk-interface/scripts/generate_token.py
   ```

4. **Create the required Docker volume**
   ```bash
   docker volume create open-webui
   ```

5. **Start the stack**
   ```bash
   docker-compose up -d
   ```

6. **Access Open WebUI**
   - Open your browser and navigate to: `http://localhost:8090`
   - Configure the API endpoint in Open WebUI to point to the SDK interface service

## 🏗️ Architecture

The stack consists of three services:

### 1. SDK Interface (`sdk-interface`)
- **Purpose**: OpenAI-compatible API bridge for multiple AI providers (Anthropic Claude, xAI Grok)
- **Technology**: FastAPI, Python 3.12
- **Port**: Internal only (accessed via Docker network)
- **Key Features**:
  - Converts OpenAI chat completion format to provider-specific API formats
  - Dynamically fetches available models from each provider
  - Handles both streaming and non-streaming responses
  - Bearer token authentication with `op_wui_` prefix tokens
  - Health check endpoint at `/health`
  - Models listing at `/v1/models`
  - Chat completions at `/v1/chat/completions`

### 2. Open WebUI (`open-webui`)
- **Purpose**: Modern web interface for AI chat
- **Image**: `ghcr.io/open-webui/open-webui:main`
- **Port**: `8090` (mapped to internal `8080`)
- **Key Features**:
  - Rich markdown rendering
  - File upload and document processing
  - Conversation history
  - RAG with OpenAI-compatible embeddings
  - Speech-to-text transcription

### 3. Watchtower (`watchtower`)
- **Purpose**: Automatic container updates
- **Schedule**: Daily at 2 AM
- **Target**: Updates the `open-webui` container automatically

## 🔐 Authentication

The SDK interface supports bearer token authentication using tokens with the `op_wui_` prefix:

- Format: `username:op_wui_<token>`
- Multiple tokens separated by semicolons
- Tokens are validated on each API request
- Health check and root endpoints bypass authentication

Generate tokens using:
```bash
python sdk-interface/scripts/generate_token.py
```

## 📡 API Endpoints

### SDK Interface

- `GET /` - API information
- `GET /v1/models` - List all available models from configured providers (dynamically fetched)
- `POST /v1/chat/completions` - Create chat completion (streaming and non-streaming)
- `GET /health` - Health check

### Request Example

```python
from openai import OpenAI

client = OpenAI(
    api_key="op_wui_your_token_here",
    base_url="http://localhost:8060/v1"
)

response = client.chat.completions.create(
    model="claude-sonnet-4-20250514",  # or any available model from /v1/models
    messages=[
        {"role": "user", "content": "Hello!"}
    ],
    stream=True
)
```

## 🛠️ Configuration

### Environment Variables

**SDK Interface:**
- `ANTHROPIC_API_KEY` - Your Anthropic API key (optional, required for Claude models)
- `XAI_API_KEY` - Your xAI API key (optional, required for Grok models)
- `API_KEYS` - Bearer tokens in format `username:token;username:token`
- `LOG_LEVEL` - Logging level (default: `info`)
- `HOST` - Bind host (default: `0.0.0.0`)
- `PORT` - Service port (default: `8060`)

**Open WebUI:**
- `AUDIO_STT_ENGINE` - Speech-to-text engine (set to `openai`)
- `RAG_EMBEDDING_ENGINE` - Embedding engine for RAG (set to `openai`)

## 📦 Network & Storage

- **Network**: Custom bridge network `open-webui-net` for service communication
- **Volume**: External volume `open-webui` for persistent data storage
- **Health Checks**: SDK interface includes health monitoring with automatic restarts

## 🔄 Updates

The stack includes Watchtower for automatic updates:
- Runs daily at 2 AM
- Updates only the `open-webui` container
- Automatically cleans up old images
- Does not update stopped containers

## 📝 Development

The SDK interface is mounted as a volume for development:
```yaml
volumes:
  - ./sdk-interface:/sdk-interface/app
```

Changes to Python files will require a container restart to take effect.

## ⚠️ Disclaimer

This is an **unofficial** community project and is **not affiliated with or endorsed by:**
- Open WebUI Inc. or its contributors
- Anthropic PBC
- xAI Corp.

"Open WebUI" is the name/branding of the upstream project. Please refer to the official repositories for licensing and branding requirements.

## 🔗 Links

- [Open WebUI Official Repository](https://github.com/open-webui/open-webui)
- [Anthropic](https://www.anthropic.com/)
- [Anthropic API Documentation](https://docs.anthropic.com/)
- [xAI](https://x.ai/)
- [xAI API Documentation](https://docs.x.ai/)

## 📄 License

Please refer to the individual component licenses:
- Open WebUI: See upstream repository
- SDK Interface: See LICENSE file in this repository
