# Google Deep Research and Open WebUI Integration Stack

A complete Docker Compose stack featuring **Google's Deep Research** with persistent session resumption, integrated with **Open WebUI** for a powerful research-focused AI experience. Also supports Anthropic Claude and xAI Grok models via an OpenAI-compatible API bridge.

## 🔬 Deep Research - Primary Feature

**Google Deep Research** is an advanced AI research agent that provides comprehensive, multi-step analysis:

- **Persistent Research Sessions**: Automatic interaction resumption for long-running research
- **Multi-Step Reasoning**: Extended thinking with transparent research process
- **Background Processing**: Research continues even if connection drops
- **Thought Summaries**: Real-time insights into the research methodology
- **OpenAI-Compatible**: Works seamlessly with Open WebUI and other OpenAI-compatible clients

### ⚠️ Important: Rate Limits & Required Concessions

Deep Research has a **1 request per minute (RPM) limit**. To avoid wasting your quota:

**In Open WebUI Settings:**

1. Go to **Settings** → **Interface**
2. **Disable "Auto-Generate Title"** - This would consume a Deep Research request just to generate a chat title
3. **Disable "Auto-Follow-Up Prompts"** - This would consume a Deep Research request for suggestion generation

**Why this matters:**

- Deep Research is designed for comprehensive, time-intensive analysis (30-60+ seconds per query)
- Auto-title and auto-follow-up features make rapid API calls that waste your limited quota
- Session resumption allows you to continue research without consuming additional quota

## 🚀 Features

### Core Functionality

- **🔬 Google Deep Research Integration**: First-class support with automatic session resumption
- **OpenAI-Compatible API Bridge**: FastAPI-based service for multiple AI providers
- **Open WebUI Interface**: Modern, feature-rich web UI for interacting with AI models
- **Multi-Provider Support**:
  - **Google Deep Research & Gemini**: Advanced research agent with thinking models
  - **Anthropic Claude**: Latest Claude models with extended context
  - **xAI Grok**: Latest Grok models
- **Streaming Support**: Real-time streaming responses with reasoning content
- **Bearer Token Authentication**: Secure API access with custom token management
- **Docker-based Deployment**: Easy setup and management with Docker Compose

### Deep Research Capabilities

- **Persistent Sessions**: Research sessions stored in SQLite database
- **Automatic Resumption**: Continue research without starting over (doesn't consume RPM)
- **Connection Recovery**: Auto-reconnects if connection drops
- **Session Tracking**: Clear SDK messages showing session status
- **MD5-based Session Matching**: Same query resumes existing research

### Open WebUI Integration

- **RAG (Retrieval Augmented Generation)**: Upload documents and chat with your data
- **Speech-to-Text**: Audio transcription powered by OpenAI's STT engine
- **Persistent Storage**: Conversation history and uploaded documents preserved
- **Multi-user Support**: Bearer token authentication for multiple users

## 📋 Prerequisites

- Docker and Docker Compose installed
- **Google API key** (required for Deep Research) - [Get one here](https://aistudio.google.com/app/apikey)
- Optional API keys:
  - Anthropic API key ([get one here](https://console.anthropic.com/))
  - xAI API key ([get one here](https://console.x.ai/))

## 🔧 Setup

1. **Clone the repository**

   ```bash
   git clone <your-repo-url>
   cd open-webui-stack
   ```

2. **Set up the SDK interface**

   ```bash
   cd sdk-interface
   cp .env.example .env
   # Edit .env and add your Google API key (required)
   ```

3. **Run database migrations**

   ```bash
   cd sdk-interface
   yoyo apply
   ```

4. **Create the Docker volume**

   ```bash
   docker volume create open-webui
   ```

5. **Start the stack**

   ```bash
   cd ..
   docker-compose up -d
   ```

6. **Configure Open WebUI**
   - Open your browser: `http://localhost:8090`
   - Add the SDK interface as an OpenAI connection
   - **Important**: Disable auto-title and auto-follow-up features (see warning above)

## 🏗️ Architecture

### 1. SDK Interface (`sdk-interface`)

- **Purpose**: OpenAI-compatible API bridge featuring Google Deep Research
- **Technology**: FastAPI, Python 3.11+, SQLite
- **Port**: Internal only (accessed via Docker network)
- **Key Features**:
  - Deep Research session persistence and resumption
  - Converts OpenAI format to provider-specific APIs
  - Handles streaming with reasoning content
  - Bearer token authentication
  - Database: `data/db.sqlite3` with `research_hashes` table

**Endpoints:**

- `GET /v1/models` - List all available models
- `POST /v1/chat/completions` - Chat completions (Deep Research, Claude, Gemini, Grok)
- `GET /health` - Health check

### 2. Open WebUI (`open-webui`)

- **Purpose**: Modern web interface optimized for Deep Research
- **Image**: `ghcr.io/open-webui/open-webui:main`
- **Port**: `8090`
- **Features**: Rich markdown, file upload, RAG, conversation history

### 3. Watchtower (`watchtower`)

- **Purpose**: Automatic container updates
- **Schedule**: Daily at 2 AM

## 🔐 Authentication

Bearer token authentication with `op_wui_` prefix:

```bash
# In sdk-interface/.env
API_KEYS=username1:op_wui_token1;username2:op_wui_token2
```

Generate tokens:

```bash
python sdk-interface/scripts/generate_token.py
```

## 💡 Usage Example

### Deep Research Query

```python
from openai import OpenAI

client = OpenAI(
    api_key="op_wui_your_token_here",
    base_url="http://localhost:8060/v1"
)

# First request - starts new research (consumes 1 RPM)
response = client.chat.completions.create(
    model="deep-research-pro-preview-12-2025",
    messages=[
        {"role": "user", "content": "Research the latest developments in quantum computing"}
    ],
    stream=True
)

for chunk in response:
    delta = chunk.choices[0].delta

    # Research reasoning/thoughts
    if hasattr(delta, 'reasoning_content'):
        print(f"[THINKING] {delta.reasoning_content}")

    # Final research output
    if delta.content:
        print(delta.content, end="")

# Same request later - automatically resumes (FREE - no RPM consumed)
# Output: [SDK] Continuing interaction with id v1_...
```

## 🛠️ Configuration

### Environment Variables

**SDK Interface (`sdk-interface/.env`):**

- **`GOOGLE_API_KEY`** - Your Google API key (required for Deep Research)
- `ANTHROPIC_API_KEY` - Your Anthropic API key (optional)
- `GROK_API_KEY` - Your xAI API key (optional)
- `API_KEYS` - Bearer tokens in format `username:token;username:token`
- `LOG_LEVEL` - Logging level (default: `info`)

**Open WebUI Configuration:**

- Disable auto-title generation
- Disable auto-follow-up prompts
- Configure API endpoint: `http://sdk-interface:8060/v1`

## 📦 Available Models

### Google Deep Research (Primary)

- **`deep-research-pro-preview-12-2025`** - Advanced research agent

### Google Gemini

- `gemini-2.0-flash-exp`
- `gemini-2.0-flash-thinking-exp`
- `gemini-1.5-pro-latest`
- And more...

### Anthropic Claude

- `claude-opus-4-5-20251101`
- `claude-sonnet-4-5-20250929`
- `claude-3-5-sonnet-20241022`
- And more...

### xAI Grok

- `grok-2-vision-1212`
- `grok-2-1212`
- And more...

Full model list available at: `http://localhost:8060/v1/models`

## 🔄 Deep Research Session Management

**How it works:**

1. Each unique message set generates an MD5 hash
2. Hash is stored in database with `interaction_id` when research starts
3. Same query later resumes from stored `interaction_id`
4. Connection drops are automatically handled with reconnection
5. **Session resumption does not consume RPM quota**

**Session Status Messages:**

- `[SDK] Connecting to Deep Research Agent...` - New session starting
- `[SDK] Interaction started...` - Research session created
- `[SDK] Continuing interaction with id v1_...` - Resuming existing session

## 📝 Development

The SDK interface is mounted as a volume for development:

```yaml
volumes:
  - ./sdk-interface:/sdk-interface/app
```

Changes require container restart. See `sdk-interface/README.md` for detailed API documentation.

## ⚠️ Disclaimer

This is an **unofficial** community project and is **not affiliated with or endorsed by:**

- Open WebUI Inc. or its contributors
- Google LLC or Alphabet Inc.
- Anthropic PBC
- xAI Corp.

## 🔗 Links

- [SDK Interface Documentation](./sdk-interface/README.md) - Detailed API documentation
- [Open WebUI Official Repository](https://github.com/open-webui/open-webui)
- [Google AI Studio](https://aistudio.google.com/)
- [Anthropic](https://www.anthropic.com/)
- [xAI](https://x.ai/)

## 📄 License

Please refer to the individual component licenses:

- Open WebUI: See upstream repository
- SDK Interface: See LICENSE file in this repository
