# Google Deep Research & Multi-Model OpenAI API Bridge

A FastAPI service that brings **Google's Deep Research** capabilities to any OpenAI-compatible application, along with support for Anthropic Claude and xAI Grok models.

## 🔬 Deep Research - Main Feature

**Google Deep Research** is an advanced AI research agent that provides comprehensive, multi-step analysis with:

- **Persistent Research Sessions**: Automatic interaction resumption for long-running research
- **Multi-Step Reasoning**: Extended thinking with transparent research process
- **Background Processing**: Research continues even if connection drops
- **Thought Summaries**: Real-time insights into the research methodology
- **OpenAI-Compatible Streaming**: Works with any tool supporting OpenAI's API

### Deep Research Models

- `deep-research-pro-preview-12-2025` - Full-featured deep research agent

### ⚠️ Important: Rate Limits & Required Concessions

Deep Research has a **1 request per minute (RPM) limit**. To avoid wasting your quota, you **must disable** these features in your OpenAI-compatible client:

**For Open WebUI users:**

1. Go to **Settings** → **Interface**
2. **Disable "Auto-Generate Title"** - This would consume a Deep Research request just to generate a chat title
3. **Disable "Auto-Follow-Up Prompts"** - This would consume a Deep Research request for suggestion generation

**Why this matters:**

- Deep Research is designed for comprehensive, time-intensive analysis (often 30-60+ seconds)
- Auto-title and auto-follow-up features make rapid API calls that waste your limited quota
- Session resumption allows you to continue research without consuming additional quota

## Features

- **Google Deep Research Integration**: First-class support for Google's research agent with automatic session resumption
- **OpenAI-Compatible API**: Use Deep Research, Claude, Gemini, and Grok models with the OpenAI API format
- **Multiple Providers**: Seamlessly switch between Google, Anthropic, and xAI models
- **Streaming Support**: Full support for streaming responses with reasoning content
- **Bearer Token Auth**: Optional API key authentication with multiple keys support
- **Automatic Model Discovery**: Fetches available models from each provider
- **Advanced Features**:
  - Persistent interaction storage for deep research sessions
  - Extended context support for Claude models
  - Thinking models support (Claude 3.7 Sonnet with extended thinking)
  - Gemini thinking models with `<thinking>` tag parsing

## Quick Start

### Prerequisites

- Python 3.11+
- **Google API Key** (required for Deep Research)
- Optional API keys:
  - `ANTHROPIC_API_KEY` for Claude models
  - `GROK_API_KEY` for Grok models

### Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd sdk-interface
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables:

```bash
cp .env.example .env
# Edit .env and add your Google API key (required for Deep Research)
```

4. Run database migrations:

```bash
yoyo apply
```

5. Run the server:

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.

## 🚀 Deep Research Usage

### Basic Deep Research Request

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-api-key"
)

response = client.chat.completions.create(
    model="deep-research-pro-preview-12-2025",
    messages=[{"role": "user", "content": "Research the latest developments in quantum computing"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Automatic Session Resumption

Deep Research automatically resumes interrupted sessions. If you make the same request again, it will:

1. Detect the previous research session by message hash
2. Resume from the stored `interaction_id`
3. Continue the research without starting over

```python
# First request - starts new research
response1 = client.chat.completions.create(
    model="deep-research-pro-preview-12-2025",
    messages=[{"role": "user", "content": "Analyze climate change trends"}],
    stream=True
)
# Output: [SDK] Connecting to Deep Research Agent...
#         [SDK] Interaction started...

# Same request later - automatically resumes the research
response2 = client.chat.completions.create(
    model="deep-research-pro-preview-12-2025",
    messages=[{"role": "user", "content": "Analyze climate change trends"}],
    stream=True
)
# Output: [SDK] Continuing interaction with id v1_...
```

## Usage Examples

### List Available Models

```bash
curl http://localhost:8000/v1/models
```

### Deep Research with Thinking Summaries

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-api-key"
)

response = client.chat.completions.create(
    model="deep-research-pro-preview-12-2025",
    messages=[
        {"role": "user", "content": "What are the implications of recent AI safety research?"}
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
```

### Chat Completion with Other Models

```python
# Claude
response = client.chat.completions.create(
    model="claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "Hello, how are you?"}]
)

# Gemini
response = client.chat.completions.create(
    model="gemini-2.0-flash-exp",
    messages=[{"role": "user", "content": "Write a short story about a robot."}],
    stream=True
)

# Grok
response = client.chat.completions.create(
    model="grok-2-vision-1212",
    messages=[{"role": "user", "content": "Explain quantum entanglement"}]
)
```

## Available Models

### Google Deep Research (Primary)

- **`deep-research-pro-preview-12-2025`** - Advanced research agent with multi-step reasoning and automatic session resumption

### Google Gemini Models

- `gemini-2.0-flash-exp`
- `gemini-2.0-flash-thinking-exp`
- `gemini-1.5-pro-latest`
- `gemini-1.5-flash-latest`
- And more...

### Anthropic Claude Models

- `claude-opus-4-5-20251101`
- `claude-sonnet-4-5-20250929`
- `claude-3-5-sonnet-20241022`
- `claude-3-5-haiku-20241022`
- And more...

### xAI Grok Models

- `grok-2-vision-1212`
- `grok-2-1212`
- And more...

The full list of available models is dynamically fetched from each provider and can be retrieved via the `/v1/models` endpoint.

## API Endpoints

### `GET /v1/models`

List all available models from all providers.

### `GET /v1/models/{model_id}`

Get details about a specific model.

### `POST /v1/chat/completions`

Create a chat completion. Supports both streaming and non-streaming modes.

**Deep Research Special Features:**

- Automatic session persistence via message hash
- Interaction ID stored in SQLite database
- Automatic resumption of interrupted research
- Background processing with reconnection support

Request body:

```json
{
  "model": "deep-research-pro-preview-12-2025",
  "messages": [
    { "role": "user", "content": "Research the future of renewable energy" }
  ],
  "stream": true
}
```

Response includes:

- `delta.content` - Final research output
- `delta.reasoning_content` - Research methodology and thinking process
- SDK status messages with `[SDK]` prefix for session management

## Configuration

### Environment Variables

- **`GOOGLE_API_KEY`** - Your Google API key (required for Deep Research)
- `ANTHROPIC_API_KEY` - Your Anthropic API key (optional)
- `GROK_API_KEY` - Your xAI Grok API key (optional)
- `API_KEYS` - Comma-separated list of valid bearer tokens (optional)
  - Format: `name:token,name2:token2`
  - Example: `user1:sk_test123,user2:sk_test456`
- `HOST` - Server host (default: 0.0.0.0)
- `PORT` - Server port (default: 8000)
- `LOG_LEVEL` - Logging level (default: info)
- `DETAILED_REQUEST_LOGGING` - Enable detailed request/response logging (default: false)

### Database

Deep Research uses SQLite for session persistence:

- **Database**: `data/db.sqlite3`
- **Table**: `research_hashes` - stores interaction IDs mapped to message hashes
- **Migrations**: Managed by yoyo-migrations

Run migrations:

```bash
yoyo apply
```

## Advanced Features

### Deep Research Session Management

**Automatic Resumption:**

- Each unique set of messages generates an MD5 hash
- Hash is used to lookup existing research sessions in the database
- If found, research continues from the stored `interaction_id`
- Connection drops are automatically handled with reconnection

**Session Tracking:**

```python
# First request - creates new session
# Output: [SDK] Connecting to Deep Research Agent...
#         [SDK] Interaction started...
#         [SDK] Gemini started reasoning with id: v1_Chd...

# Same request - resumes session
# Output: [SDK] Continuing interaction with id v1_Chd...
```

### Thinking Models

For Claude 3.7 Sonnet and Gemini thinking models, extended thinking is automatically enabled and parsed. The thinking process is streamed as `reasoning_content`.

```python
response = client.chat.completions.create(
    model="claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Solve this complex problem..."}],
    stream=True
)

for chunk in response:
    delta = chunk.choices[0].delta
    if hasattr(delta, 'reasoning_content'):
        print(f"[THINKING] {delta.reasoning_content}")
    if delta.content:
        print(delta.content, end="")
```

### Extended Context

Claude models support up to 200k tokens of context, automatically handled by the bridge.

## Development

### Running Tests

```bash
python test_continuation.py
```

### Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── config.py           # Configuration with Pydantic Settings
│   ├── models.py           # Request/Response models
│   ├── db.py               # Database connection
│   ├── anthropic_client.py # Anthropic API client
│   ├── gemini_client.py    # Gemini API client (Deep Research)
│   ├── grok_client.py      # Grok API client
│   ├── auth.py             # Bearer token authentication
│   └── main.py             # FastAPI application
├── migrations/             # Database migrations (yoyo)
├── data/                   # SQLite database storage
├── requirements.txt
├── .env.example
└── README.md
```

## License

MIT

## Notes

- This is just an internal tool made public
