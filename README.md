# 🧠 AI Research Hub: Four Powerful AI Models in One Interface

> **Quick Navigation:** [👶 Newbie Guide](#-for-newbies-your-first-ai-research-hub) | [🎯 Expert Setup](#-for-experts-advanced-deployment) | [💰 Cost Analysis](#-cost-analysis) | [☁️ Cloud Deployment](#️-optional-cloud-deployment)

Experience the cutting edge of AI research with **Google's Deep Research** combined with Claude, Gemini, and Grok - all accessible through one beautiful interface. This stack brings together four of the most powerful AI models available today, optimized for comprehensive research, analysis, and creative work.

## 🌟 Why This Stack?

Imagine having four expert researchers at your fingertips:

- **Google Deep Research**: Your tireless research assistant that dives deep into topics, following leads and synthesizing findings over 30-60+ seconds of focused analysis
- **Claude Opus 4 & Sonnet 4**: Anthropic's most capable models for complex reasoning, analysis, and extended context work
- **Gemini 2.0 Flash & Thinking**: Google's fastest models for quick iterations and structured thinking
- **Grok 2**: xAI's unique perspective with real-time web search capabilities

All unified in a single, modern web interface with conversation history, document upload, and seamless model switching.

**💳 Budget Control Built-In:** All providers (Google, Anthropic, xAI) offer pay-as-you-go API pricing with automatic balance top-ups and monthly spending limits. You're in full control of your costs - set a $10/month limit and never worry about surprise bills.

## 💰 Cost Analysis

### Self-Hosting (Recommended - $0/month)

**Perfect for:** Personal use, privacy-focused users, learning, development

**Minimum Requirements:**

- **CPU**: Dual-core processor (2010 or newer)
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 20GB free space (SSD recommended)
- **OS**: Ubuntu 20.04+, macOS, Windows with Docker Desktop
- **Network**: Stable broadband connection

**What you can use:**

- Old laptop gathering dust
- Desktop computer you already have
- Raspberry Pi 4 (4GB+ RAM)
- Home server or NAS

**Cost**: $0/month (electricity ~$2-5/month depending on usage)

**Benefits:**

- ✅ Complete privacy - your data never leaves your machine
- ✅ No monthly hosting fees
- ✅ Full control over updates and configuration
- ✅ Learn Docker and infrastructure skills

---

### Cloud Hosting (For Remote Access)

**Perfect for:** Remote access from anywhere, team collaboration, always-on availability

⚠️ **Cloud hosting costs $30-65/month** - consider if you really need it! Most users are better off with self-hosting.

**AWS (Expensive):**

- **Minimum**: t3.small (2GB RAM) - ~$15/month + $5 storage = **$20/month minimum**
- **Comfortable**: t3.medium (4GB RAM) - ~$30/month + $5 storage = **$35/month**
- **Recommended**: t3.large (8GB RAM) - ~$60/month + $5 storage = **$65/month**
- **Free tier** (first 12 months only): t2.micro (1GB RAM) - too small for this stack, will struggle

**Why AWS is expensive:**

- Data transfer costs (~$9/100GB)
- EBS storage costs ($3-5/month for 30-50GB)
- Instance pricing higher than competitors
- Complex billing with surprise charges

**⚡ Our Recommendation:**

- **For personal use**: Self-host on your laptop/desktop ← **Start here!**
- **For always-on remote access**: Oracle Cloud Free Tier or Hetzner Cloud ($4-8/month)
- **For production/business**: DigitalOcean or Linode ($10-20/month)
- **Avoid AWS unless**: You already have AWS credits or your company requires it

### API Costs (Pay-as-you-go)

The real costs are in API usage:

- **Google Deep Research**: 1 request/minute limit (free tier generous)
- **Claude Opus 4**: $15 per 1M input tokens, $75 per 1M output tokens
- **Gemini 2.0 Flash**: $0 (free tier) or very low cost
- **Grok**: Varies by plan

**Typical monthly usage for moderate use**: $10-50 in API costs

---

## 👶 For Newbies: Your First AI Research Hub

### What You'll Need (The Shopping List)

Before we start, make sure you have:

1. ✅ A computer (laptop or desktop with 4GB+ RAM) - that's it!
2. ✅ Internet connection
3. ✅ 1-2 hours of time
4. ✅ A Google account (for API keys - it's free!)

### 🔑 Step 0: Get Your API Keys (Do This First!)

Before deploying anything, you need API keys from the AI providers. **Important:** These are pay-as-you-go API accounts, NOT monthly subscriptions.

#### Google AI Studio (Required - Free Tier Available)

**Why you need this:** Powers Deep Research and Gemini models.

1. **Go to:** [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. **Sign in** with your Google account
3. Click **"Create API Key"** (or "Get API Key" if it's your first time)
4. Select a project (create one if needed, it's free)
5. **Copy the key** - it starts with `AIza...` (save this somewhere safe!)

**✅ What you're getting:** API access (pay-as-you-go)  
**❌ What to AVOID:** Don't click "Google One AI Premium" or subscribe to anything - you just need the free API key!

**Budget Control:**

- Go to [console.cloud.google.com/billing](https://console.cloud.google.com/billing)
- Click **Budgets & Alerts** → **Create Budget**
- Set your monthly limit (e.g., $10/month)
- Enable alerts at 50%, 90%, 100% of budget
- Enable **automatic billing account disablement** when budget is exceeded (optional but recommended)

**Free Tier:** Google gives generous free quotas for Gemini models. Deep Research has a 1 request/minute limit.

---

#### Anthropic Console (Optional - For Claude Models)

**Why you need this:** Powers Claude Opus 4 and Sonnet 4.

1. **Go to:** [console.anthropic.com](https://console.anthropic.com)
2. Click **"Sign Up"** (or "Sign In" if you have an account)
3. Complete email verification
4. **Skip the "Claude Pro" subscription page** - you don't need it!
5. Go to **Settings** (gear icon, top right) → **API Keys**
6. Click **"Create Key"**
7. Give it a name (like "AI Research Hub")
8. **Copy the key** - it starts with `sk-ant-...` (save it!)

**✅ What you're getting:** API access (pay-as-you-go)  
**❌ What to AVOID:** Don't subscribe to "Claude Pro" ($20/month) - that's for the chat interface, not API access!

**Budget Control:**

- Go to **Settings** → **Billing**
- Click **"Add Credit"** - minimum $5
- Under **"Usage Limits"**, set monthly spending limit (e.g., $10)
- Enable **automatic top-up** (optional): auto-add $20 when balance drops below $5
- You'll get email alerts when you hit 75%, 90%, and 100% of limits

**Pricing:** ~$3 for 1M input tokens on Sonnet 4, ~$15 for Opus 4

---

#### xAI Console (Optional - For Grok Models)

**Why you need this:** Powers Grok 2 with real-time web search.

1. **Go to:** [console.x.ai](https://console.x.ai)
2. Click **"Sign Up"** or **"Sign In"** with 𝕏 (Twitter) account
3. **Skip any Grok subscription prompts** - those are for the chat interface!
4. Go to **API Keys** section
5. Click **"Create API Key"**
6. Name it (like "AI Hub")
7. **Copy the key** - it starts with `xai-...` (save it!)

**✅ What you're getting:** API access (pay-as-you-go)  
**❌ What to AVOID:** Don't subscribe to "Grok Premium" or "Grok+" - those are for using Grok on 𝕏, not for API access!

**Budget Control:**

- Go to **Settings** → **Billing & Usage**
- Click **"Add Credits"** - minimum $10
- Set **monthly spending limit** under "Usage Limits" (e.g., $15)
- Enable **auto-recharge** (optional): automatically add $25 when balance < $5
- Configure **email alerts** at 50%, 80%, 100% of limit

**Pricing:** Varies by model, check [x.ai/api](https://x.ai/api) for current rates

---

**💡 Pro Tips for All Providers:**

- **Start small:** Add $5-10 initially, monitor usage for a week
- **Set hard limits:** All providers let you cap monthly spending
- **Enable alerts:** Get emails before you hit limits
- **Auto top-up is safe:** It respects your monthly limit - if you set $10/month max and enable $20 auto-topup, it will only add money if you haven't hit the $10/month cap
- **Monitor usage:** Check dashboards weekly at first
- **API ≠ Subscription:** Never pay for "Pro" or "Premium" subscriptions - those are for chat interfaces, not API access

**🔒 Security Note:** Treat API keys like passwords. Never share them publicly or commit them to GitHub!

---

### Choose Your Adventure

Pick the path that matches your situation:

- **[Path A: Self-Host on Your Computer](#path-a-self-host-on-your-computer-recommended)** ← **Recommended!** Works on Mac, Windows, Linux
- **[Path B: Deploy with VSCode Dev Containers](#path-b-deploy-with-vscode-dev-containers)** ← For developers who love VSCode
- **[Path C: Deploy to Cloud](#path-c-optional-deploy-to-cloud)** ← Only if you need remote access

---

### Path A: Self-Host on Your Computer (Recommended)

Perfect if you want to run this on your own hardware - works on Ubuntu, macOS, Windows, or even Raspberry Pi!

#### Step 1: Install Docker

**For Ubuntu/Linux:**

Open Terminal (`Ctrl + Alt + T` on Ubuntu) and run:

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install required dependencies (including SQLite for Python)
sudo apt install -y build-essential libsqlite3-dev git curl

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add your user to docker group (no sudo needed after this)
sudo usermod -aG docker $USER

# Install Docker Compose
sudo apt install docker-compose-plugin -y

# Reboot to apply changes
sudo reboot
```

After reboot, verify Docker is installed:

```bash
docker --version
docker compose version
```

**⚠️ Important Note About Python:** If you're planning to run tests locally (not required for Docker deployment), make sure Python has SQLite support. The `libsqlite3-dev` package we installed above ensures this.

**For macOS:**

1. Download **Docker Desktop** from [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/)
2. Install and start Docker Desktop
3. Wait for the whale icon in the menu bar to show "Docker Desktop is running"

**For Windows:**

1. Download **Docker Desktop** from [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/)
2. Install Docker Desktop (requires WSL 2)
3. Start Docker Desktop and wait for it to be ready

#### Step 2: Deploy the Stack

Open Terminal (or PowerShell on Windows) and run:

```bash
# Clone the repository
git clone https://github.com/israelsaba/open-webui-stack.git
cd open-webui-stack

# Set up using Make (recommended)
make setup

# Edit configuration file
# On Mac/Linux:
nano .env
# On Windows or if you prefer a GUI:
code .env
# or: notepad .env
```

**If you haven't gotten your API keys yet, go back to [Step 0: Get Your API Keys](#-step-0-get-your-api-keys-do-this-first)!**

Add your API keys to `.env`:

```bash
SDK__GOOGLE_API_KEY=your-google-key-here
SDK__ANTHROPIC_API_KEY=your-anthropic-key-here  # Optional
SDK__GROK_API_KEY=your-grok-key-here  # Optional
```

**Save the file:**

- nano: `Ctrl+O`, `Enter`, `Ctrl+X`
- VSCode/Notepad: Click Save

#### Step 3: Start Services

```bash
# Create Docker volume
docker volume create open-webui

# Start everything
make up

# OR manually:
docker compose up -d

# Check status (all should show "Up (healthy)")
docker compose ps
```

You should see three services running: `open-webui`, `sdk-interface`, and `watchtower`.

#### Step 4: Access Your AI Hub

Open your browser and go to: **`http://localhost:8090`**

**First-Time Setup:**

1. **Create your account** (first user becomes admin automatically)
2. You'll see the Open WebUI interface - but no models yet!

**Configure Interface Settings (Important for Deep Research):**

3. Click your profile picture (bottom left) → **Settings** → **Interface**
4. Scroll down and **disable** these two options:
   - ❌ **"Auto-Generate Title"** - Would waste Deep Research quota on title generation
   - ❌ **"Auto-Follow-Up Prompts"** - Would waste quota on suggestion generation
5. Click **Save**

**Connect to SDK Interface (The Magic Step!):**

6. Go to **Settings** → **Connections** (or **Admin Panel** → **Connections**)
7. Click **"+ Add OpenAI Connection"** (or **"Add Connection"**)
8. Fill in:
   - **Name**: `SDK Interface` (or any name you like)
   - **API Base URL**: `http://sdk-interface:8060/v1`
   - **API Key**: Leave blank OR type anything (like `test` - it's not used)
9. Click **Save** or **Add**
10. You should see ✓ Success message

**Verify It's Working:**

11. Go back to the main chat interface
12. Click the model dropdown (top left, says "Select a model")
13. You should see models like:
    - `deep-research-pro-preview-12-2025`
    - `claude-sonnet-4-5-20250929`
    - `gemini-2.0-flash-exp`
    - `grok-2-vision-1212`

**🎉 You're done!** Try asking: _"Research the latest developments in quantum computing"_

---

#### 🔧 Troubleshooting: Can't Connect to SDK Interface?

**Problem: No models showing up or connection fails**

This is usually a Docker networking issue. Let's debug:

**Step 1: Check Services Are Running**

```bash
docker compose ps
```

You should see:

```
NAME            STATUS
open-webui      Up (healthy)
sdk-interface   Up (healthy)
watchtower      Up (healthy)
```

**If sdk-interface is not running:**

```bash
docker compose logs sdk-interface --tail=50
docker compose restart sdk-interface
```

**Step 2: Verify SDK Interface Is Responding**

```bash
# Test from inside the open-webui container
docker compose exec open-webui curl http://sdk-interface:8060/health

# Should return: {"status":"healthy"}
```

**If this fails**, the Docker network is broken:

```bash
docker compose down
docker compose up -d
```

**Step 3: Check Docker Network**

```bash
docker network ls | grep open-webui
docker network inspect open-webui-net
```

You should see both `open-webui` and `sdk-interface` containers in the network.

**Step 4: Common Issues and Fixes**

| Issue                         | Solution                                                               |
| ----------------------------- | ---------------------------------------------------------------------- |
| "Connection refused"          | SDK Interface not running. Run: `docker compose restart sdk-interface` |
| "Network not found"           | Run: `docker compose down && docker compose up -d`                     |
| "Name does not resolve"       | Use `http://sdk-interface:8060/v1` NOT `http://localhost:8060/v1`      |
| "No models" but connection OK | Check logs: `docker compose logs sdk-interface \| grep ERROR`          |

**Understanding Docker Networking:**

- **Inside Docker** (container to container): Use `http://sdk-interface:8060` ✅
- **From your host** (your computer): Use `http://localhost:8060` ✅
- **From internet**: Use `http://YOUR-IP:8060` (if port exposed) ✅

**Open WebUI runs INSIDE Docker**, so it must use the container name `sdk-interface`, NOT `localhost`!

**Why `localhost` doesn't work:**

- `localhost` from inside the `open-webui` container refers to ITSELF, not your computer
- Docker provides DNS that resolves `sdk-interface` to the correct container IP
- This is why we use service names in `docker-compose.yml`

**Step 5: Manual Connection Test**

If you want to verify the SDK Interface works from your computer:

```bash
# This should return a list of models
curl http://localhost:8060/v1/models
```

If this works but Open WebUI can't connect, it's definitely a Docker network issue.

**Step 6: Nuclear Option (Fresh Start)**

If nothing works:

```bash
# Stop everything
docker compose down

# Remove network (Docker will recreate it)
docker network rm open-webui-net

# Start fresh
docker compose up -d

# Wait 30 seconds, then check
docker compose ps
```

---

#### 📍 Understanding File Paths (Mac/Linux)

When you see paths in commands, here's what they mean:

**Absolute Paths** (start from root):

- `/opt/open-webui-stack` - Starts from system root `/`
- `/home/username/projects` - Your user's home directory
- Use these when you want to be explicit about location

**Relative Paths** (relative to where you are):

- `./sdk-interface` - Current directory, then `sdk-interface` folder
- `../parent-folder` - Go up one level, then into `parent-folder`
- `~` - Shortcut for your home directory
  - Mac: `/Users/your-username`
  - Linux: `/home/your-username`

**Examples:**

```bash
# These all go to the same place (if you're in your home directory)
cd ~/open-webui-stack
cd /home/username/open-webui-stack  # Linux
cd /Users/username/open-webui-stack  # Mac
cd open-webui-stack  # If already in home directory

# Relative navigation
cd ~/open-webui-stack
cd sdk-interface        # Now in ~/open-webui-stack/sdk-interface
cd ../                  # Back to ~/open-webui-stack
cd ..                   # Back to ~
```

**Where To Install:**

**On Mac/Linux Desktop:**

- Recommended: `~/projects/open-webui-stack` or `~/open-webui-stack`
- Why: Easy to find, you own it, no sudo needed

**On Server (AWS/Cloud):**

- Recommended: `/opt/open-webui-stack`
- Why: Standard location for optional software, persists across user sessions

**Current directory** check:

```bash
pwd  # Shows where you are right now
```

---

### Path B: Deploy with VSCode Dev Containers

Perfect for developers who want a clean, reproducible environment.

#### Step 1: Install Prerequisites

1. **Install VSCode**: [code.visualstudio.com](https://code.visualstudio.com/)
2. **Install Docker Desktop**:
   - Mac: [Docker Desktop for Mac](https://docs.docker.com/desktop/install/mac-install/)
   - Windows: [Docker Desktop for Windows](https://docs.docker.com/desktop/install/windows-install/)
   - Linux: Use Docker Engine (see Path B, Step 2)
3. **Install Dev Containers extension**:
   - Open VSCode
   - Press `Ctrl/Cmd + Shift + X`
   - Search for "Dev Containers"
   - Click "Install" on the Microsoft extension

#### Step 2: Clone and Open

**Mac/Linux paths:**

- We'll clone to your home directory for easy access
- `~` = `/Users/your-username` (Mac) or `/home/your-username` (Linux)

```bash
# Open Terminal (or use VSCode's built-in terminal)
cd ~

# Clone repository
git clone https://github.com/israelsaba/open-webui-stack.git

# Open in VSCode
cd open-webui-stack
code .
```

**Optional - Open in Dev Container:**

If you want an isolated development environment:

1. Press `Ctrl/Cmd + Shift + P`
2. Type "Dev Containers: Reopen in Container"
3. Wait for container to build (2-5 minutes first time)

#### Step 3: Configure and Start

**In VSCode's integrated terminal** (Terminal → New Terminal):

```bash
# Use the root Makefile for easy setup
make setup

# Edit .env file directly in VSCode
# File explorer (left side) → click .env
# Add your API keys:
#   SDK__GOOGLE_API_KEY=your-key
#   SDK__ANTHROPIC_API_KEY=your-key (optional)
#   SDK__GROK_API_KEY=your-key (optional)
# Save with Cmd/Ctrl + S

# Create Docker volume
docker volume create open-webui

# Start all services
make up
```

**Alternative (manual docker compose):**

```bash
docker compose up -d
docker compose ps
```

#### Step 4: Access and Configure

1. Open browser: `http://localhost:8090`
2. **Follow Path A, Step 6** for complete setup instructions including:
   - Creating your account
   - Connecting to SDK Interface (`http://sdk-interface:8060/v1`)
   - Disabling auto-title/auto-follow-up
   - Troubleshooting connection issues

**Having issues?** See the [Troubleshooting section](#🔧-troubleshooting-cant-connect-to-sdk-interface) above.

---

### Path C: (Optional) Deploy to Cloud

⚠️ **Think twice before cloud hosting** - see [Cost Analysis](#-cost-analysis) above for why self-hosting is usually better!

**When you actually need cloud:**

- You travel frequently and need access from anywhere
- You're collaborating with a team
- Your local machine can't run 24/7

**Recommended Cloud Providers (Cheapest to Most Expensive):**

1. **Oracle Cloud Free Tier** (Always Free!)
   - [oracle.com/cloud/free](https://oracle.com/cloud/free/)
   - Forever free ARM instance (1-4GB RAM)
   - Perfect for this stack
   - [Deployment Guide](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm)

2. **Hetzner Cloud** (€4-8/month)
   - [hetzner.com/cloud](https://www.hetzner.com/cloud)
   - Best value: CX11 (2GB RAM, 1 vCPU) for €4/month
   - EU-based, excellent performance
   - Simple pricing, no surprises

3. **DigitalOcean** ($6-12/month)
   - [digitalocean.com](https://www.digitalocean.com/)
   - Basic Droplet: $6/month (1GB RAM) or $12/month (2GB RAM)
   - Great docs and community
   - $200 free credit for new users (60 days)

4. **Linode/Akamai** ($5-10/month)
   - [linode.com](https://www.linode.com/)
   - Nanode 1GB for $5/month
   - Clean UI, reliable
   - $100 free credit for new users

5. **AWS** ($20-65/month - NOT Recommended)
   - Expensive: t3.small minimum ~$20/month
   - Complex billing with hidden costs
   - Only use if you have existing AWS credits or corporate requirement

**General Cloud Deployment Steps:**

All providers follow the same pattern:

1. **Create Instance/Droplet/VM:**
   - Choose Ubuntu 22.04 LTS
   - Select at least 2GB RAM (4GB recommended)
   - 20-30GB storage
   - Enable SSH access

2. **Connect via SSH:**

   ```bash
   ssh root@YOUR-SERVER-IP
   ```

3. **Install Docker:**

   ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh
   sh get-docker.sh
   ```

4. **Deploy the Stack:**

   ```bash
   git clone https://github.com/israelsaba/open-webui-stack.git
   cd open-webui-stack
   make setup
   # Edit .env with your API keys
   nano .env
   make up
   ```

5. **Access:**
   - Open browser to `http://YOUR-SERVER-IP:8090`
   - Follow Path A, Step 4 for configuration

6. **Secure Your Instance:**
   - Set up firewall (allow ports 22, 8090)
   - Add SSH key auth (disable password)
   - Configure SSL/TLS for HTTPS (use Caddy or nginx + Let's Encrypt)

**Important Security Notes:**

- Never use root account for deployment
- Always enable firewall
- Use SSH keys, not passwords
- Consider adding Cloudflare for DDoS protection
- Set up automatic backups

For detailed cloud deployment guides, check each provider's documentation or ask in the GitHub Discussions.

---

## 🎯 For Experts: Advanced Deployment

### Quick Start

```bash
git clone https://github.com/israelsaba/open-webui-stack.git
cd open-webui-stack

# Setup (creates .env from .env.example)
make setup

# Edit root .env and add your API keys with SDK__ prefix:
# SDK__GOOGLE_API_KEY=your-key
# SDK__ANTHROPIC_API_KEY=your-key
# SDK__GROK_API_KEY=your-key

# Start everything
docker volume create open-webui
make up

# Or manually:
# docker compose up -d
```

**Root Makefile Commands:**

- `make help` - Show all available commands
- `make setup` - Initialize project and sdk-interface
- `make test` - Run tests (mock mode)
- `make test-cov` - Run tests with coverage
- `make test-real` - Run tests against real APIs
- `make up/down/restart` - Docker compose operations
- `make logs` - View logs from all services
- `make run` - Run sdk-interface in dev mode

### Architecture Overview

```
┌─────────────────┐
│   Open WebUI    │  Port 8090 (Web UI)
│   (Frontend)    │
└────────┬────────┘
         │
         │ HTTP
         ▼
┌─────────────────┐
│ SDK Interface   │  Port 8060 (Internal)
│   (API Bridge)  │
└────────┬────────┘
         │
         ├─────────► Google Deep Research + Gemini
         ├─────────► Anthropic Claude
         └─────────► xAI Grok
```

### Custom Deployments

**Kubernetes:**

- Convert docker-compose.yml to K8s manifests
- Use ConfigMaps for .env files
- Persistent volumes for sqlite and open-webui data
- Consider using managed postgres for sdk-interface

**Reverse Proxy (Nginx/Traefik):**

- SSL termination recommended
- Sample nginx config in `docs/nginx-example.conf`
- WebSocket support required for streaming

**High Availability:**

- SDK interface is stateless (except sqlite sessions)
- Open WebUI requires sticky sessions
- Consider redis for session storage
- External postgres for production

**Security Hardening:**

- Use AWS Secrets Manager / Vault for API keys
- Rotate bearer tokens regularly
- Enable rate limiting at reverse proxy
- Regular security updates via Watchtower

**Monitoring:**

- Prometheus metrics at `/metrics` (add to sdk-interface)
- Grafana dashboards available in `docs/grafana/`
- Log aggregation with ELK or Loki recommended

### Environment Variables Reference

See `sdk-interface/.env.example` for complete reference.

**Required:**

- `GOOGLE_API_KEY` - For Deep Research and Gemini

**Optional:**

- `ANTHROPIC_API_KEY` - For Claude models
- `GROK_API_KEY` - For Grok models
- `API_KEYS` - Bearer token auth (format: `user:token;user2:token2`)
- `LOG_LEVEL` - debug/info/warning/error (default: info)
- `INTERACTION_POLL_INTERVAL` - Deep Research polling (default: 30s)

### Performance Tuning

**SDK Interface:**

- Increase uvicorn workers for concurrent requests
- Use external postgres instead of sqlite for production
- Enable connection pooling
- Consider caching layer (Redis) for model lists

**Open WebUI:**

- Adjust `MAX_UPLOAD_SIZE` for large documents
- Configure S3 for file storage instead of local volume
- Enable CDN for static assets

### Development Setup

See [CLAUDE.md](./CLAUDE.md) and [AGENTS.md](./AGENTS.md) for detailed development instructions.

**Quick test setup:**

```bash
cd sdk-interface
make setup          # Install all dependencies
make run           # Start dev server
make test          # Run test suite
make test-cov      # Run tests with coverage
```

---

## 📚 Additional Resources

- **[CLAUDE.md](./CLAUDE.md)** - Instructions for AI assistants (Claude, GPT, etc.) working on this codebase
- **[AGENTS.md](./AGENTS.md)** - Instructions for autonomous AI agents deploying and maintaining this stack
- **[sdk-interface/README.md](./sdk-interface/README.md)** - Detailed API documentation
- **[GitHub Issues](https://github.com/israelsaba/open-webui-stack/issues)** - Bug reports and feature requests

## 🆘 Troubleshooting

### Common Issues

**"Connection refused" when accessing web UI:**

```bash
# Check if services are running
docker compose ps

# Check logs
docker compose logs open-webui
docker compose logs sdk-interface

# Restart services
docker compose restart
```

**Deep Research not working:**

- Verify GOOGLE_API_KEY is set correctly in sdk-interface/.env
- Check you haven't exceeded rate limit (1 req/min)
- Look for errors in logs: `docker compose logs sdk-interface`

**Models not showing up:**

- Verify API keys are correct
- Check network connectivity to provider APIs
- Review logs for authentication errors

**Out of memory errors:**

- Increase Docker memory limits in Docker Desktop settings
- Or allocate more RAM to your VM/instance
- Minimum 4GB recommended, 8GB ideal

**"ModuleNotFoundError: No module named '_sqlite3'" when running tests:**

This happens when Python was compiled without SQLite support. Fix it:

```bash
# On Ubuntu/Debian - Install SQLite development headers
sudo apt install libsqlite3-dev

# Option 1: Reinstall Python with SQLite support
# Ubuntu/Debian - use deadsnakes PPA for Python 3.12
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev

# Option 2: Build Python from source with SQLite
# Download and build Python 3.12 (example)
wget https://www.python.org/ftp/python/3.12.13/Python-3.12.13.tgz
tar -xf Python-3.12.13.tgz
cd Python-3.12.13
./configure --enable-optimizations
make -j $(nproc)
sudo make altinstall

# After Python is reinstalled, recreate virtual environment
cd ~/open-webui-stack/sdk-interface
rm -rf .venv
make setup  # Uses python3 by default, or set: make setup PYTHON=python3.12
```

**Important:** Your database at `sdk-interface/data/db.sqlite3` is NOT affected by rebuilding Python or recreating the virtual environment. Only Python packages in `.venv/` are reinstalled.

**Backup first (optional but recommended):**
```bash
cp sdk-interface/data/db.sqlite3 ~/db.sqlite3.backup
```

**Quick fix for testing:** If you only need to run the stack (not tests), just use Docker - no Python setup needed!

### Get Help

1. Check [existing GitHub issues](https://github.com/israelsaba/open-webui-stack/issues)
2. Search [Open WebUI discussions](https://github.com/open-webui/open-webui/discussions)
3. Review logs: `docker compose logs`
4. Create a new issue with logs and your setup details

## ⚠️ Important Notes

### Deep Research Rate Limits

Google Deep Research has a **1 request per minute (RPM)** limit. To avoid wasting your quota:

**In Open WebUI Settings:**

1. Settings → Interface
2. ❌ Disable "Auto-Generate Title"
3. ❌ Disable "Auto-Follow-Up Prompts"

**Why:** These features make rapid API calls that waste your limited quota. Deep Research is designed for comprehensive 30-60+ second analyses, not quick title generation.

### Session Resumption

Deep Research sessions are persistent! If you ask the same question again, it continues from where it left off **without consuming your RPM quota**. This is a key feature for long-running research.

## 🔐 Security Best Practices

- Never commit `.env` files with real API keys
- Use AWS Secrets Manager or similar for production
- Enable firewall rules to restrict access
- Rotate bearer tokens regularly
- Keep Docker images updated (Watchtower handles this)
- Use HTTPS in production (add reverse proxy)

## 📄 License

This project is a community integration. Individual components have their own licenses:

- SDK Interface: MIT License (see LICENSE)
- Open WebUI: [Upstream license](https://github.com/open-webui/open-webui)

## ⚠️ Disclaimer

This is an **unofficial** community project. Not affiliated with or endorsed by Open WebUI Inc., Google LLC, Anthropic PBC, or xAI Corp.

---

**Ready to start?** Choose your path above and begin your AI research journey! 🚀
