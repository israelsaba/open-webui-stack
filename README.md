# 🧠 AI Research Hub: Four Powerful AI Models in One Interface

> **Quick Navigation:** [👶 Newbie Guide](#-for-newbies-your-first-ai-research-hub) | [🎯 Expert Setup](#-for-experts-advanced-deployment) | [💰 Cost Analysis](#-cost-analysis)

Experience the cutting edge of AI research with **Google's Deep Research** combined with Claude, Gemini, and Grok - all accessible through one beautiful interface. This stack brings together four of the most powerful AI models available today, optimized for comprehensive research, analysis, and creative work.

## 🌟 Why This Stack?

Imagine having four expert researchers at your fingertips:

- **Google Deep Research**: Your tireless research assistant that dives deep into topics, following leads and synthesizing findings over 30-60+ seconds of focused analysis
- **Claude Opus 4 & Sonnet 4**: Anthropic's most capable models for complex reasoning, analysis, and extended context work
- **Gemini 2.0 Flash & Thinking**: Google's fastest models for quick iterations and structured thinking
- **Grok 2**: xAI's unique perspective with real-time web search capabilities

All unified in a single, modern web interface with conversation history, document upload, and seamless model switching.

## 💰 Cost Analysis

### Cloud Hosting (AWS)

**Minimum Viable Setup:**
- **EC2 Instance**: t3.medium (2 vCPU, 4GB RAM)
  - Cost: ~$30/month (on-demand) or ~$18/month (1-year reserved)
- **Storage**: 30GB EBS GP3
  - Cost: ~$3/month
- **Data Transfer**: 100GB/month
  - Cost: ~$9/month
- **Total**: ~$42/month (on-demand) or ~$30/month (reserved)

**Recommended Setup for Production:**
- **EC2 Instance**: t3.large (2 vCPU, 8GB RAM)
  - Cost: ~$60/month (on-demand) or ~$36/month (1-year reserved)
- **Storage**: 50GB EBS GP3
  - Cost: ~$5/month
- **Total**: ~$65/month (on-demand) or ~$41/month (reserved)

**Free Tier Eligible** (first 12 months):
- 750 hours/month of t2.micro or t3.micro
- 30GB EBS storage
- 15GB data transfer out
- **Estimated cost**: $0-5/month for light usage

### Self-Hosting (Old Laptop/Desktop)

**Minimum Requirements:**
- **CPU**: Dual-core processor (2010 or newer)
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 20GB free space (SSD recommended)
- **OS**: Ubuntu 20.04+ or any Linux with Docker support
- **Network**: Stable broadband connection

**Cost**: $0/month (electricity ~$2-5/month depending on usage)

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

1. ✅ A computer (laptop or desktop with 4GB+ RAM)
2. ✅ Internet connection
3. ✅ 1-2 hours of time
4. ✅ A Google account (for API keys - it's free!)
5. ✅ Optional: AWS account (also free to start)
6. ✅ Optional: Credit card for AWS (won't be charged unless you exceed free tier)

### Choose Your Adventure

Pick the path that matches your situation:

- **[Path A: Deploy on AWS Free Tier](#path-a-deploy-on-aws-free-tier)** ← Best for remote access, free for 12 months
- **[Path B: Deploy on Your Own Ubuntu Machine](#path-b-deploy-on-your-own-ubuntu-machine)** ← Best for privacy, uses your hardware
- **[Path C: Deploy with VSCode Dev Containers](#path-c-deploy-with-vscode-dev-containers)** ← Best for developers

---

### Path A: Deploy on AWS Free Tier

AWS (Amazon Web Services) will host your AI hub in the cloud. The free tier is perfect for getting started.

#### Step 1: Create Your AWS Account

1. Go to [aws.amazon.com](https://aws.amazon.com)
2. Click **"Create an AWS Account"** (big orange button, top right)
3. Fill in:
   - **Email address**: Your personal email
   - **Password**: Something secure (use a password manager!)
   - **AWS account name**: Something like "MyAIHub" or your name
4. Click **"Continue"**
5. Fill in your contact information (they need this for billing, even though you're using free tier)
6. Enter your credit card details (required, but you won't be charged unless you exceed free tier limits)
7. Verify your identity (phone call or SMS)
8. Choose **"Basic Support - Free"** plan
9. **Congratulations!** You have an AWS account 🎉

#### Step 2: Launch Your First EC2 Instance (Your Cloud Computer)

1. Log into AWS Console: [console.aws.amazon.com](https://console.aws.amazon.com)
2. In the search bar at the top, type **"EC2"** and click on it
3. Make sure you're in a region close to you (top right, e.g., "US East" or "EU West")
4. Click the big orange button **"Launch Instance"**

**Configure your instance:**

5. **Name**: Type something like "AI-Research-Hub"
6. **Application and OS Images**: 
   - Click **"Ubuntu"**
   - Select **"Ubuntu Server 22.04 LTS"** (the one marked "Free tier eligible")
7. **Instance type**: 
   - Select **"t3.micro"** for free tier (or **"t3.medium"** for better performance ~$30/month)
8. **Key pair**: 
   - Click **"Create new key pair"**
   - Name it "ai-hub-key"
   - Type: **RSA**
   - Format: **.pem** (for Mac/Linux) or **.ppk** (for Windows with PuTTY)
   - Click **"Create key pair"** - this downloads a file. **SAVE IT SECURELY** (you can't download it again!)
9. **Network settings**:
   - Check **"Allow SSH traffic from Anywhere"** (we'll secure this later)
   - Check **"Allow HTTPS traffic from the internet"**
   - Check **"Allow HTTP traffic from the internet"**
10. **Storage**: 
    - Change from 8 GB to **30 GB** (still free tier eligible)
11. Click **"Launch instance"** (big orange button at bottom)

Wait 2-3 minutes for your instance to start.

#### Step 3: Connect to Your Cloud Computer

**For Mac/Linux users:**

1. Open Terminal (Applications → Utilities → Terminal)
2. Navigate to where you saved the key file:
   ```bash
   cd ~/Downloads
   chmod 400 ai-hub-key.pem
   ```
3. Go back to AWS Console, click on your instance
4. Click the **"Connect"** button at the top
5. Copy the ssh command (looks like `ssh -i "ai-hub-key.pem" ubuntu@ec2-XX-XX-XX-XX.compute.amazonaws.com`)
6. Paste it in your Terminal and press Enter
7. Type **"yes"** when asked about fingerprint

**For Windows users:**

1. Download [PuTTY](https://www.putty.org/)
2. In AWS Console, click your instance → **"Connect"** → **"SSH client"** tab
3. Copy the public DNS (looks like `ec2-XX-XX-XX-XX.compute.amazonaws.com`)
4. Open PuTTY:
   - **Host Name**: paste the DNS you copied
   - **Port**: 22
   - **Connection type**: SSH
5. On the left, go to **Connection → SSH → Auth**
6. Click **"Browse"** and select your `.ppk` key file
7. Click **"Open"**
8. Login as: **ubuntu**

You're in! You should see a command prompt like `ubuntu@ip-XXX:~$`

#### Step 4: Install Docker (The Container Platform)

Copy and paste these commands one at a time (press Enter after each):

```bash
# Update the system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add your user to docker group (so you don't need sudo)
sudo usermod -aG docker ubuntu

# Install Docker Compose
sudo apt install docker-compose-plugin -y

# Log out and back in for changes to take effect
exit
```

Now reconnect using the same SSH command from Step 3.

Test Docker:
```bash
docker --version
docker compose version
```

You should see version numbers. Success! 🎉

#### Step 5: Deploy the AI Research Hub

```bash
# Clone the repository
git clone https://github.com/israelsaba/open-webui-stack.git
cd open-webui-stack

# Set up configuration
cd sdk-interface
cp .env.example .env
nano .env
```

You're now in a text editor. You need to add your API keys:

**Getting Your Google API Key:**
1. Open a new browser tab: [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Click **"Create API Key"**
3. Copy the key (starts with `AIza`)
4. In the terminal, paste it after `GOOGLE_API_KEY=`

Optional: Add Anthropic, Grok keys the same way if you have them.

**Save and exit:**
- Press `Ctrl + O` (to save)
- Press `Enter` (to confirm)
- Press `Ctrl + X` (to exit)

```bash
# Go back to root directory
cd ..

# Create necessary volume
docker volume create open-webui

# Start everything!
docker compose up -d

# Wait 30 seconds for everything to start, then check status
docker compose ps
```

You should see three services running: `open-webui`, `sdk-interface`, and `watchtower`.

#### Step 6: Access Your AI Hub

1. In AWS Console, find your instance's **Public IPv4 address** (looks like `3.XXX.XXX.XXX`)
2. Open your browser and go to: `http://YOUR-IP-ADDRESS:8090`
3. Create your account (first user becomes admin)
4. **Important**: Go to Settings → Interface:
   - ❌ **Disable** "Auto-Generate Title"
   - ❌ **Disable** "Auto-Follow-Up Prompts"
5. Go to Settings → Connections → **Add OpenAI Connection**:
   - **API Base URL**: `http://sdk-interface:8060/v1`
   - **API Key**: Leave blank or use any text (auth is handled internally)

**You're done!** 🎉 Try asking Deep Research a question like "Research the latest developments in quantum computing."

#### Step 7: Secure Your Setup (Important!)

Right now, anyone with your IP can access your hub. Let's fix that:

1. In AWS Console → EC2 → **Security Groups**
2. Click on the security group attached to your instance
3. Edit **Inbound Rules**:
   - For the HTTP rule (port 8090), change source from `0.0.0.0/0` to **"My IP"**
   - Keep SSH (port 22) as "My IP" too
4. **Save rules**

Now only your IP address can access the hub!

---

### Path B: Deploy on Your Own Ubuntu Machine

Perfect if you have an old laptop or desktop running Ubuntu!

#### Step 1: Install Ubuntu (If Needed)

**Already have Ubuntu?** Skip to Step 2!

**Need to install Ubuntu?** Follow this excellent guide:
- [Ubuntu Installation Guide](https://ubuntu.com/tutorials/install-ubuntu-desktop)
- We recommend **Ubuntu 22.04 LTS** for stability

**Quick tips:**
- Download Ubuntu from [ubuntu.com/download/desktop](https://ubuntu.com/download/desktop)
- Create a bootable USB using [Rufus](https://rufus.ie/) (Windows) or [Etcher](https://www.balena.io/etcher/) (Mac/Linux)
- Boot from USB and follow the installer (choose "Install Ubuntu")
- Recommended: 30GB+ storage for Ubuntu partition

#### Step 2: Install Docker

Open Terminal (`Ctrl + Alt + T`) and run:

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add your user to docker group
sudo usermod -aG docker $USER

# Install Docker Compose
sudo apt install docker-compose-plugin -y

# Reboot for changes to take effect
sudo reboot
```

After reboot, verify:
```bash
docker --version
docker compose version
```

#### Step 3: Deploy the Stack

```bash
# Install git if not present
sudo apt install git -y

# Clone repository
git clone https://github.com/israelsaba/open-webui-stack.git
cd open-webui-stack

# Set up configuration
cd sdk-interface
cp .env.example .env
nano .env
```

Add your API keys (see Path A, Step 5 for how to get them), then:

```bash
# Save with Ctrl+O, Enter, Ctrl+X
cd ..

# Create volume
docker volume create open-webui

# Start services
docker compose up -d

# Check status
docker compose ps
```

#### Step 4: Access Your Hub

Open Firefox or Chrome and go to: `http://localhost:8090`

Follow the same setup from Path A, Step 6!

---

### Path C: Deploy with VSCode Dev Containers

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

#### Step 2: Clone and Open in Container

```bash
# Clone repository
git clone https://github.com/israelsaba/open-webui-stack.git
cd open-webui-stack
code .
```

In VSCode:
1. Press `Ctrl/Cmd + Shift + P`
2. Type "Dev Containers: Reopen in Container"
3. Wait for container to build (2-5 minutes first time)

#### Step 3: Set Up and Run

In VSCode's integrated terminal:

```bash
cd sdk-interface
cp .env.example .env
# Edit .env with your API keys using VSCode editor

# Run migrations
make setup
cd ..

# Start services
docker compose up -d
```

Access at `http://localhost:8090`

---

## 🎯 For Experts: Advanced Deployment

### Quick Start

```bash
git clone https://github.com/israelsaba/open-webui-stack.git
cd open-webui-stack/sdk-interface
cp .env.example .env
# Configure API keys in .env
docker volume create open-webui
cd .. && docker compose up -d
```

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
