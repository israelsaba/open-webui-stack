# 🤖 Instructions for Autonomous AI Agents

> This document is for autonomous AI agents (AutoGPT, AgentGPT, BabyAGI, etc.) that are deploying, maintaining, or monitoring this AI Research Hub stack.

## 🎯 Agent Mission

You are an autonomous agent responsible for deploying and maintaining an AI Research Hub that aggregates Google Deep Research, Anthropic Claude, Google Gemini, and xAI Grok models through a unified OpenAI-compatible API.

## 📋 Pre-Flight Checklist

Before starting any deployment, verify:

- [ ] You have SSH access or cloud provider credentials
- [ ] You have API keys for at least Google (required for Deep Research)
- [ ] Target environment has Docker and Docker Compose installed
- [ ] You have permissions to create/modify files and run containers
- [ ] You understand the user's deployment target (AWS, local, etc.)

## 🚀 Standard Deployment Procedure

### Phase 1: Environment Preparation

**1.1 System Requirements Check**
```bash
# Check Docker
docker --version  # Should be 20.10+
docker compose version  # Should be 2.0+

# Check available resources
free -h  # At least 4GB RAM recommended
df -h    # At least 20GB free space
```

**Expected Output:** Version numbers and sufficient resources

**1.2 Install Docker (if needed)**
```bash
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Verify
newgrp docker
docker run hello-world
```

**Expected Output:** "Hello from Docker!" message

### Phase 2: Code Deployment

**2.1 Clone Repository**
```bash
cd /opt  # or user's preferred directory
git clone https://github.com/israelsaba/open-webui-stack.git
cd open-webui-stack
```

**Expected Output:** Repository cloned successfully

**2.2 Configure Environment**
```bash
cd sdk-interface
cp .env.example .env

# Use secure method to set API keys
# NEVER log or echo actual API keys
cat > .env <<EOF
GOOGLE_API_KEY=${GOOGLE_API_KEY}
ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:-}
GROK_API_KEY=${GROK_API_KEY:-}
API_KEYS=admin:$(openssl rand -hex 32 | base64 | head -c 32)
LOG_LEVEL=info
EOF

# Verify (without showing secrets)
grep -q "GOOGLE_API_KEY=" .env && echo "✓ Google API key set"
```

**Expected Output:** Confirmation that keys are set

**2.3 Set Up Root Environment**
```bash
cd ..
cat > .env <<EOF
IS_EXTERNAL_OI_VOLUME=false
SDK_PORTS=8060:8060
EOF
```

### Phase 3: Database Setup

**3.1 Run Migrations**
```bash
cd sdk-interface

# Install dependencies if running migrations locally
python3 -m venv .venv
source .venv/bin/activate
pip install yoyo-migrations

# Apply migrations
yoyo apply --database sqlite:///data/db.sqlite3 migrations/

# Verify
sqlite3 data/db.sqlite3 "SELECT name FROM sqlite_master WHERE type='table';"
```

**Expected Output:** Tables including `research_hashes` should be listed

### Phase 4: Container Deployment

**4.1 Create Docker Volume**
```bash
docker volume create open-webui
docker volume ls | grep open-webui
```

**Expected Output:** `open-webui` volume listed

**4.2 Start Services**
```bash
docker compose up -d

# Wait for services to be healthy
sleep 30

# Verify all services are running
docker compose ps
```

**Expected Output:**
```
NAME            STATUS                   PORTS
open-webui      Up (healthy)             0.0.0.0:8090->8080/tcp
sdk-interface   Up (healthy)             8060/tcp
watchtower      Up (healthy)             8080/tcp
```

**4.3 Health Checks**
```bash
# Check SDK interface
curl -f http://localhost:8060/health || echo "SDK interface not ready"

# Check Open WebUI
curl -f http://localhost:8090/ || echo "Open WebUI not ready"

# Check logs for errors
docker compose logs --tail=50 | grep -i error
```

**Expected Output:** Both health checks return 200, no critical errors in logs

### Phase 5: Validation

**5.1 Test Model Listing**
```bash
curl -s http://localhost:8060/v1/models | jq '.data[].id' | head -10
```

**Expected Output:** List of available models from all configured providers

**5.2 Test Completion (Non-Streaming)**
```bash
curl -s -X POST http://localhost:8060/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-2.0-flash-exp",
    "messages": [{"role": "user", "content": "Say test"}],
    "stream": false,
    "max_tokens": 10
  }' | jq '.choices[0].message.content'
```

**Expected Output:** JSON response with message content

**5.3 Test Deep Research (Streaming)**
```bash
curl -N -X POST http://localhost:8060/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deep-research-pro-preview-12-2025",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "stream": true,
    "max_tokens": 100
  }' | head -20
```

**Expected Output:** SSE stream with `data:` prefixed JSON chunks

### Phase 6: Security Hardening

**6.1 Configure Firewall (if applicable)**
```bash
# Ubuntu/Debian with ufw
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 8090/tcp  # Open WebUI
sudo ufw deny 8060/tcp   # Block external access to SDK interface
sudo ufw enable
sudo ufw status
```

**6.2 Set Up Reverse Proxy (Production)**
```bash
# Install nginx
sudo apt install nginx certbot python3-certbot-nginx -y

# Configure nginx (example)
cat > /etc/nginx/sites-available/ai-hub <<'EOF'
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8090;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
EOF

ln -s /etc/nginx/sites-available/ai-hub /etc/nginx/sites-enabled/
nginx -t && systemctl reload nginx

# SSL certificate
certbot --nginx -d your-domain.com
```

### Phase 7: Monitoring Setup

**7.1 Set Up Log Rotation**
```bash
cat > /etc/logrotate.d/docker-compose <<'EOF'
/var/lib/docker/containers/*/*.log {
    rotate 7
    daily
    compress
    missingok
    delaycompress
    copytruncate
}
EOF
```

**7.2 Create Health Check Script**
```bash
cat > /opt/open-webui-stack/healthcheck.sh <<'EOF'
#!/bin/bash
set -e

# Check services are running
docker compose ps | grep -q "Up (healthy)" || exit 1

# Check SDK interface responds
curl -f -s http://localhost:8060/health > /dev/null || exit 1

# Check Open WebUI responds
curl -f -s http://localhost:8090/ > /dev/null || exit 1

echo "All health checks passed"
EOF

chmod +x /opt/open-webui-stack/healthcheck.sh

# Test it
/opt/open-webui-stack/healthcheck.sh
```

**7.3 Set Up Cron Job for Monitoring**
```bash
(crontab -l 2>/dev/null; echo "*/5 * * * * /opt/open-webui-stack/healthcheck.sh || echo 'Health check failed' | mail -s 'AI Hub Alert' admin@example.com") | crontab -
```

## 🔧 Maintenance Procedures

### Updating the Stack

**Update containers (handled by Watchtower automatically):**
```bash
# Manual update if needed
docker compose pull
docker compose up -d
docker compose ps
```

### Backup Procedures

**Backup SQLite database:**
```bash
mkdir -p /opt/backups
DATE=$(date +%Y%m%d_%H%M%S)
cp /opt/open-webui-stack/sdk-interface/data/db.sqlite3 /opt/backups/db_$DATE.sqlite3

# Backup Open WebUI data
docker run --rm -v open-webui:/data -v /opt/backups:/backup alpine tar czf /backup/open-webui_$DATE.tar.gz -C /data .
```

**Restore from backup:**
```bash
# Restore SQLite
cp /opt/backups/db_YYYYMMDD_HHMMSS.sqlite3 /opt/open-webui-stack/sdk-interface/data/db.sqlite3

# Restore Open WebUI data
docker run --rm -v open-webui:/data -v /opt/backups:/backup alpine sh -c "cd /data && tar xzf /backup/open-webui_YYYYMMDD_HHMMSS.tar.gz"

# Restart services
docker compose restart
```

### Log Management

**View logs:**
```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f sdk-interface

# Last 100 lines with timestamps
docker compose logs --tail=100 -t sdk-interface

# Errors only
docker compose logs | grep -i error
```

**Clear logs (when disk space is low):**
```bash
# WARNING: This removes all logs
docker compose down
rm -rf /var/lib/docker/containers/*/*.log
docker compose up -d
```

## 🚨 Troubleshooting Decision Tree

### Issue: Services won't start

**Step 1: Check Docker**
```bash
systemctl status docker
docker ps
```
- If Docker is down: `systemctl start docker`
- If command fails: Reinstall Docker

**Step 2: Check Docker Compose file**
```bash
docker compose config
```
- If errors: Fix syntax in docker-compose.yml
- If environment variables missing: Check .env files

**Step 3: Check logs**
```bash
docker compose logs
```
- Act based on specific error messages

### Issue: Cannot connect to services

**Step 1: Verify services are running**
```bash
docker compose ps
```
- If not all healthy: Check logs
- If ports not mapped: Check docker-compose.yml ports section

**Step 2: Check network connectivity**
```bash
docker network ls | grep open-webui-net
docker network inspect open-webui-net
```
- If network missing: `docker compose down && docker compose up -d`

**Step 3: Check firewall**
```bash
sudo ufw status
netstat -tlnp | grep -E '8060|8090'
```
- Adjust firewall rules as needed

### Issue: API errors or failures

**Step 1: Verify API keys**
```bash
# Check keys are set (without revealing them)
cd /opt/open-webui-stack/sdk-interface
grep -q "GOOGLE_API_KEY=." .env && echo "Google key set" || echo "Google key MISSING"
```
- If missing: Add API keys to .env
- Restart: `docker compose restart sdk-interface`

**Step 2: Test provider APIs directly**
```bash
# Test Google API
curl -s "https://generativelanguage.googleapis.com/v1beta/models?key=$GOOGLE_API_KEY" | jq '.models[0].name'
```
- If fails: API key invalid or quota exceeded
- Check provider dashboards for status

**Step 3: Check rate limits**
```bash
docker compose logs sdk-interface | grep -i "rate limit\|quota\|429"
```
- If rate limited: Wait or upgrade plan
- Deep Research: 1 request/minute limit

## 📊 Monitoring Metrics

### Key Metrics to Track

**1. System Resources:**
```bash
# CPU and Memory
docker stats --no-stream

# Disk usage
df -h /var/lib/docker
du -sh /opt/open-webui-stack/sdk-interface/data
```

**2. Service Health:**
```bash
# Container uptime
docker ps --format "table {{.Names}}\t{{.Status}}"

# Recent restarts
docker events --since 24h --filter 'event=restart'
```

**3. API Usage:**
```bash
# Request count (approximate from logs)
docker compose logs sdk-interface | grep "POST /v1/chat/completions" | wc -l

# Error rate
docker compose logs sdk-interface | grep ERROR | wc -l
```

### Alerting Conditions

**Critical Alerts:**
- Any container in unhealthy state for >5 minutes
- SDK interface returning 500 errors consistently
- Disk space <10% free
- Memory usage >90%

**Warning Alerts:**
- Container restarts >3 times in 1 hour
- Response time >5 seconds average
- Disk space <20% free

## 🔐 Security Checklist

- [ ] API keys stored securely (not in version control)
- [ ] Firewall configured to block unnecessary ports
- [ ] SSL/TLS enabled for production (HTTPS)
- [ ] Regular backups scheduled
- [ ] Log rotation configured
- [ ] Watchtower enabled for automatic updates
- [ ] No default passwords in use
- [ ] SSH key-based auth only (disable password auth)
- [ ] Regular security updates applied

## 📝 Reporting Template

After deployment, report using this format:

```
## Deployment Report

**Date:** YYYY-MM-DD HH:MM UTC
**Environment:** [AWS/Local/Other]
**Agent:** [Your agent identifier]

### Deployment Status: [SUCCESS/PARTIAL/FAILED]

**Services Running:**
- open-webui: [UP/DOWN]
- sdk-interface: [UP/DOWN]
- watchtower: [UP/DOWN]

**Health Checks:**
- Models endpoint: [PASS/FAIL]
- Completion endpoint: [PASS/FAIL]
- Deep Research: [PASS/FAIL]

**Configuration:**
- Google API configured: [YES/NO]
- Anthropic API configured: [YES/NO]
- Grok API configured: [YES/NO]

**Access URLs:**
- Web UI: http://[IP/DOMAIN]:8090
- API: http://[IP/DOMAIN]:8060 (internal only)

**Issues Encountered:**
[List any issues and how they were resolved]

**Next Steps:**
[Any manual steps required by human operator]
```

## 🔗 Quick Reference

**Common Paths:**
- Repository: `/opt/open-webui-stack/`
- Configuration: `/opt/open-webui-stack/sdk-interface/.env`
- Database: `/opt/open-webui-stack/sdk-interface/data/db.sqlite3`
- Logs: `docker compose logs`

**Common Commands:**
- Start: `docker compose up -d`
- Stop: `docker compose down`
- Restart: `docker compose restart`
- Logs: `docker compose logs -f`
- Status: `docker compose ps`
- Update: `docker compose pull && docker compose up -d`

**Emergency Commands:**
- Force restart: `docker compose down && docker compose up -d --force-recreate`
- Clear all data: `docker compose down -v` (WARNING: Deletes volumes!)
- Reset SDK interface: `docker compose restart sdk-interface`

---

**Agent Directive**: Always prioritize user data safety. When in doubt, create a backup before making changes. Report all actions clearly.
