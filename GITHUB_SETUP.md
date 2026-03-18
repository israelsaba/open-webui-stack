# GitHub Repository Setup for CI/CD

This document explains how to configure GitHub Secrets and Variables for the testing pipeline.

## 📌 Required GitHub Secrets

Go to your repository → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

### 🔑 API Keys (Secrets)

| Secret Name | Description | Required | How to Get |
|------------|-------------|----------|------------|
| `ANTHROPIC_API_KEY` | Claude API key | No* | [console.anthropic.com](https://console.anthropic.com/) |
| `GOOGLE_API_KEY` | Gemini/Deep Research API key | Yes | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) |
| `GROK_API_KEY` | xAI Grok API key | No* | [console.x.ai](https://console.x.ai/) |
| `SDK_API_KEY` | Bearer token for authenticated endpoints | No | Any string (e.g., `op_wui_test123`) |
| `CODECOV_TOKEN` | Codecov upload token | No** | [codecov.io](https://codecov.io/) after linking repo |

\* Tests will skip if not provided  
\** Coverage will still work, just won't upload to Codecov

### Adding a Secret

1. Click **"New repository secret"**
2. **Name**: Enter the secret name exactly as shown above (case-sensitive)
3. **Secret**: Paste the API key value
4. Click **"Add secret"**

**Example:**
```
Name: GOOGLE_API_KEY
Secret: AIzaSyABC123...xyz
```

## 📌 Optional GitHub Variables

Go to your repository → **Settings** → **Secrets and variables** → **Actions** → **Variables** tab → **New repository variable**

### 🔧 Configuration (Variables)

| Variable Name | Description | Default | Example |
|--------------|-------------|---------|---------|
| `SDK_BASE_URL` | SDK interface endpoint | `http://localhost:8060` | `http://192.168.2.4:8060` |
| `TEST_MODEL_ANTHROPIC` | Override default Anthropic test model | `claude-sonnet-4-5-20250929` | `claude-opus-4-5-20251101` |
| `TEST_MODEL_GEMINI` | Override default Gemini test model | `gemini-2.0-flash-exp` | `gemini-1.5-pro-latest` |
| `TEST_MODEL_GEMINI_DEEP_RESEARCH` | Override default Deep Research model | `deep-research-pro-preview-12-2025` | Same |
| `TEST_MODEL_GROK` | Override default Grok test model | `grok-code-fast-1` | `grok-2-vision-1212` |

### When to Use Variables

- **Testing against specific model versions**: Set `TEST_MODEL_*` variables
- **Testing against deployed instance**: Set `SDK_BASE_URL` to your server
- **Most cases**: Leave blank to use sensible defaults

## 🚀 Setting Up Codecov (Optional)

Codecov provides nice coverage badges and PR comments.

### Step 1: Sign up for Codecov

1. Go to [codecov.io](https://codecov.io/)
2. Click **"Sign up with GitHub"**
3. Authorize Codecov to access your repositories

### Step 2: Add Repository

1. In Codecov dashboard, click **"Add new repository"**
2. Find and select `israelsaba/open-webui-stack`
3. Copy the **Upload Token** shown

### Step 3: Add Token to GitHub

1. Go to your repo → **Settings** → **Secrets and variables** → **Actions**
2. Click **"New repository secret"**
3. Name: `CODECOV_TOKEN`
4. Secret: Paste the upload token from Codecov
5. Click **"Add secret"**

### Step 4: Add Badge to README (Optional)

Add this to the top of your README.md:

```markdown
[![codecov](https://codecov.io/gh/israelsaba/open-webui-stack/branch/main/graph/badge.svg)](https://codecov.io/gh/israelsaba/open-webui-stack)
```

## ✅ Verification

### Test the Workflow

1. Go to your repository → **Actions** tab
2. Click on **"Tests"** workflow
3. Click **"Run workflow"** dropdown → **"Run workflow"**
4. Wait for completion (~2-5 minutes)

### Expected Results

**With all API keys:**
- ✅ All tests pass
- ✅ Coverage report generated
- ✅ Coverage uploaded to Codecov (if token provided)

**With only Google API key:**
- ⚠️ Anthropic and Grok tests skipped
- ✅ Gemini and Deep Research tests pass
- ✅ Coverage report generated

**With no API keys:**
- ⚠️ Provider-specific tests skipped
- ✅ Basic tests (health, models endpoint) pass
- ✅ Coverage report generated (but lower %)

## 🔒 Security Best Practices

### ✅ Do:
- Store API keys as **Secrets**, never as Variables
- Use separate API keys for CI/CD (not production keys)
- Set rate limits on test API keys if provider allows
- Rotate API keys quarterly
- Review **Settings** → **Actions** → **General** → ensure "Read and write permissions"

### ❌ Don't:
- Never commit API keys to `.env` files
- Never echo or log secret values in workflows
- Never share Codecov tokens publicly
- Never use production API keys for testing

## 📊 Understanding the CI/CD Pipeline

### What Happens on Every Push/PR

1. **Setup**: Python 3.12 installed, dependencies cached
2. **Lint**: Ruff checks code style (continues even if warnings)
3. **Test**: Pytest runs all tests with coverage
   - Skips tests marked `@pytest.mark.slow`
   - Auto-skips tests if API keys not available
4. **Coverage**: Generates XML and HTML reports
5. **Upload**: Sends coverage to Codecov (if token present)
6. **Artifact**: Uploads HTML coverage report (downloadable for 7 days)
7. **PR Comment**: Posts coverage summary on pull requests

### Viewing Results

**In GitHub:**
- Go to **Actions** tab → Click on workflow run
- See test results in **"Run tests with coverage"** step
- Download coverage HTML from **Artifacts** section

**In Codecov (if setup):**
- Coverage graphs over time
- File-by-file coverage breakdown
- Pull request coverage diff

## 🛠️ Troubleshooting

### "API key not configured" errors

**Cause**: Secret not set or named incorrectly  
**Fix**: Double-check secret names (case-sensitive) in **Settings** → **Secrets and variables** → **Actions**

### Tests timing out

**Cause**: Deep Research tests can take 30-60 seconds  
**Fix**: Already handled - slow tests are marked and skipped in CI

### Coverage upload fails

**Cause**: Invalid or missing `CODECOV_TOKEN`  
**Fix**: The workflow continues even if upload fails. Check token in secrets.

### Workflow not triggering

**Cause**: Workflow file permissions or branch protection  
**Fix**: Go to **Settings** → **Actions** → **General** → ensure actions are allowed

## 📝 Summary Checklist

Before running CI/CD, ensure:

- [ ] `GOOGLE_API_KEY` secret is set (minimum requirement)
- [ ] Optional: `ANTHROPIC_API_KEY` secret is set
- [ ] Optional: `GROK_API_KEY` secret is set
- [ ] Optional: `CODECOV_TOKEN` secret is set
- [ ] Workflow file exists at `.github/workflows/test.yml`
- [ ] Tests directory exists at `sdk-interface/tests/`
- [ ] You've tested locally with `make test-cov`

**Ready!** Push to main or create a PR to trigger the workflow. 🚀

---

**Need help?** Check [CLAUDE.md](./CLAUDE.md) for development guidance or create an issue on GitHub.
