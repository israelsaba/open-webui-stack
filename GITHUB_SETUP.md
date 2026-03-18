# GitHub Repository Setup for CI/CD

This document explains how to configure GitHub repository settings for the testing pipeline.

## ⚠️ Important: API Keys for Testing

**API keys should NOT be stored in GitHub Secrets for tests.** They should be set as local environment variables when testing locally.

The CI/CD pipeline runs tests **without API keys** and will auto-skip any provider-specific integration tests. This is by design for security and to avoid consuming API quotas in CI.

## 📌 Optional GitHub Secrets

Go to your repository → **Settings** → **Secrets and variables** → **Actions** → **New repository secret**

### 🔑 Only for Coverage Upload

| Secret Name | Description | Required | How to Get |
|------------|-------------|----------|------------|
| `CODECOV_TOKEN` | Codecov upload token | No | [codecov.io](https://codecov.io/) after linking repo |

**That's it!** No API keys needed for CI/CD.

### Adding the Codecov Secret (Optional)

1. Click **"New repository secret"**
2. **Name**: `CODECOV_TOKEN`
3. **Secret**: Paste the token from Codecov
4. Click **"Add secret"**

## 🧪 Local Testing with Real APIs

To run integration tests locally with actual API providers:

### Step 1: Set Environment Variables

```bash
# In your terminal or add to ~/.bashrc or ~/.zshrc
export GOOGLE_API_KEY="your-google-api-key-here"
export ANTHROPIC_API_KEY="your-anthropic-api-key-here"
export GROK_API_KEY="your-grok-api-key-here"

# Optional: Override test models
export TEST_MODEL_ANTHROPIC="claude-opus-4-5-20251101"
export TEST_MODEL_GEMINI="gemini-2.0-flash-exp"
```

### Step 2: Run Tests Locally

```bash
cd sdk-interface
make setup          # One-time: install dependencies
make test-cov       # Run tests with coverage
```

Tests will automatically use environment variables and skip any tests for which API keys aren't available.

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

### Expected Results (CI/CD)

**In GitHub Actions (no API keys):**
- ⚠️ Provider-specific integration tests skipped (expected)
- ✅ Basic tests (health check, etc.) pass
- ✅ Linting passes
- ✅ Coverage report generated
- ✅ Coverage uploaded to Codecov (if token configured)

**Locally (with API keys set as env vars):**
- ✅ All integration tests run
- ✅ Provider-specific tests pass (if APIs are working)
- ✅ Full coverage including provider integrations
- 📊 See actual API responses and behaviors

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

### Provider-specific tests are skipped in CI

**Cause**: This is expected behavior - no API keys in CI  
**Solution**: This is correct! Tests auto-skip to avoid consuming API quotas. Run locally with API keys for full integration testing.

### Want to test specific providers in CI?

**Answer**: Don't! API keys should never be in GitHub for security reasons. Instead:
1. Test locally with your API keys before pushing
2. Let CI verify code quality (lint, basic tests)
3. Trust that if basic tests pass, provider integrations work (they're well-tested locally)

### Coverage upload fails

**Cause**: Invalid or missing `CODECOV_TOKEN`  
**Fix**: The workflow continues even if upload fails. Check token in secrets or skip Codecov entirely.

### Workflow not triggering

**Cause**: Workflow file permissions or branch protection  
**Fix**: Go to **Settings** → **Actions** → **General** → ensure actions are allowed

## 📝 Summary Checklist

Before running CI/CD, ensure:

- [ ] Optional: `CODECOV_TOKEN` secret is set (only if you want Codecov integration)
- [ ] Workflow file exists at `.github/workflows/test.yml` ✅ (already committed)
- [ ] Tests directory exists at `sdk-interface/tests/` ✅ (already committed)
- [ ] You've tested locally with `make test-cov` (with your API keys set as env vars)

**Ready!** Push to main or create a PR to trigger the workflow. 🚀

## 🔐 Why No API Keys in CI?

**Security**: API keys in GitHub secrets could be exposed via:
- Malicious PRs that print environment variables
- Compromised GitHub accounts
- Accidental logging

**Cost**: CI/CD runs on every push/PR, consuming API quotas unnecessarily

**Best Practice**: 
- ✅ Test locally with real APIs before committing
- ✅ Use CI for code quality checks (linting, basic tests)
- ✅ Trust that well-tested code works across providers

---

**Need help?** Check [CLAUDE.md](./CLAUDE.md) for development guidance or create an issue on GitHub.
