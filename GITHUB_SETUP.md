# GitHub Repository Setup for CI/CD

This document explains how to configure GitHub repository settings for the testing pipeline.

## 🎭 Mock Testing Approach

**CI/CD uses mocked API responses** - no real API keys needed!

The testing infrastructure supports two modes:

- **Mock Mode** (`TEST_MODE=mock`): Uses respx to mock all external API calls - perfect for CI/CD
- **Real Mode** (`TEST_MODE=real`): Uses actual provider APIs - for local integration testing

This approach provides:
- ✅ **Security**: No API keys in GitHub
- ✅ **Speed**: Fast tests without network calls
- ✅ **Reliability**: Tests don't fail due to API rate limits or outages
- ✅ **Cost**: Zero API quota consumption in CI

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

## 🧪 Local Testing Options

### Option 1: Mock Mode (Default - Recommended)

Test with mocked responses (fast, no API keys needed):

```bash
cd sdk-interface
cp .env.test.example .env.test
# Edit .env.test and set TEST_MODE=mock (or leave as default)
make test-cov
```

### Option 2: Real API Mode

Test against actual provider APIs:

```bash
cd sdk-interface
cp .env.test.example .env.test
```

Edit `.env.test`:
```bash
TEST_MODE=real
GOOGLE_API_KEY=your-google-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
GROK_API_KEY=your-grok-api-key-here
```

Then run:
```bash
make test-cov
```

### Option 3: Environment Variables Only

Skip `.env.test` and use shell environment variables:

```bash
export TEST_MODE=real
export GOOGLE_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
export GROK_API_KEY="your-key"
make test-cov
```

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

**In GitHub Actions (Mock Mode):**
- ✅ All tests run (including provider integration tests)
- ✅ All tests pass (using mocked responses)
- ✅ Linting passes
- ✅ Full coverage report generated
- ✅ Coverage uploaded to Codecov (if token configured)
- ⚡ Fast execution (~1-2 minutes)

**Locally in Mock Mode (TEST_MODE=mock):**
- ✅ All tests run with mocked responses
- ✅ Fast execution
- ✅ No API keys needed
- ✅ Perfect for quick validation

**Locally in Real Mode (TEST_MODE=real):**
- ✅ All integration tests run against real APIs
- ✅ Provider-specific tests pass (if APIs are working)
- ✅ Full coverage including actual provider integrations
- 📊 See actual API responses and behaviors
- ⏱️ Slower execution due to network calls

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

### Tests pass in CI but fail locally

**Cause**: CI uses mock mode, local might be using real mode with invalid/expired API keys  
**Solution**: Set `TEST_MODE=mock` in your `.env.test` for consistent behavior

### Want to verify real API integration?

**Answer**: Yes! Use real mode locally:
1. Set `TEST_MODE=real` in `.env.test`
2. Add your API keys
3. Run `make test-cov`
4. Commit confident that both mocked and real tests pass

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

## 🎭 Why Mock Testing?

**Security**: No API keys in GitHub = No exposure risk
- Malicious PRs can't steal keys
- No accidental logging of secrets
- Reduced attack surface

**Speed**: Mocked responses are instant
- No network latency
- No rate limiting
- Predictable test duration

**Cost**: Zero API consumption
- Free test runs
- Save quotas for development
- No surprise bills

**Reliability**: Tests never fail due to:
- API outages
- Rate limits
- Network issues
- Provider changes

**Best Practice**:
- ✅ CI uses mocks for speed and security
- ✅ Local real-mode testing before major releases
- ✅ Mocks based on actual API responses
- ✅ Update mocks when provider APIs change

---

**Need help?** Check [CLAUDE.md](./CLAUDE.md) for development guidance or create an issue on GitHub.
