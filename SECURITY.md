# Security Policy

## Supported Versions

Security fixes are applied to `main` and the latest published release. Use the
latest release when possible.

## Reporting A Vulnerability

Do not open a public issue for a vulnerability. Use GitHub's private security
advisory reporting for this repository when available. If it is unavailable,
contact the repository maintainers privately through the account listed on the
repository before sharing any details.

Include the affected commit or version, deployment mode, reproduction steps,
impact, and a minimal proof of concept. Redact credentials, personal data,
prompts, and provider responses.

## Security Boundaries

- Keep `.env` files and provider keys outside version control.
- Do not expose the SDK port publicly without authentication and TLS.
- Review agent tools and permissions before sending requests to this service.
- Treat `SKILL.md`, agent instructions, prompts, and tool schemas as untrusted
  input until reviewed.
