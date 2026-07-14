# Security Policy

## Supported Versions

FlintAI SDK is currently in active development. Security fixes are applied to
the latest released version on PyPI.

| Version | Supported |
| ------- | --------- |
| 0.2.x   | ✅        |
| < 0.2   | ❌        |

## Reporting a Vulnerability

Please **do not** report security vulnerabilities through public GitHub issues,
discussions, or pull requests.

Instead, report them privately using GitHub's
[private vulnerability reporting](https://github.com/sandbox-quantum/flintai-sdk-py/security/advisories/new)
feature (the "Report a vulnerability" button under the repository's **Security**
tab).

When reporting, please include as much of the following as possible:

- A description of the vulnerability and its potential impact
- Steps to reproduce or a proof of concept
- Affected version(s)
- Any suggested remediation

We will acknowledge receipt of your report and work with you on a coordinated
disclosure. Please give us a reasonable amount of time to address the issue
before any public disclosure.

## Scope

This policy covers the `flintai-sdk-py` package in this repository. The SDK
routes LLM traffic through the FlintAI guardrails proxy and handles API keys via
environment variables and request headers; reports related to credential
handling, request routing, or dependency vulnerabilities are especially
welcome.
