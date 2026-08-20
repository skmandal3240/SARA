# Security Policy

## Supported versions

SARA is in active pre-1.0 development. Only the latest `main` branch receives security fixes.

| Version | Supported |
|---------|-----------|
| `main` (latest) | ✅ |
| older commits | ❌ |

## Reporting a vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Email: skmandal3240@gmail.com (subject: `[SARA] security`).

Include:
- What the vulnerability is and how you found it.
- A minimal reproducer (code, config, or steps).
- Impact: data leak, model bypass, sandbox escape, etc.

We will respond within 72 hours. If confirmed, we will:
1. Patch and ship a fix on `main`.
2. Credit you in the release notes (unless you prefer to stay anonymous).

## Threat model

SARA is an **edge-first** multimodal transformer with a sandboxed agent runtime. The threat model we care about:

- **Sandbox escape** — `python_exec` / `code_run` must stay AST-guarded and never expose `os`, `subprocess`, `eval`, `exec` directly.
- **Data exfiltration** — raw user data (camera frames, mic, vault) must not leave the device without an explicit `cloud` grant.
- **Prompt injection via assets** — images, audio, web pages, and tool observations are untrusted. Tool previews are mandatory before execution.
- **Privilege escalation via grants** — grants must be preview-first, time-bounded, and audit-logged.

Out of scope (for now): denial of service on the device, model-weight exfiltration via timing attacks, kernel-level attacks (the OS is the trust boundary).
