# Pre-flight check: confirm GitHub Models / OpenAI access BEFORE running the full pipeline.
# Usage: put this file next to your .env, then run:  python check_models_access.py
#
# NOTE: written with zero indented blocks (all bodies on the same line as their
# colon, semicolon-separated) because indentation keeps getting stripped when
# this file is transferred onto the target machine. Uglier, but copy-paste proof.

import os, sys

env_lines = open(".env", encoding="utf-8").read().splitlines() if os.path.exists(".env") else []
[os.environ.setdefault(l.partition("=")[0].strip(), l.partition("=")[2].strip()) for l in env_lines if l.strip() and not l.strip().startswith("#") and "=" in l]

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "").strip()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
MODEL = os.environ.get("PT_MODEL", "gpt-4o")

api_key = GITHUB_TOKEN or OPENAI_API_KEY
base_url = "https://models.inference.ai.azure.com" if GITHUB_TOKEN else None
mode = "GitHub Models" if GITHUB_TOKEN else ("OpenAI (direct)" if OPENAI_API_KEY else None)

if mode is None: print("FAIL: neither GITHUB_TOKEN nor OPENAI_API_KEY is set. Check your .env."); sys.exit(1)

print(f"Testing {mode} — endpoint: {base_url or 'api.openai.com (default)'}, model: {MODEL}")

try: from openai import OpenAI
except ImportError: print("FAIL: 'openai' package not installed. Run: pip install openai"); sys.exit(1)

try: client = OpenAI(api_key=api_key, base_url=base_url)
except Exception as exc: print(f"FAIL: could not initialize client — {type(exc).__name__}: {exc}"); sys.exit(1)

try: resp = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "Reply with just: OK"}], max_tokens=5)
except Exception as exc: msg = str(exc); print(f"FAIL: {type(exc).__name__}: {exc}"); reason = "Auth problem: token invalid/revoked, or missing the 'Models' permission (fine-grained PAT) / models scope (classic PAT)." if ("401" in msg or "Unauthorized" in msg or "Bad credentials" in msg) else ("Network problem: endpoint likely blocked by the Capgemini firewall/proxy." if ("Connection" in msg or "timeout" in msg.lower()) else "Unexpected error — see message above."); print(f"→ {reason}"); sys.exit(1)

print(f"PASS: got a response back — '{resp.choices[0].message.content.strip()}'")
print("You're clear to run the full pipeline.")
