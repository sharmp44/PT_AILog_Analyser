"""Load pipeline configuration from config.yaml (with optional local override)."""
import os
import yaml
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()          # loads .env from project root automatically
except ImportError:
    pass

_DEFAULT_CFG = Path(__file__).parent.parent / "config.yaml"
_LOCAL_CFG   = Path(__file__).parent.parent / "config.local.yaml"


def load_config(path: Path | None = None) -> dict:
    cfg_path = path or _DEFAULT_CFG
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Merge local overrides if they exist
    if _LOCAL_CFG.exists():
        with open(_LOCAL_CFG) as f:
            local = yaml.safe_load(f) or {}
        cfg = _deep_merge(cfg, local)

    # Allow env var override for API key.
    # GITHUB_TOKEN takes priority → switches to GitHub Models automatically.
    # Falls back to OPENAI_API_KEY for the default OpenAI endpoint.
    github_token = os.environ.get("GITHUB_TOKEN", "")
    openai_key   = os.environ.get("OPENAI_API_KEY", "")

    if github_token:
        cfg.setdefault("openai", {})["api_key"]  = github_token
        cfg["openai"]["base_url"] = "https://models.inference.ai.azure.com"
    elif openai_key:
        cfg.setdefault("openai", {})["api_key"] = openai_key
        cfg["openai"]["base_url"] = ""   # ensure OpenAI endpoint when using OpenAI key

    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result
