"""Load pipeline configuration from config.yaml (with optional local override)."""
import os
import yaml
from pathlib import Path

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

    # Allow env var override for OpenAI key
    env_key = os.environ.get("OPENAI_API_KEY", "")
    if env_key:
        cfg.setdefault("openai", {})["api_key"] = env_key

    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result
