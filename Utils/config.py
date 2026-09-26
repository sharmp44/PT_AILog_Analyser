"""Load pipeline configuration from config.yaml (with optional local override)."""
import os
import yaml
from pathlib import Path

try:
    from dotenv import load_dotenv
    # Always load from project root (parent of Utils/), regardless of cwd
    _ENV_FILE = Path(__file__).parent.parent / ".env"
    load_dotenv(dotenv_path=_ENV_FILE)
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

    # Allow env var override for API key / provider.
    # Priority: Azure OpenAI (KDD Section 6.1, Option A) > GitHub Models > direct OpenAI.
    azure_key         = os.environ.get("AZURE_OPENAI_API_KEY", "")
    azure_endpoint    = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    azure_deployment  = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
    azure_api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "")
    github_token = os.environ.get("GITHUB_TOKEN", "")
    openai_key   = os.environ.get("OPENAI_API_KEY", "")

    # NOTE: priority is Azure > direct OpenAI > GitHub Models. GitHub Models was
    # moved to lowest priority because it has been reported non-functional since
    # Aug 2026 - if both OPENAI_API_KEY and a leftover GITHUB_TOKEN are present
    # in .env, we should not silently route through the broken endpoint.
    if azure_key and azure_endpoint:
        # Option A — client-hosted Azure OpenAI inside the client network.
        oc = cfg.setdefault("openai", {})
        oc["provider"]         = "azure"
        oc["api_key"]          = azure_key
        oc["azure_endpoint"]   = azure_endpoint
        oc["azure_deployment"] = azure_deployment or oc.get("azure_deployment", "")
        oc["api_version"]      = azure_api_version or oc.get("api_version", "2024-10-21")
        oc["base_url"] = ""
    elif openai_key:
        oc = cfg.setdefault("openai", {})
        oc["provider"] = "openai"
        oc["api_key"]  = openai_key
        oc["base_url"] = ""   # ensure OpenAI endpoint when using OpenAI key
    elif github_token:
        oc = cfg.setdefault("openai", {})
        oc["provider"] = "openai"
        oc["api_key"]  = github_token
        oc["base_url"] = "https://models.github.ai/inference"
        # GitHub Models' new endpoint requires publisher-prefixed model names
        # (e.g. "openai/gpt-4o" instead of "gpt-4o").
        model = oc.get("model", "")
        if model and "/" not in model:
            oc["model"] = f"openai/{model}"

    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result
