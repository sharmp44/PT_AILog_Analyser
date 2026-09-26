"""Shared LLM client factory.

Centralises how the Pattern and Causal agents obtain an OpenAI-compatible
client, so switching providers (direct OpenAI / GitHub Models -> client-hosted
Azure OpenAI, KDD Section 6.1 "Option A") is a config/env change, not a code
change in the agents themselves.
"""
from __future__ import annotations


def get_llm_client(llm_cfg: dict, *, timeout: float = 60.0, max_retries: int = 5):
    """Return (client, model_name) for the configured LLM provider.

    llm_cfg keys:
      provider          "openai" (default) or "azure"
      api_key           required for every provider
      model             model name - used as-is for "openai" (and GitHub Models / Groq)
      base_url          optional, OpenAI-compatible endpoint override (GitHub Models, Groq, ...)
      azure_endpoint    required when provider == "azure", e.g. https://<resource>.openai.azure.com
      azure_deployment  required when provider == "azure" - the *deployment name* created in Azure
                        (Azure OpenAI addresses a deployment, not a bare model name)
      api_version       required when provider == "azure", e.g. "2024-10-21"
    """
    provider = (llm_cfg.get("provider") or "openai").lower()
    api_key = llm_cfg.get("api_key", "").strip()

    if provider == "azure":
        from openai import AzureOpenAI

        azure_endpoint = llm_cfg.get("azure_endpoint")
        azure_deployment = llm_cfg.get("azure_deployment")
        api_version = llm_cfg.get("api_version")
        missing = [
            name
            for name, val in (
                ("azure_endpoint", azure_endpoint),
                ("azure_deployment", azure_deployment),
                ("api_version", api_version),
            )
            if not val
        ]
        if missing:
            raise ValueError(
                f"openai.provider is 'azure' but missing config: {', '.join(missing)}. "
                "Set AZURE_OPENAI_ENDPOINT / AZURE_OPENAI_DEPLOYMENT / AZURE_OPENAI_API_VERSION "
                "env vars (or the equivalent config.yaml keys)."
            )

        client = AzureOpenAI(
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            timeout=timeout,
            max_retries=max_retries,
        )
        return client, azure_deployment

    from openai import OpenAI

    base_url = llm_cfg.get("base_url", "")
    client = OpenAI(
        api_key=api_key,
        timeout=timeout,
        max_retries=max_retries,
        **({"base_url": base_url} if base_url else {}),
    )
    return client, llm_cfg.get("model", "gpt-4o")
