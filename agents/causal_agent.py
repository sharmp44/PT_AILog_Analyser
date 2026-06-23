"""
Causal Agent
------------
Synthesises findings from all other agents into a causal chain.

Uses OpenAI GPT-4o to:
  1. Read all agent findings (anomaly, trend, threshold, pattern).
  2. Identify which signal appeared first (leading indicator).
  3. Build a cause → effect chain.
  4. Return ranked root cause hypotheses with evidence citations.

Returns a CausalAnalysis dict.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

from Utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class CausalAnalysis:
    leading_indicator:  str
    cause_effect_chain: list[str]
    hypotheses: list[dict]   # [{rank, hypothesis, evidence, confidence}]
    summary:    str
    model_used: str

    def to_dict(self) -> dict:
        return asdict(self)


_FALLBACK_ANALYSIS = CausalAnalysis(
    leading_indicator="Unable to determine (no OpenAI key configured)",
    cause_effect_chain=["See individual agent findings for details"],
    hypotheses=[],
    summary="OpenAI API key not configured. Causal analysis skipped. "
            "Review anomaly, trend, threshold, and pattern findings manually.",
    model_used="none",
)


def run(
    anomaly_findings:   list[dict],
    trend_findings:     list[dict],
    threshold_findings: list[dict],
    pattern_findings:   list[dict],
    cfg: dict | None = None,
) -> dict:
    """Run causal reasoning over all agent findings."""
    cfg = cfg or {}
    llm_cfg  = cfg.get("openai", {})
    api_key  = llm_cfg.get("api_key", "").strip()
    base_url = llm_cfg.get("base_url", "")   # e.g. https://api.groq.com/openai/v1

    if not api_key:
        log.warning("[causal_agent] No API key – returning fallback analysis")
        return _FALLBACK_ANALYSIS.to_dict()

    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=api_key,
            timeout=60.0,
            max_retries=5,
            **({ "base_url": base_url } if base_url else {}),
        )
    except ImportError:
        log.warning("[causal_agent] openai not installed – returning fallback")
        return _FALLBACK_ANALYSIS.to_dict()

    # ── Build the evidence packet ────────────────────────────────────────────
    evidence = {
        "anomaly_findings":   anomaly_findings[:30],   # cap to avoid token overflow
        "trend_findings":     trend_findings,
        "threshold_findings": threshold_findings,
        "pattern_findings":   pattern_findings,
    }
    evidence_json = json.dumps(evidence, indent=2, default=str)

    prompt = f"""You are a senior performance engineering expert analysing the results of an AI-driven load test log analysis.

Below is a JSON object containing findings from four specialist agents:
- anomaly_findings: statistical outliers in metrics
- trend_findings: monotonic degradation patterns detected during steady state
- threshold_findings: SLA and rule-based breaches
- pattern_findings: log text patterns (errors, exceptions, timeouts, etc.)

Your task:
1. Identify the LEADING INDICATOR – which metric or event appeared first and likely triggered the cascade.
2. Build a CAUSE → EFFECT chain (a list of steps, each a short sentence).
3. Produce up to 5 ranked ROOT CAUSE HYPOTHESES. For each:
   - rank (1 = most likely)
   - hypothesis (1 sentence)
   - evidence (comma-separated list of finding types/metrics that support it)
   - confidence ("High" | "Medium" | "Low")
4. Write a 3–5 sentence EXECUTIVE SUMMARY suitable for a test report.

Return ONLY valid JSON with this schema:
{{
  "leading_indicator": "<metric or event name>",
  "cause_effect_chain": ["step 1", "step 2", ...],
  "hypotheses": [
    {{"rank": 1, "hypothesis": "...", "evidence": "...", "confidence": "High"}},
    ...
  ],
  "summary": "<3-5 sentence narrative>"
}}

Evidence:
{evidence_json}
"""

    model = cfg.get("openai", {}).get("model", "gpt-4o")
    log.info(f"[causal_agent] Calling {model} for causal analysis …")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=cfg.get("openai", {}).get("max_tokens", 4096),
            temperature=0.2,
        )
        raw = response.choices[0].message.content.strip()
        data = json.loads(raw)
    except Exception as exc:
        log.error(f"[causal_agent] OpenAI call failed: {exc}")
        return _FALLBACK_ANALYSIS.to_dict()

    result = CausalAnalysis(
        leading_indicator  = data.get("leading_indicator", "unknown"),
        cause_effect_chain = data.get("cause_effect_chain", []),
        hypotheses         = data.get("hypotheses", []),
        summary            = data.get("summary", ""),
        model_used         = model,
    )
    log.info(f"[causal_agent] {len(result.hypotheses)} hypotheses generated")
    return result.to_dict()
