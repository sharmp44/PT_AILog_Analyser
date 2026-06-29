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
import re
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

    prompt = f"""You are a performance testing and engineering SME with 15+ years of hands-on experience \
across enterprise-scale load testing, application performance tuning, capacity planning, and production incident RCA. \
Your background spans .NET, Java/JVM, IIS, SQL Server, Oracle, Windows infrastructure, \
and tools including LoadRunner, JMeter, Dynatrace, AppDynamics, and Windows PerfMon (.BLG). \
You have led performance engineering for large-scale banking, e-commerce, and telco programmes \
and have signed off (or rejected) go-live decisions based on exactly this type of evidence.

You are performing the final causal synthesis across four streams of automated analysis from a load test run:
- anomaly_findings: statistical outliers detected in time-series metrics
- trend_findings: monotonic degradation trends observed during steady-state phases
- threshold_findings: SLA and rule-based breaches (response time, error rate, resource ceilings)
- pattern_findings: log-level error and warning patterns (exceptions, timeouts, OOM, GC, etc.)

Think like an SME who must stand up in a post-test review and explain exactly what went wrong and why. \
Cross-correlate signals across all four streams — a timeout in pattern_findings should be checked against \
thread pool or connection pool exhaustion in anomaly_findings; a memory trend should be connected to GC events. \
Do not treat each stream in isolation.

Your task:
1. Identify the LEADING INDICATOR — the earliest signal that preceded the cascade. \
Be specific: name the metric, counter, or log event, and explain why it is the trigger not a symptom.
2. Build a CAUSE → EFFECT chain — an ordered list of steps showing how the leading indicator propagated \
into downstream failures. Each step should be a single precise sentence an engineer can act on.
3. Produce up to 5 ranked ROOT CAUSE HYPOTHESES. For each:
   - rank (1 = most likely)
   - hypothesis (1 clear sentence stating the root cause)
   - evidence (specific findings, metrics, or log patterns that support it)
   - confidence ("High" | "Medium" | "Low")
   - recommended_action (1 sentence — the first tuning or investigation step you would take)
4. Write a 4–6 sentence EXECUTIVE SUMMARY for a test report audience (delivery manager + architect). \
Cover: what failed, when, the probable root cause, business impact, and the single most important remediation.

Return ONLY valid JSON with this schema:
{{
  "leading_indicator": "<specific metric or event and brief rationale>",
  "cause_effect_chain": ["step 1", "step 2", ...],
  "hypotheses": [
    {{
      "rank": 1,
      "hypothesis": "...",
      "evidence": "...",
      "confidence": "High",
      "recommended_action": "..."
    }},
    ...
  ],
  "summary": "<4-6 sentence expert narrative>"
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
        raw = (response.choices[0].message.content or "").strip()
        # Strip markdown code fences that GPT-4o often wraps around JSON
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```\s*$", "", raw).strip()
        data = json.loads(raw) if raw else {}
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
