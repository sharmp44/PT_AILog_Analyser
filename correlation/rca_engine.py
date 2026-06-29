"""
RCA Engine
----------
Combines:
  - causal_agent output (hypotheses + chain)
  - timeline (leading indicator)
  - confidence scorer output

into a final RCA result dict ready for the report generator.
"""
from __future__ import annotations

import json
from pathlib import Path

from correlation import scorer as scorer_mod
from correlation import timeline as timeline_mod
from Utils.logger import get_logger

log = get_logger(__name__)


def _compute_verdict(lr_kpis: dict, cfg: dict) -> tuple[str, dict]:
    """
    Determine PASS / FAIL based on LR-measured avg response time and achieved TPS.

    Gates (configured in config.yaml under sla:):
      avg_response_time_sla_sec  — achieved avg must be BELOW this value
      expected_tps               — achieved TPS must meet or exceed this (skipped when 0)

    Returns (verdict, verdict_detail).
    """
    sla      = cfg.get("sla", {})
    rt_sla   = float(sla.get("avg_response_time_sla_sec", 1.0))
    exp_tps  = float(sla.get("expected_tps", 0))

    avg_rt       = lr_kpis.get("avg_response_sec")
    achieved_tps = lr_kpis.get("achieved_tps")

    # Response time gate
    if avg_rt is None:
        rt_pass, rt_status = None, "NO_DATA"
    else:
        rt_pass   = float(avg_rt) < rt_sla
        rt_status = "PASS" if rt_pass else "FAIL"

    # TPS gate
    if exp_tps <= 0:
        tps_pass, tps_status = None, "NOT_CONFIGURED"
    elif achieved_tps is None:
        tps_pass, tps_status = None, "NO_DATA"
    else:
        tps_pass   = float(achieved_tps) >= exp_tps
        tps_status = "PASS" if tps_pass else "FAIL"

    # Overall verdict — FAIL if any enforced gate fails
    enforced = [r for r in [rt_pass, tps_pass] if r is not None]
    if not enforced:
        verdict = "UNKNOWN"
    elif all(enforced):
        verdict = "PASS"
    else:
        verdict = "FAIL"

    verdict_detail = {
        "avg_response_time": {
            "actual_sec":    round(float(avg_rt), 3) if avg_rt is not None else None,
            "threshold_sec": rt_sla,
            "status":        rt_status,
        },
        "tps": {
            "achieved": round(float(achieved_tps), 2) if achieved_tps is not None else None,
            "expected": exp_tps if exp_tps > 0 else "not configured",
            "status":   tps_status,
        },
    }

    log.info(
        f"[rca_engine] Verdict={verdict} | "
        f"avg_rt={avg_rt}s (SLA<{rt_sla}s) → {rt_status} | "
        f"TPS={achieved_tps} (exp>={exp_tps}) → {tps_status}"
    )
    return verdict, verdict_detail


def build(
    causal_analysis:    dict,
    timeline:           dict,
    anomaly_findings:   list[dict],
    trend_findings:     list[dict],
    threshold_findings: list[dict],
    pattern_findings:   list[dict],
    lr_kpis:            dict | None = None,
    cfg:                dict | None = None,
    output_dir: str | Path | None = None,
) -> dict:
    """
    Assemble the complete RCA result.

    Verdict is driven by LR KPIs:
      - avg response time vs avg_response_time_sla_sec (PASS if below)
      - achieved TPS vs expected_tps (PASS if at or above)

    lr_kpis should contain: {"avg_response_sec": float, "achieved_tps": float}

    Returns a dict with:
      - verdict            : "PASS" | "FAIL" | "UNKNOWN"
      - verdict_detail     : per-gate breakdown
      - summary, leading_indicator, cause_effect_chain
      - scored_hypotheses, timeline, finding_counts, raw_findings
    """
    cfg     = cfg or {}
    lr_kpis = lr_kpis or {}

    hypotheses = causal_analysis.get("hypotheses", [])
    scored     = scorer_mod.score(hypotheses)

    finding_counts = {
        "anomaly":   len(anomaly_findings),
        "trend":     len(trend_findings),
        "threshold": len(threshold_findings),
        "pattern":   len(pattern_findings),
        "total":     len(anomaly_findings) + len(trend_findings) +
                     len(threshold_findings) + len(pattern_findings),
    }

    verdict, verdict_detail = _compute_verdict(lr_kpis, cfg)

    result = {
        "verdict":            verdict,
        "verdict_detail":     verdict_detail,
        "summary":            causal_analysis.get("summary", ""),
        "leading_indicator":  timeline.get("leading_indicator"),
        "cause_effect_chain": causal_analysis.get("cause_effect_chain", []),
        "scored_hypotheses":  scored,
        "timeline":           timeline,
        "finding_counts":     finding_counts,
        "raw_findings": {
            "anomaly":   anomaly_findings,
            "trend":     trend_findings,
            "threshold": threshold_findings,
            "pattern":   pattern_findings,
        },
    }

    if output_dir:
        out_path = Path(output_dir) / "rca_result.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        log.info(f"[rca_engine] Raw RCA JSON saved to {out_path}")

    log.info(f"[rca_engine] Verdict={verdict}, {finding_counts['total']} total findings, "
             f"{len(scored)} hypotheses scored")
    return result
