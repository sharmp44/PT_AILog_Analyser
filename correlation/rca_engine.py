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


def build(
    causal_analysis:    dict,
    timeline:           dict,
    anomaly_findings:   list[dict],
    trend_findings:     list[dict],
    threshold_findings: list[dict],
    pattern_findings:   list[dict],
    output_dir: str | Path | None = None,
) -> dict:
    """
    Assemble the complete RCA result.

    Returns a dict with:
      - summary
      - leading_indicator
      - cause_effect_chain
      - scored_hypotheses
      - timeline
      - finding_counts
      - raw_findings
    """
    hypotheses  = causal_analysis.get("hypotheses", [])
    scored      = scorer_mod.score(hypotheses)

    finding_counts = {
        "anomaly":   len(anomaly_findings),
        "trend":     len(trend_findings),
        "threshold": len(threshold_findings),
        "pattern":   len(pattern_findings),
        "total":     len(anomaly_findings) + len(trend_findings) +
                     len(threshold_findings) + len(pattern_findings),
    }

    # Overall pass/fail verdict
    critical_count = sum(
        1 for f in threshold_findings if f.get("severity") == "critical"
    ) + sum(
        1 for f in pattern_findings   if f.get("severity") == "critical"
    )
    verdict = "FAIL" if critical_count > 0 else "PASS"

    result = {
        "verdict":            verdict,
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

    # Optionally save raw RCA JSON for debugging
    if output_dir:
        out_path = Path(output_dir) / "rca_result.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        log.info(f"[rca_engine] Raw RCA JSON saved to {out_path}")

    log.info(f"[rca_engine] Verdict={verdict}, {finding_counts['total']} total findings, "
             f"{len(scored)} hypotheses scored")
    return result
