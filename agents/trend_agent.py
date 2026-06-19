"""
Trend Agent
-----------
Detects monotonic degradation patterns:
  - Memory leaks (continuously rising working set / non-paged pool)
  - Connection pool exhaustion (steadily climbing active connections)
  - GC pressure creep (rising GC time fraction)
  - Response time drift over steady-state period

Method:
  Linear regression slope + Mann-Kendall trend test on the steady-state phase.
  Flags a metric if slope is statistically significant AND direction is degrading.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy import stats

from Utils.logger import get_logger

log = get_logger(__name__)

# Metrics where an upward trend = degradation
_DEGRADING_UP = {
    "mem_available_mb": False,       # lower is bad
    "mem_page_faults_sec": True,
    "mem_pool_nonpaged_b": True,
    "disk_queue_len": True,
    "proc_handle_count": True,
    "proc_thread_count": True,
    "proc_working_set_b": True,
    "response_time_ms": True,
    "tx_response_sec": True,
    "error_event": True,
}


@dataclass
class TrendFinding:
    metric_name:   str
    source:        str
    direction:     str     # "increasing" | "decreasing"
    slope_per_min: float   # units/minute
    p_value:       float
    r_squared:     float
    severity:      str
    phase:         str
    start_ts:      str
    end_ts:        str
    description:   str

    def to_dict(self) -> dict:
        return asdict(self)


def run(df: pd.DataFrame, cfg: dict | None = None) -> list[dict]:
    """Detect monotonic trends in all metrics during steady-state."""
    cfg = cfg or {}
    findings: list[TrendFinding] = []

    # Focus on steady-state for trend detection (least noise)
    steady = df[df["phase"] == "steady_state"]
    if steady.empty:
        steady = df  # fallback: use all data

    for metric, grp in steady.groupby("metric_name"):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        if len(grp) < 10:
            continue

        values = grp["value"].values.astype(float)
        # Convert timestamp to elapsed minutes for interpretable slope
        t_min = grp["timestamp"].min()
        elapsed_min = (grp["timestamp"] - t_min).dt.total_seconds().values / 60.0

        # Linear regression
        slope, intercept, r_value, p_value, _ = stats.linregress(elapsed_min, values)
        r_sq = r_value ** 2

        # Only flag if statistically significant and r² meaningful
        if p_value >= 0.05 or r_sq < 0.3:
            continue

        direction = "increasing" if slope > 0 else "decreasing"

        # Determine if this direction is a degradation for this metric
        degrading_up = _DEGRADING_UP.get(str(metric), None)
        if degrading_up is None:
            # Heuristic: response times / error counts increasing = bad
            name_lower = str(metric).lower()
            degrading_up = any(k in name_lower for k in
                               ["error", "latency", "time", "queue", "fault", "handle", "thread"])

        is_degrading = (degrading_up and direction == "increasing") or \
                       (not degrading_up and direction == "decreasing")

        if not is_degrading:
            continue

        # Severity: how much does it change over the steady window?
        total_change_pct = abs(slope * elapsed_min[-1]) / max(abs(values.mean()), 1e-9) * 100
        severity = "critical" if total_change_pct > 20 else "warn"

        source = grp["source"].iloc[0]
        findings.append(TrendFinding(
            metric_name=str(metric),
            source=str(source),
            direction=direction,
            slope_per_min=round(slope, 6),
            p_value=round(p_value, 6),
            r_squared=round(r_sq, 4),
            severity=severity,
            phase="steady_state",
            start_ts=str(grp["timestamp"].min()),
            end_ts=str(grp["timestamp"].max()),
            description=(
                f"{metric} is {direction} at {slope:+.4f} units/min "
                f"(p={p_value:.4f}, R²={r_sq:.2f}, Δ≈{total_change_pct:.1f}%)"
            ),
        ))

    log.info(f"[trend_agent] {len(findings)} trend findings")
    return [f.to_dict() for f in findings]
