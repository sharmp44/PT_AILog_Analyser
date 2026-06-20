"""
Threshold Agent
---------------
Rule-based SLA breach detector.
Maps each breach to the business KPI it violates.

Rules are derived from the SLA section in config.yaml and can be
extended by adding entries to _RULES below.

Returns a list of ThresholdFinding dicts.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable

import numpy as np
import pandas as pd

from Utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class ThresholdFinding:
    metric_name:   str
    source:        str
    rule_name:     str
    threshold:     float
    actual:        float       # p95 or max or rate, depending on rule
    breach_count:  int
    severity:      str
    business_kpi:  str
    phase:         str
    start_ts:      str
    end_ts:        str

    def to_dict(self) -> dict:
        return asdict(self)


# ── Rule definitions ─────────────────────────────────────────────────────────
# Each rule: (metric_name_pattern, aggregation_fn, threshold_cfg_key, operator,
#             rule_label, business_kpi)
# operator: "gt" | "lt"

_RULES: list[tuple[str, str, str, str, str, str]] = [
    # metric_pattern                    agg     cfg_key                        op   label                              kpi
    # ── Web / App ──────────────────────────────────────────────────────────────
    ("response_time_ms",               "p95",  "p95_latency_ms",              "gt","P95 Latency SLA Breach",         "User Experience / SLA"),
    ("tx_response_sec",                "p95",  "p95_latency_ms",              "gt","Transaction P95 Breach",         "User Experience / SLA"),
    ("cpu_pct",                        "max",  "cpu_pct",                     "gt","CPU Saturation",                 "System Capacity"),
    ("processor_cpu_pct",              "max",  "cpu_pct",                     "gt","CPU Saturation",                 "System Capacity"),
    ("mem_committed_pct",              "max",  "memory_pct",                  "gt","Memory Pressure",                "System Capacity"),
    ("disk_queue_len",                 "max",  "disk_queue_length",           "gt","Disk Queue Saturation",          "I/O Throughput"),
    ("error_event",                    "sum",  None,                          "gt","Error Event Detected",            "Reliability"),
    ("http_status",                    "none", None,                          "gt","HTTP 5xx Errors",                 "Availability"),
    ("tps",                            "drop", "tps_drop_pct",                "lt","TPS Drop",                       "Throughput"),
    # ── SQL Server ─────────────────────────────────────────────────────────────
    ("sql_buffer_cache_hit_pct",       "min",  "sql_min_buffer_cache_hit_pct","lt","SQL Buffer Cache Hit Low",       "Database Performance"),
    ("sql_page_life_expectancy_s",     "min",  "sql_min_page_life_expectancy_s","lt","SQL Page Life Expectancy Low", "Database Memory"),
    ("sql_deadlocks_sec",              "max",  "sql_max_deadlocks_sec",       "gt","SQL Deadlocks Detected",         "Database Concurrency"),
    ("sql_lock_waits_sec",             "max",  "sql_max_lock_waits_sec",      "gt","SQL Lock Contention",            "Database Concurrency"),
    ("sql_processes_blocked",          "max",  "sql_max_processes_blocked",   "gt","SQL Blocked Processes",          "Database Concurrency"),
    ("sql_recompilations_sec",         "max",  "sql_max_recompilations_sec",  "gt","SQL Excessive Re-compilations", "Database Performance"),
]


def _agg(series: pd.Series, agg: str) -> float:
    if agg == "p95":
        return float(np.percentile(series.values, 95))
    if agg == "p90":
        return float(np.percentile(series.values, 90))
    if agg == "max":
        return float(series.max())
    if agg == "min":
        return float(series.min())
    if agg == "sum":
        return float(series.sum())
    if agg == "mean":
        return float(series.mean())
    return float(series.max())


def run(df: pd.DataFrame, cfg: dict | None = None) -> list[dict]:
    """Evaluate all threshold rules against the dataset."""
    cfg = cfg or {}
    sla = {**cfg.get("sla", {}), **{
        f"sql_{k}": v for k, v in cfg.get("sql_sla", {}).items()
    }}

    findings: list[ThresholdFinding] = []

    for (pattern, agg, cfg_key, op, label, kpi) in _RULES:

        # ── Special case: HTTP 5xx ────────────────────────────────────────
        if pattern == "http_status":
            matches = df[df["metric_name"] == "http_status"]
            if matches.empty:
                continue
            errors = matches[matches["value"] >= 500]
            if errors.empty:
                continue
            error_rate = len(errors) / len(matches) * 100
            threshold_pct = sla.get("error_rate_pct", 1.0)
            if error_rate > threshold_pct:
                findings.append(ThresholdFinding(
                    metric_name="http_status", source="iis",
                    rule_name="HTTP 5xx Error Rate", threshold=threshold_pct,
                    actual=round(error_rate, 2), breach_count=len(errors),
                    severity="critical", business_kpi=kpi, phase="all",
                    start_ts=str(errors["timestamp"].min()),
                    end_ts=str(errors["timestamp"].max()),
                ))
            continue

        # ── Special case: TPS drop ────────────────────────────────────────
        if agg == "drop":
            matches = df[df["metric_name"].str.contains(pattern, case=False, na=False)]
            if matches.empty or len(matches) < 10:
                continue
            # Compare first quartile vs last quartile
            q1_mean = matches.head(len(matches) // 4)["value"].mean()
            q4_mean = matches.tail(len(matches) // 4)["value"].mean()
            if q1_mean <= 0:
                continue
            drop_pct = (q1_mean - q4_mean) / q1_mean * 100
            threshold_drop = sla.get("tps_drop_pct", 20.0)
            if drop_pct > threshold_drop:
                findings.append(ThresholdFinding(
                    metric_name=pattern, source=str(matches["source"].iloc[0]),
                    rule_name=label, threshold=threshold_drop,
                    actual=round(drop_pct, 2), breach_count=1,
                    severity="critical", business_kpi=kpi, phase="all",
                    start_ts=str(matches["timestamp"].min()),
                    end_ts=str(matches["timestamp"].max()),
                ))
            continue

        # ── Standard threshold rules ──────────────────────────────────────
        matches = df[df["metric_name"].str.contains(pattern, case=False, na=False)]
        if matches.empty:
            continue

        threshold = float(sla.get(cfg_key, 0)) if cfg_key else 0.0
        if threshold == 0 and cfg_key:
            continue

        actual = _agg(matches["value"], agg)

        breaches: pd.DataFrame
        if op == "gt":
            breaches = matches[matches["value"] > threshold]
        else:
            breaches = matches[matches["value"] < threshold]

        overall_breach = (actual > threshold) if op == "gt" else (actual < threshold)
        if breaches.empty and not overall_breach:
            continue

        if overall_breach or not breaches.empty:
            severity = "critical" if (
                (op == "gt" and actual > threshold * 1.2) or
                (op == "lt" and actual < threshold * 0.8)
            ) else "warn"
            findings.append(ThresholdFinding(
                metric_name=pattern, source=str(matches["source"].iloc[0]),
                rule_name=label, threshold=threshold,
                actual=round(actual, 4), breach_count=len(breaches),
                severity=severity, business_kpi=kpi,
                phase=str(matches["phase"].mode().iloc[0] if not matches["phase"].empty else ""),
                start_ts=str(breaches["timestamp"].min() if not breaches.empty else matches["timestamp"].min()),
                end_ts=str(breaches["timestamp"].max() if not breaches.empty else matches["timestamp"].max()),
            ))

    log.info(f"[threshold_agent] {len(findings)} threshold breaches")
    return [f.to_dict() for f in findings]
