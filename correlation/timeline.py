"""
Timeline Alignment
------------------
Merges anomaly events from all sources onto a unified timeline
and identifies the leading indicator — the metric that degraded first
before user-visible impact.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import pandas as pd

from Utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class TimelineEvent:
    timestamp:   str
    source:      str
    event_type:  str   # "anomaly" | "threshold" | "trend_start" | "pattern"
    metric_name: str
    severity:    str
    description: str

    def to_dict(self) -> dict:
        return asdict(self)


def build(
    anomaly_findings:   list[dict],
    trend_findings:     list[dict],
    threshold_findings: list[dict],
    pattern_findings:   list[dict],
) -> dict:
    """
    Merge all agent findings into a chronological unified timeline.

    Returns:
        {
          "events": [...],              # sorted list of TimelineEvent dicts
          "leading_indicator": {...},   # first critical/warn event
          "first_event_ts": "...",
          "last_event_ts": "...",
          "duration_minutes": float,
        }
    """
    events: list[TimelineEvent] = []

    # ── Anomaly findings ──────────────────────────────────────────────────────
    for f in anomaly_findings:
        events.append(TimelineEvent(
            timestamp=str(f.get("timestamp", "")),
            source=str(f.get("source", "")),
            event_type="anomaly",
            metric_name=str(f.get("metric_name", "")),
            severity=str(f.get("severity", "warn")),
            description=f"Anomaly ({f.get('method','')}) on {f.get('metric_name','')} "
                        f"value={f.get('value','')} score={f.get('score','')}",
        ))

    # ── Trend findings ────────────────────────────────────────────────────────
    for f in trend_findings:
        events.append(TimelineEvent(
            timestamp=str(f.get("start_ts", "")),
            source=str(f.get("source", "")),
            event_type="trend_start",
            metric_name=str(f.get("metric_name", "")),
            severity=str(f.get("severity", "warn")),
            description=f.get("description", ""),
        ))

    # ── Threshold findings ────────────────────────────────────────────────────
    for f in threshold_findings:
        events.append(TimelineEvent(
            timestamp=str(f.get("start_ts", "")),
            source=str(f.get("source", "")),
            event_type="threshold",
            metric_name=str(f.get("metric_name", "")),
            severity=str(f.get("severity", "warn")),
            description=f"{f.get('rule_name','')} — threshold={f.get('threshold','')} "
                        f"actual={f.get('actual','')} breaches={f.get('breach_count','')}",
        ))

    # ── Pattern findings ──────────────────────────────────────────────────────
    for f in pattern_findings:
        events.append(TimelineEvent(
            timestamp=str(f.get("first_ts", "")),
            source=str(f.get("source", "")),
            event_type="pattern",
            metric_name=str(f.get("pattern_name", "")),
            severity=str(f.get("severity", "warn")),
            description=f.get("description", ""),
        ))

    if not events:
        return {"events": [], "leading_indicator": None,
                "first_event_ts": "", "last_event_ts": "", "duration_minutes": 0}

    # ── Sort by timestamp (best-effort; empty strings sort first) ─────────────
    def _ts_key(e: TimelineEvent) -> pd.Timestamp:
        try:
            return pd.to_datetime(e.timestamp, utc=True)
        except Exception:
            return pd.Timestamp.min.tz_localize("UTC")

    events.sort(key=_ts_key)

    # ── Leading indicator: first non-info event ────────────────────────────────
    leading = next(
        (e for e in events if e.severity in ("warn", "critical")),
        events[0],
    )

    # ── Time span ────────────────────────────────────────────────────────────
    valid_ts = [_ts_key(e) for e in events if e.timestamp]
    first_ts = min(valid_ts) if valid_ts else pd.Timestamp.min.tz_localize("UTC")
    last_ts  = max(valid_ts) if valid_ts else pd.Timestamp.min.tz_localize("UTC")
    duration_min = (last_ts - first_ts).total_seconds() / 60.0

    log.info(f"[timeline] {len(events)} events, duration={duration_min:.1f} min, "
             f"leading={leading.metric_name}")

    return {
        "events":             [e.to_dict() for e in events],
        "leading_indicator":  leading.to_dict(),
        "first_event_ts":     str(first_ts),
        "last_event_ts":      str(last_ts),
        "duration_minutes":   round(duration_min, 2),
    }
