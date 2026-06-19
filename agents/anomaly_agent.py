"""
Anomaly Agent
-------------
Detects statistical outliers in each metric using three complementary methods:
  1. Z-score (fast, parametric)
  2. IQR (robust against skew)
  3. Isolation Forest (catches multi-dimensional anomaly clusters)

Returns a list of AnomalyFinding dicts, one per anomalous data point.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from Utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class AnomalyFinding:
    timestamp:   pd.Timestamp
    source:      str
    metric_name: str
    value:       float
    method:      str          # "zscore" | "iqr" | "isolation_forest"
    score:       float        # magnitude / anomaly score
    severity:    str          # "warn" | "critical"
    phase:       str
    raw_line:    str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp"] = str(self.timestamp)
        return d


def run(df: pd.DataFrame, cfg: dict | None = None) -> list[dict]:
    """
    Run all anomaly detectors on every metric in `df`.

    Returns a list of finding dicts suitable for JSON serialisation.
    """
    cfg = cfg or {}
    z_thresh   = float(cfg.get("anomaly", {}).get("z_score_threshold", 3.0))
    if_contam  = float(cfg.get("anomaly", {}).get("isolation_forest_contamination", 0.05))

    findings: list[AnomalyFinding] = []

    for metric, grp in df.groupby("metric_name"):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        values = grp["value"].values.astype(float)

        if len(values) < 5:
            continue

        # ── 1. Z-score ──────────────────────────────────────────────────────
        mean, std = values.mean(), values.std()
        if std > 0:
            z_scores = np.abs((values - mean) / std)
            for idx in np.where(z_scores > z_thresh)[0]:
                row = grp.iloc[idx]
                sev = "critical" if z_scores[idx] > z_thresh * 1.5 else "warn"
                findings.append(AnomalyFinding(
                    timestamp=row["timestamp"], source=row["source"],
                    metric_name=metric, value=float(values[idx]),
                    method="zscore", score=float(z_scores[idx]),
                    severity=sev, phase=row.get("phase", ""),
                    raw_line=str(row.get("raw_line", "")),
                ))

        # ── 2. IQR ──────────────────────────────────────────────────────────
        q1, q3 = np.percentile(values, 25), np.percentile(values, 75)
        iqr    = q3 - q1
        if iqr > 0:
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            for idx in np.where((values < lo) | (values > hi))[0]:
                row = grp.iloc[idx]
                dist = max(abs(values[idx] - lo), abs(values[idx] - hi))
                sev  = "critical" if dist > 3 * iqr else "warn"
                findings.append(AnomalyFinding(
                    timestamp=row["timestamp"], source=row["source"],
                    metric_name=metric, value=float(values[idx]),
                    method="iqr", score=float(dist / iqr),
                    severity=sev, phase=row.get("phase", ""),
                    raw_line=str(row.get("raw_line", "")),
                ))

        # ── 3. Isolation Forest ─────────────────────────────────────────────
        if len(values) >= 20:
            X = values.reshape(-1, 1)
            clf = IsolationForest(contamination=if_contam, random_state=42)
            preds = clf.fit_predict(X)   # -1 = anomaly, 1 = normal
            scores = clf.score_samples(X)
            for idx in np.where(preds == -1)[0]:
                row = grp.iloc[idx]
                findings.append(AnomalyFinding(
                    timestamp=row["timestamp"], source=row["source"],
                    metric_name=metric, value=float(values[idx]),
                    method="isolation_forest", score=float(-scores[idx]),
                    severity="warn", phase=row.get("phase", ""),
                    raw_line=str(row.get("raw_line", "")),
                ))

    log.info(f"[anomaly_agent] {len(findings)} anomalies detected across {df['metric_name'].nunique()} metrics")
    return [f.to_dict() for f in findings]
