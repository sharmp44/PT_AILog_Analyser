"""
IIS Log Parser
--------------
W3C Extended Log Format → normalised DataFrame.

Handles:
  - #Version / #Software / #Date / #Fields header directives
  - Comment lines starting with #
  - Variable column sets (Fields line defines which columns are present)
  - time-taken in milliseconds (standard IIS field)

Output schema (unified):
  timestamp   : datetime64[ns, UTC]
  source      : str   ("iis")
  metric_name : str   (e.g. "response_time_ms", "http_status", "bytes_sent")
  value       : float
  unit        : str
  severity    : str   ("info" | "warn" | "critical")
  raw_line    : str

Additional columns kept for correlation:
  uri_stem, method, status_code, client_ip
"""
from __future__ import annotations

import re
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

from Utils.logger import get_logger

log = get_logger(__name__)

# IIS W3C field → (metric_name, unit, multiply_factor)
# time-taken is already in milliseconds in IIS logs
_FIELD_MAP: dict[str, tuple[str, str, float]] = {
    "time-taken":   ("response_time_ms", "ms",   1.0),
    "sc-bytes":     ("bytes_sent",       "B",    1.0),
    "cs-bytes":     ("bytes_recv",       "B",    1.0),
    "sc-status":    ("http_status",      "",     1.0),
    "sc-substatus": ("http_substatus",   "",     1.0),
}

_SEVERITY_MAP = {
    range(100, 400): "info",
    range(400, 500): "warn",
    range(500, 600): "critical",
}


def _http_severity(status: int) -> str:
    for r, sev in _SEVERITY_MAP.items():
        if status in r:
            return sev
    return "info"


def parse(file_path: str | Path, cfg: dict | None = None) -> pd.DataFrame:
    """Parse an IIS W3C log file into unified schema rows."""
    path = Path(file_path)
    cfg = cfg or {}
    log.info(f"Parsing IIS log: {path}")

    fields: list[str] = []
    base_date: str = ""
    raw_rows: list[dict] = []

    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")

            if line.startswith("#Fields:"):
                fields = line[len("#Fields:"):].strip().split()
                continue
            if line.startswith("#Date:"):
                base_date = line[len("#Date:"):].strip()
                continue
            if line.startswith("#"):
                continue
            if not line.strip():
                continue

            parts = line.split(" ")
            if not fields or len(parts) < len(fields):
                continue

            row = dict(zip(fields, parts))

            # Build timestamp from date + time fields
            date_str = row.get("date", base_date)
            time_str = row.get("time", "00:00:00")
            ts_str = f"{date_str} {time_str}"
            try:
                ts = pd.to_datetime(ts_str, utc=True)
            except Exception:
                continue

            # Build metric rows from this log entry
            status_code = int(row.get("sc-status", 0) or 0)
            severity = _http_severity(status_code)

            for field, (metric_name, unit, factor) in _FIELD_MAP.items():
                raw_val = row.get(field, "-")
                if raw_val == "-":
                    continue
                try:
                    val = float(raw_val) * factor
                except ValueError:
                    continue

                # Override severity for response time
                if field == "time-taken":
                    sla_ms = cfg.get("sla", {}).get("p95_latency_ms", 2000)
                    sev = "critical" if val > sla_ms else ("warn" if val > sla_ms * 0.8 else "info")
                else:
                    sev = severity

                raw_rows.append({
                    "timestamp":   ts,
                    "source":      "iis",
                    "metric_name": metric_name,
                    "value":       val,
                    "unit":        unit,
                    "severity":    sev,
                    "raw_line":    line,
                    # extra correlation fields
                    "uri_stem":    row.get("cs-uri-stem", ""),
                    "method":      row.get("cs-method", ""),
                    "status_code": status_code,
                    "client_ip":   row.get("c-ip", ""),
                })

    df = pd.DataFrame(raw_rows)
    if df.empty:
        log.warning(f"No data parsed from IIS log {path}")
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    log.info(f"IIS parsed: {len(df)} rows, span={df['timestamp'].min()} → {df['timestamp'].max()}")
    return df
