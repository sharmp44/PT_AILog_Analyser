"""
LoadRunner Log Parser
---------------------
Handles two common LR output artefacts:

1. **output.mdb / results.csv** – transaction-level result CSVs exported from
   Analysis or via lr_analyze. Columns vary by LR version but we normalise the
   most common ones.

2. **vuser*.log** – per-vuser text logs containing:
   - "Notify: Transaction ... started/ended" lines
   - "Error: ..." / "Warning: ..." lines
   - Think-time deviations

3. **SLA breach report** – a simple CSV with columns:
   TransactionName, Goal, Actual, Status

Output unified schema:
  timestamp   : datetime64[ns, UTC]
  source      : str   ("loadrunner")
  metric_name : str
  value       : float
  unit        : str
  severity    : str
  raw_line    : str

Extra columns for correlation:
  transaction_name, vuser_id, phase
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from Utils.logger import get_logger

log = get_logger(__name__)

# ── Patterns for vuser log parsing ─────────────────────────────────────────
_RX_TS     = re.compile(r"(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})")
_RX_TX_END = re.compile(
    r"Notify: Transaction \"(?P<tx>[^\"]+)\" ended with a duration of "
    r"(?P<dur>[\d.]+) secs? status: (?P<status>\w+)",
    re.IGNORECASE,
)
_RX_TX_TPS = re.compile(
    r"Notify: Transaction \"(?P<tx>[^\"]+)\" started",
    re.IGNORECASE,
)
_RX_ERROR  = re.compile(r"Error\s*[:\-]?\s*(.+)", re.IGNORECASE)
_RX_WARN   = re.compile(r"Warning\s*[:\-]?\s*(.+)", re.IGNORECASE)
_RX_VUSER  = re.compile(r"vuser[_\-]?(\d+)", re.IGNORECASE)

# ── Result CSV column aliases ───────────────────────────────────────────────
_COL_ALIASES: dict[str, str] = {
    "transaction name":  "transaction_name",
    "transactionname":   "transaction_name",
    "name":              "transaction_name",
    "average (sec)":     "avg_response_sec",
    "average":           "avg_response_sec",
    "minimum (sec)":     "min_response_sec",
    "maximum (sec)":     "max_response_sec",
    "90th percentile":   "p90_response_sec",
    "95th percentile":   "p95_response_sec",
    "std. deviation":    "std_dev_sec",
    "pass":              "pass_count",
    "fail":              "fail_count",
    "stop":              "stop_count",
    "start time":        "start_time",
    "end time":          "end_time",
    "throughput (hits/sec)": "tps",
    "throughput":        "tps",
}


def _normalise_col(c: str) -> str:
    return _COL_ALIASES.get(c.strip().lower(), c.strip().lower().replace(" ", "_"))


# ── CSV result file parser ──────────────────────────────────────────────────

def _parse_results_csv(path: Path, cfg: dict) -> pd.DataFrame:
    raw = pd.read_csv(path, low_memory=False)
    raw.columns = [_normalise_col(c) for c in raw.columns]

    rows = []
    metrics = ["avg_response_sec", "p95_response_sec", "p90_response_sec",
               "max_response_sec", "tps", "fail_count"]

    # Try to get a test start time for relative timestamps
    base_ts = pd.Timestamp.utcnow()
    if "start_time" in raw.columns:
        try:
            base_ts = pd.to_datetime(raw["start_time"].dropna().iloc[0], utc=True)
        except Exception:
            pass

    for _, row in raw.iterrows():
        tx = str(row.get("transaction_name", "unknown")).strip()

        # Build a ts from start_time if present, else use base_ts
        if "start_time" in raw.columns:
            try:
                ts = pd.to_datetime(row["start_time"], utc=True)
            except Exception:
                ts = base_ts
        else:
            ts = base_ts

        for metric in metrics:
            if metric not in row.index:
                continue
            try:
                val = float(row[metric])
            except (ValueError, TypeError):
                continue

            unit = "s" if "sec" in metric else ("tps" if metric == "tps" else "")
            sla_sec = cfg.get("sla", {}).get("p95_latency_ms", 2000) / 1000
            if metric == "p95_response_sec":
                sev = "critical" if val > sla_sec else ("warn" if val > sla_sec * 0.8 else "info")
            elif metric == "fail_count":
                sev = "critical" if val > 0 else "info"
            else:
                sev = "info"

            rows.append({
                "timestamp":        ts,
                "source":           "loadrunner",
                "metric_name":      metric,
                "value":            val,
                "unit":             unit,
                "severity":         sev,
                "raw_line":         f"{tx}|{metric}={val}",
                "transaction_name": tx,
                "vuser_id":         "",
                "phase":            "steady",
            })

    return pd.DataFrame(rows)


# ── Vuser log parser ────────────────────────────────────────────────────────

def _parse_vuser_log(path: Path, cfg: dict) -> pd.DataFrame:
    rows = []
    current_ts = pd.Timestamp.utcnow()

    vuser_match = _RX_VUSER.search(path.name)
    vuser_id = vuser_match.group(1) if vuser_match else "0"

    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.rstrip()

            # Try to extract timestamp from line
            ts_match = _RX_TS.search(line)
            if ts_match:
                try:
                    current_ts = pd.to_datetime(ts_match.group(1), utc=True)
                except Exception:
                    pass

            # Transaction end (response time)
            m = _RX_TX_END.search(line)
            if m:
                tx   = m.group("tx")
                dur  = float(m.group("dur"))
                stat = m.group("status").lower()
                sev  = "critical" if stat == "fail" else "info"
                rows.append({
                    "timestamp":        current_ts,
                    "source":           "loadrunner",
                    "metric_name":      "tx_response_sec",
                    "value":            dur,
                    "unit":             "s",
                    "severity":         sev,
                    "raw_line":         line,
                    "transaction_name": tx,
                    "vuser_id":         vuser_id,
                    "phase":            "unknown",
                })
                continue

            # Error lines
            m = _RX_ERROR.search(line)
            if m:
                rows.append({
                    "timestamp":        current_ts,
                    "source":           "loadrunner",
                    "metric_name":      "error_event",
                    "value":            1.0,
                    "unit":             "",
                    "severity":         "critical",
                    "raw_line":         line,
                    "transaction_name": "",
                    "vuser_id":         vuser_id,
                    "phase":            "unknown",
                })
                continue

            # Warning lines
            m = _RX_WARN.search(line)
            if m:
                rows.append({
                    "timestamp":        current_ts,
                    "source":           "loadrunner",
                    "metric_name":      "warning_event",
                    "value":            1.0,
                    "unit":             "",
                    "severity":         "warn",
                    "raw_line":         line,
                    "transaction_name": "",
                    "vuser_id":         vuser_id,
                    "phase":            "unknown",
                })

    return pd.DataFrame(rows)


# ── Public entry point ──────────────────────────────────────────────────────

def parse(file_path: str | Path, cfg: dict | None = None) -> pd.DataFrame:
    """
    Parse a LoadRunner artefact.
    Auto-detects type from filename suffix:
      .csv / .txt  → results CSV
      .log         → vuser log
    """
    path = Path(file_path)
    cfg  = cfg or {}
    log.info(f"Parsing LoadRunner file: {path}")

    suffix = path.suffix.lower()
    if suffix in (".csv", ".txt", ".tsv"):
        df = _parse_results_csv(path, cfg)
    elif suffix == ".log":
        df = _parse_vuser_log(path, cfg)
    else:
        # Attempt vuser log format as fallback
        log.warning(f"Unknown LR extension '{suffix}', trying vuser log format")
        df = _parse_vuser_log(path, cfg)

    if df.empty:
        log.warning(f"No data parsed from LR file {path}")
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    log.info(f"LR parsed: {len(df)} rows, span={df['timestamp'].min()} → {df['timestamp'].max()}")
    return df
