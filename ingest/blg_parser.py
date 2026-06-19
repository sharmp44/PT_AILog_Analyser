"""
.BLG Parser
-----------
Windows PerfMon binary logs → normalised DataFrame.

Strategy:
  1. Use `relog` (Windows) or `perfmon2csv` to convert .blg → .csv first.
  2. If already a .csv (pre-converted), parse directly.
  3. Output: unified schema DataFrame.

Unified schema columns:
  timestamp   : datetime64[ns, UTC]
  source      : str   ("blg")
  metric_name : str
  value       : float
  unit        : str
  severity    : str   ("info" | "warn" | "critical")
  raw_line    : str
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

from Utils.logger import get_logger

log = get_logger(__name__)

# Map common PerfMon counter suffixes → human-friendly names
_COUNTER_MAP = {
    "% processor time":       ("cpu_pct",              "%"),
    "% user time":            ("cpu_user_pct",          "%"),
    "% privileged time":      ("cpu_kernel_pct",        "%"),
    "available mbytes":       ("mem_available_mb",      "MB"),
    "% committed bytes in use":("mem_committed_pct",   "%"),
    "disk reads/sec":         ("disk_reads_sec",        "ops/s"),
    "disk writes/sec":        ("disk_writes_sec",       "ops/s"),
    "avg. disk queue length": ("disk_queue_len",        ""),
    "bytes total/sec":        ("net_bytes_total_sec",   "B/s"),
    "bytes sent/sec":         ("net_bytes_sent_sec",    "B/s"),
    "bytes received/sec":     ("net_bytes_recv_sec",    "B/s"),
    "handle count":           ("proc_handle_count",     ""),
    "thread count":           ("proc_thread_count",     ""),
    "working set":            ("proc_working_set_b",    "B"),
    "page faults/sec":        ("mem_page_faults_sec",   "faults/s"),
    "pool nonpaged bytes":    ("mem_pool_nonpaged_b",   "B"),
}


def _convert_blg_to_csv(blg_path: Path) -> Path:
    """Use relog.exe (Windows) to convert .blg → .csv in the same dir."""
    csv_path = blg_path.with_suffix(".csv")
    if csv_path.exists():
        log.info(f"Found pre-converted CSV: {csv_path}")
        return csv_path

    if sys.platform != "win32":
        raise RuntimeError(
            f"Cannot convert .BLG on non-Windows. "
            f"Please pre-convert {blg_path.name} to CSV using relog or perfmon2csv."
        )

    log.info(f"Running relog to convert {blg_path.name} …")
    result = subprocess.run(
        ["relog", str(blg_path), "-f", "CSV", "-o", str(csv_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"relog failed: {result.stderr}")

    log.info(f"relog produced {csv_path}")
    return csv_path


def _friendly_name(raw_counter: str) -> tuple[str, str]:
    """Return (metric_name, unit) from a raw PerfMon column header."""
    lower = raw_counter.lower()
    for pattern, (name, unit) in _COUNTER_MAP.items():
        if pattern in lower:
            # Extract object/instance for uniqueness, e.g. Processor(_Total)
            obj_match = re.search(r"\\\\[^\\]+\\([^\\(]+)(?:\([^)]+\))?\\", raw_counter)
            prefix = obj_match.group(1).replace(" ", "_").lower() + "_" if obj_match else ""
            return prefix + name, unit
    # Fallback: sanitise the raw name
    sanitised = re.sub(r"[^\w]", "_", lower).strip("_")
    return sanitised, ""


def parse(file_path: str | Path, cfg: dict | None = None) -> pd.DataFrame:
    """
    Parse a .BLG or pre-converted .CSV PerfMon file.

    Returns a DataFrame in the unified pipeline schema.
    """
    path = Path(file_path)
    cfg = cfg or {}

    if path.suffix.lower() == ".blg":
        path = _convert_blg_to_csv(path)

    log.info(f"Parsing BLG/CSV: {path}")

    # PerfMon CSV has a header row like:
    #   "(PDH-CSV 4.0) (UTC)(0)","\\MACHINE\Counter1","\\MACHINE\Counter2",...
    # and data rows:
    #   "MM/DD/YYYY HH:MM:SS.mmm","value1","value2",...
    raw = pd.read_csv(path, header=0, low_memory=False)
    raw.columns = [c.strip().strip('"') for c in raw.columns]

    # First column is always the timestamp
    ts_col = raw.columns[0]
    metric_cols = raw.columns[1:]

    rows = []
    for _, row in raw.iterrows():
        ts_raw = str(row[ts_col]).strip()
        try:
            ts = pd.to_datetime(ts_raw, utc=True)
        except Exception:
            continue  # skip malformed timestamp rows

        for col in metric_cols:
            val_raw = row[col]
            try:
                val = float(val_raw)
            except (ValueError, TypeError):
                continue

            metric_name, unit = _friendly_name(col)
            rows.append({
                "timestamp":   ts,
                "source":      "blg",
                "metric_name": metric_name,
                "value":       val,
                "unit":        unit,
                "severity":    "info",   # assigned by agents later
                "raw_line":    f"{col}={val_raw}",
            })

    df = pd.DataFrame(rows)
    if df.empty:
        log.warning(f"No data parsed from {path}")
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    log.info(f"BLG parsed: {len(df)} rows, {df['metric_name'].nunique()} metrics, "
             f"span={df['timestamp'].min()} → {df['timestamp'].max()}")
    return df
