"""
SQL Server Metrics Parser
-------------------------
Parses PerfMon CSV files that contain SQLServer: performance counters.

Expected format (same as BLG/PerfMon CSV export):
  "(PDH-CSV 4.0) (UTC)(0)","\\SERVER\\SQLServer:Buffer Manager\\Buffer cache hit ratio",...
  "06/19/2024 09:00:00.000","98.5",...

Recognised SQL Server counter categories:
  - Buffer Manager  (cache hit ratio, page life expectancy, lazy writes)
  - SQL Statistics  (batch requests, compilations, re-compilations)
  - General Statistics (user connections, processes blocked)
  - Locks           (lock waits, deadlocks)
  - Memory Manager  (total/target/free server memory)
  - Databases       (transactions/sec, log flush waits)
  - Access Methods  (full scans, index searches)
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from Utils.logger import get_logger

log = get_logger(__name__)

# ── Counter → (metric_name, unit) ────────────────────────────────────────────
_SQL_COUNTER_MAP: list[tuple[str, str, str]] = [
    # pattern (lowercase substring)      metric_name                      unit
    ("buffer cache hit ratio",          "sql_buffer_cache_hit_pct",      "%"),
    ("page life expectancy",            "sql_page_life_expectancy_s",    "s"),
    ("lazy writes/sec",                 "sql_lazy_writes_sec",           "/s"),
    ("checkpoint pages/sec",            "sql_checkpoint_pages_sec",      "pages/s"),
    ("readahead pages/sec",             "sql_readahead_pages_sec",       "pages/s"),
    ("batch requests/sec",              "sql_batch_requests_sec",        "req/s"),
    ("sql compilations/sec",            "sql_compilations_sec",          "/s"),
    ("sql re-compilations/sec",         "sql_recompilations_sec",        "/s"),
    ("user connections",                "sql_user_connections",          "count"),
    ("processes blocked",               "sql_processes_blocked",         "count"),
    ("lock waits/sec",                  "sql_lock_waits_sec",            "/s"),
    ("deadlocks/sec",                   "sql_deadlocks_sec",             "/s"),
    ("lock wait time (ms)",             "sql_lock_wait_time_ms",         "ms"),
    ("transactions/sec",                "sql_transactions_sec",          "tx/s"),
    ("log flush waits/sec",             "sql_log_flush_waits_sec",       "/s"),
    ("log bytes flushed/sec",           "sql_log_bytes_flushed_sec",     "B/s"),
    ("target server memory",            "sql_target_memory_kb",          "KB"),
    ("total server memory",             "sql_total_memory_kb",           "KB"),
    ("stolen server memory",            "sql_stolen_memory_kb",          "KB"),
    ("free memory",                     "sql_free_memory_kb",            "KB"),
    ("full scans/sec",                  "sql_full_scans_sec",            "/s"),
    ("index searches/sec",              "sql_index_searches_sec",        "/s"),
    ("forwarded records/sec",           "sql_forwarded_records_sec",     "/s"),
    ("active temp tables",              "sql_active_temp_tables",        "count"),
]

_IS_SQL_HDR = re.compile(r"sqlserver:", re.IGNORECASE)


def is_sql_perfmon(path: Path) -> bool:
    """Return True if the CSV contains SQLServer: counter headers."""
    try:
        with open(path, errors="replace") as fh:
            for _ in range(5):
                line = fh.readline()
                if _IS_SQL_HDR.search(line):
                    return True
    except Exception:
        pass
    return False


def _friendly_name(raw_col: str) -> tuple[str, str]:
    lower = raw_col.lower()
    for pattern, name, unit in _SQL_COUNTER_MAP:
        if pattern in lower:
            return name, unit
    # Fallback: sanitise
    sanitised = re.sub(r"[^\w]", "_", lower).strip("_")
    return sanitised, ""


def parse(file_path: str | Path, cfg: dict | None = None) -> pd.DataFrame:
    """
    Parse a PerfMon CSV that contains SQL Server counters.
    Returns a DataFrame in the unified pipeline schema.
    """
    path = Path(file_path)
    cfg = cfg or {}
    log.info(f"Parsing SQL metrics CSV: {path}")

    raw = pd.read_csv(path, header=0, low_memory=False)
    raw.columns = [c.strip().strip('"') for c in raw.columns]

    ts_col = raw.columns[0]
    metric_cols = [c for c in raw.columns[1:] if _IS_SQL_HDR.search(c)]

    if not metric_cols:
        log.warning(f"No SQLServer: counters found in {path.name}")
        return pd.DataFrame()

    rows = []
    for _, row in raw.iterrows():
        ts_raw = str(row[ts_col]).strip()
        try:
            ts = pd.to_datetime(ts_raw, utc=True)
        except Exception:
            continue

        for col in metric_cols:
            val_raw = row[col]
            try:
                val = float(val_raw)
            except (ValueError, TypeError):
                continue

            metric_name, unit = _friendly_name(col)
            rows.append({
                "timestamp":   ts,
                "source":      "sql",
                "metric_name": metric_name,
                "value":       val,
                "unit":        unit,
                "severity":    "info",
                "raw_line":    f"{col}={val_raw}",
            })

    df = pd.DataFrame(rows)
    if df.empty:
        log.warning(f"No SQL data parsed from {path.name}")
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    log.info(
        f"SQL parsed: {len(df)} rows, {df['metric_name'].nunique()} metrics, "
        f"span={df['timestamp'].min()} → {df['timestamp'].max()}"
    )
    return df
