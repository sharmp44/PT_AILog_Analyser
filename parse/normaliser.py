"""
Log Normaliser
--------------
Accepts a raw DataFrame from any parser and ensures it conforms to the
unified pipeline schema, handles timezone alignment, and fills in
missing columns with sensible defaults.

Unified schema
--------------
Required:
  timestamp    datetime64[ns, UTC]
  source       str
  metric_name  str
  value        float64
  unit         str
  severity     str  (info | warn | critical)
  raw_line     str

Optional (filled with "" / NaN if absent):
  uri_stem         str  (IIS)
  method           str  (IIS)
  status_code      int  (IIS)
  client_ip        str  (IIS)
  transaction_name str  (LR)
  vuser_id         str  (LR)
  phase            str  (LR)
"""
from __future__ import annotations

import pandas as pd

from Utils.logger import get_logger

log = get_logger(__name__)

_REQUIRED_COLS: list[str] = [
    "timestamp", "source", "metric_name", "value", "unit", "severity", "raw_line",
]

_OPTIONAL_COLS: dict[str, object] = {
    "uri_stem":         "",
    "method":           "",
    "status_code":      0,
    "client_ip":        "",
    "transaction_name": "",
    "vuser_id":         "",
    "phase":            "",
}

_VALID_SEVERITIES = {"info", "warn", "critical"}


def normalise(df: pd.DataFrame, source_name: str | None = None) -> pd.DataFrame:
    """
    Normalise and validate a raw parsed DataFrame.

    Args:
        df:           Raw DataFrame from an ingest parser.
        source_name:  Override the 'source' column value if provided.

    Returns:
        Clean, schema-conformant DataFrame (may be empty if input is empty).
    """
    if df.empty:
        return _empty_schema()

    df = df.copy()

    # ── Source ──────────────────────────────────────────────────────────────
    if source_name:
        df["source"] = source_name
    elif "source" not in df.columns:
        df["source"] = "unknown"

    # ── Timestamp ────────────────────────────────────────────────────────────
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame has no 'timestamp' column")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    n_bad_ts = df["timestamp"].isna().sum()
    if n_bad_ts:
        log.warning(f"Dropping {n_bad_ts} rows with unparseable timestamps")
        df = df.dropna(subset=["timestamp"])

    # ── Numeric value ────────────────────────────────────────────────────────
    df["value"] = pd.to_numeric(df.get("value"), errors="coerce").fillna(0.0)

    # ── String columns ────────────────────────────────────────────────────────
    for col in ("metric_name", "unit", "severity", "raw_line"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    # ── Severity guard ────────────────────────────────────────────────────────
    invalid_sev = ~df["severity"].isin(_VALID_SEVERITIES)
    if invalid_sev.any():
        df.loc[invalid_sev, "severity"] = "info"

    # ── Optional columns ─────────────────────────────────────────────────────
    for col, default in _OPTIONAL_COLS.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = df[col].fillna(default)

    # ── Column order ─────────────────────────────────────────────────────────
    ordered = _REQUIRED_COLS + list(_OPTIONAL_COLS.keys())
    extra = [c for c in df.columns if c not in ordered]
    df = df[ordered + extra]

    # ── Sort ──────────────────────────────────────────────────────────────────
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    log.info(
        f"[normalise] {source_name or 'data'}: {len(df)} rows, "
        f"{df['metric_name'].nunique()} unique metrics"
    )
    return df


def _empty_schema() -> pd.DataFrame:
    cols = _REQUIRED_COLS + list(_OPTIONAL_COLS.keys())
    return pd.DataFrame(columns=cols)
