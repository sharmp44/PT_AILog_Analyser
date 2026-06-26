"""
IIS Log Parser  (vectorised)
----------------------------
W3C Extended Log Format → normalised DataFrame.

Reads the entire file with pd.read_csv (skipping # comment lines),
then builds all rows with vectorised pandas operations instead of
a Python for-loop. 10-50x faster than the line-by-line approach on
large IIS logs (150 MB+).

Optional row sampling (iis_sample_every in cfg) lets you trade some
statistical resolution for even faster parse times.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from Utils.logger import get_logger

log = get_logger(__name__)

_SEVERITY_THRESH = {
    "crit_lo": 500,
    "warn_lo":  400,
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _read_fields(path: Path) -> list[str]:
    """Return column names from the last #Fields: directive before data rows."""
    fields: list[str] = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#Fields:"):
                fields = line[len("#Fields:"):].strip().split()
            elif not line.startswith("#") and line.strip():
                break          # first data row — stop scanning
    return fields


def _safe_col(df: pd.DataFrame, name: str, default="") -> pd.Series:
    """Return column or a series of defaults if column absent."""
    return df[name] if name in df.columns else pd.Series(default, index=df.index)


# ── Public entry point ───────────────────────────────────────────────────────

def parse(file_path: str | Path, cfg: dict | None = None) -> pd.DataFrame:
    """
    Parse an IIS W3C log file into the unified pipeline schema.

    cfg keys:
      iis_sample_every  int   keep 1-in-N rows (default 1 = keep all)
    """
    path = Path(file_path)
    cfg  = cfg or {}
    sample_every = int(cfg.get("iis_sample_every", 1))

    log.info(f"[iis_parser] Parsing (vectorised): {path.name}  "
             f"({path.stat().st_size / 1_048_576:.1f} MB)")

    # ── Step 1: extract field names ──────────────────────────────────────────
    fields = _read_fields(path)
    if not fields:
        log.warning(f"[iis_parser] No #Fields: directive found in {path.name}")
        return pd.DataFrame()

    # ── Step 2: bulk read — pandas skips all '#' comment lines ───────────────
    try:
        raw = pd.read_csv(
            path,
            sep=" ",
            names=fields,
            comment="#",
            on_bad_lines="skip",
            low_memory=False,
            encoding="utf-8",
            encoding_errors="replace",
        )
    except Exception as exc:
        log.warning(f"[iis_parser] pd.read_csv failed ({exc}), skipping {path.name}")
        return pd.DataFrame()

    if raw.empty:
        log.warning(f"[iis_parser] No data rows in {path.name}")
        return pd.DataFrame()

    # ── Step 3: optional sampling ─────────────────────────────────────────────
    if sample_every > 1:
        raw = raw.iloc[::sample_every].copy()
        log.info(f"[iis_parser] Sampled 1-in-{sample_every} → {len(raw):,} rows")

    # ── Step 4: vectorised timestamp ─────────────────────────────────────────
    if "date" not in raw.columns or "time" not in raw.columns:
        log.warning(f"[iis_parser] Missing date/time columns in {path.name}")
        return pd.DataFrame()

    raw["timestamp"] = pd.to_datetime(
        raw["date"].astype(str) + " " + raw["time"].astype(str),
        utc=True, errors="coerce",
    )
    raw = raw.dropna(subset=["timestamp"])
    if raw.empty:
        return pd.DataFrame()

    # ── Step 5: derive common columns ────────────────────────────────────────
    status_code = pd.to_numeric(_safe_col(raw, "sc-status", 0),
                                errors="coerce").fillna(0).astype(int)
    substatus   = pd.to_numeric(_safe_col(raw, "sc-substatus", 0),
                                errors="coerce").fillna(0).astype(int)
    win32_status = pd.to_numeric(_safe_col(raw, "sc-win32-status", 0),
                                 errors="coerce").fillna(0).astype(int)

    # Per-request severity based on HTTP status
    http_sev = pd.Series("info", index=raw.index, dtype=str)
    http_sev[status_code >= 500] = "critical"
    http_sev[(status_code >= 400) & (status_code < 500)] = "warn"

    uri_stem   = _safe_col(raw, "cs-uri-stem",   "").astype(str)
    uri_query  = _safe_col(raw, "cs-uri-query",  "").astype(str)
    method     = _safe_col(raw, "cs-method",     "").astype(str)
    client_ip  = _safe_col(raw, "c-ip",          "").astype(str)
    username   = _safe_col(raw, "cs-username",   "").astype(str)
    # cs(User-Agent) field name has parentheses — handle both forms
    user_agent = (_safe_col(raw, "cs(User-Agent)", "")
                  if "cs(User-Agent)" in raw.columns
                  else _safe_col(raw, "cs-User-Agent", "")).astype(str)
    bytes_recv = pd.to_numeric(_safe_col(raw, "cs-bytes",  0), errors="coerce").fillna(0)
    bytes_sent_col = pd.to_numeric(_safe_col(raw, "sc-bytes", 0), errors="coerce").fillna(0)
    s_ip       = _safe_col(raw, "s-ip",   "").astype(str)
    s_port     = _safe_col(raw, "s-port", "").astype(str)
    sitename   = _safe_col(raw, "s-sitename", "").astype(str)

    sla_ms = cfg.get("sla", {}).get("p95_latency_ms", 2000)

    dfs: list[pd.DataFrame] = []

    # ── 5a: response time rows (time-taken field) ─────────────────────────────
    if "time-taken" in raw.columns:
        mask_rt = pd.to_numeric(raw["time-taken"], errors="coerce").notna()
        raw_rt  = raw[mask_rt].copy()
        tt      = pd.to_numeric(raw_rt["time-taken"], errors="coerce")

        rt_sev = pd.Series("info", index=raw_rt.index, dtype=str)
        rt_sev[tt > sla_ms]                          = "critical"
        rt_sev[(tt > sla_ms * 0.8) & (tt <= sla_ms)] = "warn"

        dfs.append(pd.DataFrame({
            "timestamp":    raw_rt["timestamp"],       # Series keeps tz dtype
            "source":       "iis",
            "metric_name":  "response_time_ms",
            "value":        tt,
            "unit":         "ms",
            "severity":     rt_sev,
            "raw_line":     "",
            "uri_stem":     uri_stem.loc[raw_rt.index],
            "uri_query":    uri_query.loc[raw_rt.index],
            "method":       method.loc[raw_rt.index],
            "status_code":  status_code.loc[raw_rt.index],
            "substatus":    substatus.loc[raw_rt.index],
            "win32_status": win32_status.loc[raw_rt.index],
            "client_ip":    client_ip.loc[raw_rt.index],
            "username":     username.loc[raw_rt.index],
            "user_agent":   user_agent.loc[raw_rt.index],
            "bytes_recv":   bytes_recv.loc[raw_rt.index],
            "bytes_sent":   bytes_sent_col.loc[raw_rt.index],
            "s_ip":         s_ip.loc[raw_rt.index],
            "s_port":       s_port.loc[raw_rt.index],
            "sitename":     sitename.loc[raw_rt.index],
        }))

    # ── 5b: one http_status row per request (for error counting) ─────────────
    dfs.append(pd.DataFrame({
        "timestamp":    raw["timestamp"],              # Series keeps tz dtype
        "source":       "iis",
        "metric_name":  "http_status",
        "value":        status_code.astype(float),
        "unit":         "",
        "severity":     http_sev,
        "raw_line":     "",
        "uri_stem":     uri_stem,
        "uri_query":    uri_query,
        "method":       method,
        "status_code":  status_code,
        "substatus":    substatus,
        "win32_status": win32_status,
        "client_ip":    client_ip,
        "username":     username,
        "user_agent":   user_agent,
        "bytes_recv":   bytes_recv,
        "bytes_sent":   bytes_sent_col,
        "s_ip":         s_ip,
        "s_port":       s_port,
        "sitename":     sitename,
    }))

    # ── 5c: bytes sent (sc-bytes) — kept for backwards compat ─────────────────
    if "sc-bytes" in raw.columns:
        mask_sb = pd.to_numeric(raw["sc-bytes"], errors="coerce").notna()
        raw_sb  = raw[mask_sb].copy()
        sb      = pd.to_numeric(raw_sb["sc-bytes"], errors="coerce")
        if not sb.empty:
            dfs.append(pd.DataFrame({
                "timestamp":    raw_sb["timestamp"],   # Series keeps tz dtype
                "source":       "iis",
                "metric_name":  "bytes_sent",
                "value":        sb,
                "unit":         "B",
                "severity":     "info",
                "raw_line":     "",
                "uri_stem":     uri_stem.loc[raw_sb.index],
                "uri_query":    uri_query.loc[raw_sb.index],
                "method":       method.loc[raw_sb.index],
                "status_code":  status_code.loc[raw_sb.index],
                "substatus":    substatus.loc[raw_sb.index],
                "win32_status": win32_status.loc[raw_sb.index],
                "client_ip":    client_ip.loc[raw_sb.index],
                "username":     username.loc[raw_sb.index],
                "user_agent":   user_agent.loc[raw_sb.index],
                "bytes_recv":   bytes_recv.loc[raw_sb.index],
                "bytes_sent":   bytes_sent_col.loc[raw_sb.index],
                "s_ip":         s_ip.loc[raw_sb.index],
                "s_port":       s_port.loc[raw_sb.index],
                "sitename":     sitename.loc[raw_sb.index],
            }))

    if not dfs:
        return pd.DataFrame()

    df = pd.concat(dfs, ignore_index=True)
    # Re-cast after concat to guarantee tz-aware UTC (handles any edge-case dtype drift)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    log.info(f"[iis_parser] Done: {len(df):,} rows  "
             f"span={df['timestamp'].min()} → {df['timestamp'].max()}")
    return df
