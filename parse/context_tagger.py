"""
Context Tagger
--------------
Enriches the normalised DataFrame with test-phase labels
(ramp_up | steady_state | ramp_down | unknown) based on:

  1. Explicit phase metadata from LoadRunner logs (preferred).
  2. Config-defined time buckets relative to test start.

Also stamps each row with a `run_id` derived from the earliest timestamp.
"""
from __future__ import annotations

import hashlib

import pandas as pd

from Utils.logger import get_logger

log = get_logger(__name__)

_PHASE_COL = "phase"


def tag(df: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """
    Add / refine phase labels and run_id to every row.

    Args:
        df:  Normalised DataFrame (all sources merged OK).
        cfg: Pipeline config dict.

    Returns:
        DataFrame with 'phase' and 'run_id' columns populated.
    """
    if df.empty:
        return df

    cfg = cfg or {}
    phases_cfg = cfg.get("test_phases", {})
    ramp_up_min   = float(phases_cfg.get("ramp_up_minutes",    5))
    steady_min    = float(phases_cfg.get("steady_state_minutes", 20))
    ramp_down_min = float(phases_cfg.get("ramp_down_minutes",  5))

    df = df.copy()

    # ── Run ID (unique per test run based on earliest timestamp) ─────────────
    t_min = df["timestamp"].min()
    run_id = hashlib.md5(str(t_min).encode()).hexdigest()[:8]
    df["run_id"] = run_id

    # ── Phase tagging ─────────────────────────────────────────────────────────
    # Rows that already have a valid phase from LR logs → keep them.
    known_phases = {"ramp_up", "steady_state", "ramp_down"}
    mask_unknown = ~df[_PHASE_COL].isin(known_phases)

    if mask_unknown.any():
        elapsed_sec = (df.loc[mask_unknown, "timestamp"] - t_min).dt.total_seconds()

        ramp_up_end   = ramp_up_min   * 60
        steady_end    = ramp_up_end   + steady_min * 60
        ramp_down_end = steady_end    + ramp_down_min * 60

        conditions = [
            elapsed_sec <= ramp_up_end,
            (elapsed_sec > ramp_up_end) & (elapsed_sec <= steady_end),
            (elapsed_sec > steady_end)  & (elapsed_sec <= ramp_down_end),
        ]
        choices = ["ramp_up", "steady_state", "ramp_down"]
        phase_series = pd.Series("unknown", index=elapsed_sec.index)
        for cond, choice in zip(conditions, choices):
            phase_series[cond] = choice

        df.loc[mask_unknown, _PHASE_COL] = phase_series

    phase_counts = df[_PHASE_COL].value_counts().to_dict()
    log.info(f"[tagger] run_id={run_id} phase distribution: {phase_counts}")
    return df
