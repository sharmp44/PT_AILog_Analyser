"""
DuckDB Time-Series Store
------------------------
Persists normalised + tagged events for the lifetime of a pipeline run.
Provides fast window-query helpers used by the AI agents.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from Utils.logger import get_logger

log = get_logger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS events (
    timestamp        TIMESTAMPTZ NOT NULL,
    source           VARCHAR,
    metric_name      VARCHAR,
    value            DOUBLE,
    unit             VARCHAR,
    severity         VARCHAR,
    raw_line         VARCHAR,
    uri_stem         VARCHAR,
    method           VARCHAR,
    status_code      INTEGER,
    client_ip        VARCHAR,
    transaction_name VARCHAR,
    vuser_id         VARCHAR,
    phase            VARCHAR,
    run_id           VARCHAR
);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_ts_metric ON events (timestamp, metric_name);
"""


class Store:
    """Thin wrapper around a DuckDB connection for the pipeline events table."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(str(self.db_path))
        self._con.execute(_CREATE_TABLE)
        self._con.execute(_CREATE_INDEX)
        log.info(f"[store] Connected to {self.db_path}")

    # ── Write ────────────────────────────────────────────────────────────────

    def insert(self, df: pd.DataFrame) -> int:
        """Insert a normalised DataFrame into the events table."""
        if df.empty:
            return 0
        self._con.execute("INSERT INTO events SELECT * FROM df")
        count = len(df)
        log.info(f"[store] Inserted {count} rows")
        return count

    # ── Read ─────────────────────────────────────────────────────────────────

    def query(self, sql: str) -> pd.DataFrame:
        return self._con.execute(sql).df()

    def get_metric(
        self,
        metric_name: str,
        source: str | None = None,
        phase: str | None = None,
        run_id: str | None = None,
    ) -> pd.DataFrame:
        """Fetch all rows for a specific metric, optionally filtered."""
        clauses = [f"metric_name = '{metric_name}'"]
        if source:
            clauses.append(f"source = '{source}'")
        if phase:
            clauses.append(f"phase = '{phase}'")
        if run_id:
            clauses.append(f"run_id = '{run_id}'")
        where = " AND ".join(clauses)
        return self._con.execute(
            f"SELECT * FROM events WHERE {where} ORDER BY timestamp"
        ).df()

    def get_all(self, run_id: str | None = None) -> pd.DataFrame:
        if run_id:
            return self._con.execute(
                f"SELECT * FROM events WHERE run_id = '{run_id}' ORDER BY timestamp"
            ).df()
        return self._con.execute("SELECT * FROM events ORDER BY timestamp").df()

    def window_stats(
        self,
        metric_name: str,
        window_seconds: int = 60,
        run_id: str | None = None,
    ) -> pd.DataFrame:
        """
        Rolling window statistics (mean, std, min, max, count) per window bucket.
        Returns one row per window bucket.
        """
        run_filter = f"AND run_id = '{run_id}'" if run_id else ""
        sql = f"""
        SELECT
            time_bucket(INTERVAL '{window_seconds} seconds', timestamp) AS bucket,
            AVG(value)   AS mean,
            STDDEV(value) AS std,
            MIN(value)   AS min,
            MAX(value)   AS max,
            COUNT(*)     AS count
        FROM events
        WHERE metric_name = '{metric_name}' {run_filter}
        GROUP BY bucket
        ORDER BY bucket
        """
        return self._con.execute(sql).df()

    def list_metrics(self, run_id: str | None = None) -> list[str]:
        run_filter = f"WHERE run_id = '{run_id}'" if run_id else ""
        rows = self._con.execute(
            f"SELECT DISTINCT metric_name FROM events {run_filter} ORDER BY metric_name"
        ).fetchall()
        return [r[0] for r in rows]

    def get_run_id(self) -> str | None:
        """Return the most recent run_id in the store."""
        row = self._con.execute(
            "SELECT run_id FROM events ORDER BY timestamp DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else None

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._con.close()
        log.info("[store] Connection closed")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
