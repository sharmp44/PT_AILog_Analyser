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
    run_id           VARCHAR,
    -- IIS extended fields (NULL for non-IIS sources)
    substatus        INTEGER,
    win32_status     INTEGER,
    uri_query        VARCHAR,
    bytes_recv       DOUBLE,
    bytes_sent       DOUBLE,
    user_agent       VARCHAR,
    username         VARCHAR,
    s_ip             VARCHAR,
    s_port           VARCHAR,
    sitename         VARCHAR
);
"""

# Canonical column list — every DataFrame inserted must provide exactly these columns
# (extras are dropped, missing ones are filled with NULL/0/empty string before insert)
_SCHEMA_COLS: list[tuple[str, object]] = [
    ("timestamp",        None),
    ("source",           ""),
    ("metric_name",      ""),
    ("value",            0.0),
    ("unit",             ""),
    ("severity",         "info"),
    ("raw_line",         ""),
    ("uri_stem",         ""),
    ("method",           ""),
    ("status_code",      0),
    ("client_ip",        ""),
    ("transaction_name", ""),
    ("vuser_id",         ""),
    ("phase",            ""),
    ("run_id",           ""),
    ("substatus",        0),
    ("win32_status",     0),
    ("uri_query",        ""),
    ("bytes_recv",       0.0),
    ("bytes_sent",       0.0),
    ("user_agent",       ""),
    ("username",         ""),
    ("s_ip",             ""),
    ("s_port",           ""),
    ("sitename",         ""),
]
_SCHEMA_COL_NAMES = [c for c, _ in _SCHEMA_COLS]

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
        self._migrate_schema()
        self._con.execute(_CREATE_INDEX)
        log.info(f"[store] Connected to {self.db_path}")

    def _migrate_schema(self) -> None:
        """Add any new columns that exist in _SCHEMA_COLS but not in the table yet.

        This keeps the on-disk DuckDB compatible when new fields are added to the
        schema without requiring a full database recreate.
        """
        try:
            existing = {
                row[0].lower()
                for row in self._con.execute(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'events'"
                ).fetchall()
            }
            _type_map = {
                int:   "INTEGER",
                float: "DOUBLE",
                str:   "VARCHAR",
                type(None): "VARCHAR",
            }
            for col, default in _SCHEMA_COLS:
                if col.lower() not in existing:
                    sql_type = _type_map.get(type(default), "VARCHAR")
                    self._con.execute(f"ALTER TABLE events ADD COLUMN {col} {sql_type}")
                    log.info(f"[store] Migrated: added column '{col}' ({sql_type})")
        except Exception as exc:
            log.warning(f"[store] Schema migration warning: {exc}")

    # ── Write ────────────────────────────────────────────────────────────────

    def insert(self, df: pd.DataFrame) -> int:
        """Insert a normalised DataFrame into the events table.

        The DataFrame may contain extra columns (e.g. IIS-specific fields like
        uri_query, substatus) or be missing optional columns (e.g. LoadRunner
        rows have no uri_stem). We align it to the schema before inserting so
        the column count always matches regardless of source type.
        """
        if df.empty:
            return 0

        # Build an aligned copy: add missing schema cols with defaults, drop extras
        aligned = df.copy()
        for col, default in _SCHEMA_COLS:
            if col not in aligned.columns:
                aligned[col] = default
        aligned = aligned[_SCHEMA_COL_NAMES]  # exact schema order, no extras

        self._con.execute("INSERT INTO events SELECT * FROM aligned")
        count = len(aligned)
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
