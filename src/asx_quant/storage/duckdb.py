"""DuckDB analytical warehouse over the Parquet lake.

The DuckDB file is a *query* layer, not the system of record: Parquet remains
the durable source of truth (see docs/architecture.md). This module creates
schemas, metadata tables and lazily-materialised views. Recreating the
warehouse is always safe because it is rebuilt from Parquet.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import duckdb
import pandas as pd

from asx_quant.exceptions import StorageError
from asx_quant.observability.logging import get_logger

logger = get_logger("storage.duckdb")

#: SQL executed at warehouse initialisation. Kept as one deterministic list so
#: the schema is reproducible and testable.
SCHEMA_DDL: list[str] = [
    "CREATE SCHEMA IF NOT EXISTS meta",
    "CREATE SCHEMA IF NOT EXISTS bronze",
    "CREATE SCHEMA IF NOT EXISTS silver",
    "CREATE SCHEMA IF NOT EXISTS gold",
    """
    CREATE TABLE IF NOT EXISTS meta.warehouse_meta (
        key            VARCHAR PRIMARY KEY,
        value          VARCHAR,
        updated_at     TIMESTAMPTZ
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS meta.dataset_registry (
        dataset_name   VARCHAR,
        layer          VARCHAR,
        parquet_path   VARCHAR,
        record_count   BIGINT,
        last_refreshed TIMESTAMPTZ,
        PRIMARY KEY (dataset_name, layer)
    )
    """,
]

#: Registered Parquet views are created lazily via :meth:`Warehouse.register_view`.
_VIEW_DDL = """
    CREATE OR REPLACE VIEW {qualified_name} AS
    SELECT * FROM read_parquet('{parquet_path}')
"""


class Warehouse:
    """A DuckDB connection bound to the project's data lake."""

    def __init__(self, duckdb_path: Path) -> None:
        self._path = duckdb_path
        self._conn: duckdb.DuckDBPyConnection | None = None

    # ------------------------------------------------------------ lifecycle
    def connect(self) -> duckdb.DuckDBPyConnection:
        """Open (or return the existing) connection, initialising schemas."""
        if self._conn is None:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._conn = duckdb.connect(str(self._path))
            except duckdb.Error as exc:
                raise StorageError(
                    f"Could not open DuckDB warehouse at '{self._path}': {exc}"
                ) from exc
            self.initialise()
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> Warehouse:
        self.connect()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __iter__(self) -> Iterator:  # pragma: no cover - not used, avoids misuse
        raise TypeError(
            "Warehouse is not iterable; use query() or context-manager protocol."
        )

    def initialise(self) -> None:
        """Create schemas and metadata tables (idempotent)."""
        conn = self._conn
        if conn is None:
            raise StorageError(
                "initialise() called before connect(); use connect() instead."
            )
        for ddl in SCHEMA_DDL:
            stmt = " ".join(ddl.split())
            conn.execute(stmt)
        conn.execute(
            """
            INSERT INTO meta.warehouse_meta (key, value, updated_at)
            VALUES ('duckdb_initialised_at', now()::VARCHAR, now())
            ON CONFLICT (key) DO UPDATE
              SET value = EXCLUDED.value, updated_at = EXCLUDED.updated_at
            """
        )
        logger.info("DuckDB warehouse initialised at '%s'", self._path)

    # ------------------------------------------------------------------ views
    def register_view(
        self,
        schema: str,
        view_name: str,
        parquet_path: Path,
        layer: str,
    ) -> None:
        """Expose a Parquet dataset as a queryable view and register it.

        Args:
            schema: Target schema (bronze, silver or gold).
            view_name: View name, e.g. ``equity_prices``.
            parquet_path: Backing Parquet file (may be a glob for partitions).
            layer: Layer label for the registry table.
        """
        conn = self.connect()
        if layer not in {"bronze", "silver", "gold"}:
            raise StorageError(
                f"Invalid layer '{layer}'; must be bronze, silver or gold."
            )
        # DuckDB string literals treat backslashes as escape characters, so
        # Windows paths must be normalised to forward slashes.
        posix_path = str(parquet_path).replace("\\", "/").replace("'", "''")
        conn.execute(
            _VIEW_DDL.format(
                qualified_name=f"{schema}.{view_name}",
                parquet_path=posix_path,
            )
        )
        record_count = conn.execute(
            f"SELECT COUNT(*) FROM {schema}.{view_name}"
        ).fetchone()[0]
        conn.execute(
            """
            INSERT INTO meta.dataset_registry
                (dataset_name, layer, parquet_path, record_count, last_refreshed)
            VALUES (?, ?, ?, ?, now())
            ON CONFLICT (dataset_name, layer) DO UPDATE
              SET parquet_path   = EXCLUDED.parquet_path,
                  record_count   = EXCLUDED.record_count,
                  last_refreshed = EXCLUDED.last_refreshed
            """,
            [view_name, layer, posix_path, record_count],
        )
        logger.info(
            "Registered view %s.%s (%d rows) over '%s'",
            schema, view_name, record_count, parquet_path,
        )

    # ---------------------------------------------------------------- queries
    def query(self, sql: str, params: list | None = None) -> pd.DataFrame:
        """Run a query and return a dataframe."""
        conn = self.connect()
        try:
            return conn.execute(sql, params or []).fetchdf()
        except duckdb.Error as exc:
            raise StorageError(f"Query failed: {exc}\nSQL: {sql}") from exc

    def query_scalar(self, sql: str, params: list | None = None) -> tuple | None:
        """Run a query expected to return a single row (or none)."""
        conn = self.connect()
        try:
            return conn.execute(sql, params or []).fetchone()
        except duckdb.Error as exc:
            raise StorageError(f"Query failed: {exc}\nSQL: {sql}") from exc


def get_warehouse(settings) -> Warehouse:  # noqa: ANN001
    """Factory bound to the configured warehouse file."""
    return Warehouse(settings.duckdb_path)
