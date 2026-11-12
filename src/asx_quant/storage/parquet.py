"""Parquet write/read helpers implementing the project's storage conventions.

Every dataset written through :class:`ParquetWriter` gains the audit columns
``ingestion_run_id`` and ``retrieved_at`` (if not already present) and is
written deterministically with configured compression, so identical inputs
produce byte-comparable schema and identical file-level statistics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from asx_quant.observability.logging import get_logger
from asx_quant.storage.paths import utc_now_iso

logger = get_logger("storage.parquet")

AUDIT_COLUMNS = ("ingestion_run_id", "retrieved_at")


class ParquetWriter:
    """Convention-enforcing Parquet writer for silver/gold datasets."""

    def __init__(self, settings) -> None:  # noqa: ANN001
        self._settings = settings

    def write(
        self,
        df: pd.DataFrame,
        path: Path,
        *,
        run_id: str,
        overwrite: bool = False,
        dataset_name: str = "",
    ) -> int:
        """Write a dataframe as Parquet with audit columns.

        Args:
            df: Dataframe to persist. Must not contain duplicate columns.
            path: Destination file path.
            run_id: Pipeline run identifier stamped into ``ingestion_run_id``.
            overwrite: Silver/Gold datasets may be regenerated (they are
                reproducible), but accidental overwrites must be explicit.
            dataset_name: Human-readable name for logging.

        Returns:
            Number of records written.

        Raises:
            StorageError: If ``overwrite=False`` and the file already exists.
        """
        import pyarrow  # noqa: F401  (kept local to keep import cost off import path)

        from asx_quant.exceptions import StorageError

        if df.columns.duplicated().any():
            duplicated = df.columns[df.columns.duplicated()].tolist()
            raise StorageError(
                f"Refusing to write dataset '{dataset_name or path.name}': "
                f"duplicate columns {duplicated}."
            )
        if path.exists() and not overwrite:
            raise StorageError(
                f"Refusing to overwrite '{path}' with overwrite=False. "
                "Silver/Gold regeneration must be explicit; Bronze batches "
                "must use a new batch id instead."
            )

        out = df.copy()
        out = _stamp_audit(out, run_id)

        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(out, preserve_index=False)
        pq.write_table(
            table,
            path,
            compression=self._settings.parquet_compression,
            compression_level=self._settings.parquet_compression_level,
            write_statistics=True,
        )
        logger.info(
            "Wrote %d rows to %s (%s)", len(out), path, dataset_name or path.name
        )
        return len(out)


def read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a Parquet file back into a dataframe."""
    if not path.is_file():
        raise FileNotFoundError(f"Parquet dataset not found: '{path}'")
    return pd.read_parquet(path, columns=columns)


def inspect_dataset(path: Path) -> dict[str, Any]:
    """Return lightweight metadata (schema, row count) without loading data."""
    if not path.is_file():
        raise FileNotFoundError(f"Parquet dataset not found: '{path}'")
    pf = pq.ParquetFile(path)
    return {
        "path": str(path),
        "num_rows": pf.metadata.num_rows,
        "num_columns": pf.metadata.num_columns,
        "columns": [c.name for c in pf.schema_arrow],
        "created_by": pf.metadata.created_by,
    }


def _stamp_audit(df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    if "ingestion_run_id" not in df.columns:
        df = df.assign(ingestion_run_id=run_id)
    if "retrieved_at" not in df.columns:
        df = df.assign(retrieved_at=utc_now_iso())
    return df


__all__ = ["AUDIT_COLUMNS", "ParquetWriter", "inspect_dataset", "read_parquet"]
