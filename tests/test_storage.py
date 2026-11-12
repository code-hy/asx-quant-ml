"""Tests for the storage layer: lake paths, Parquet conventions, DuckDB."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from asx_quant.config.loader import Settings, load_settings
from asx_quant.exceptions import StorageError
from asx_quant.storage.duckdb import Warehouse
from asx_quant.storage.parquet import ParquetWriter, inspect_dataset, read_parquet
from asx_quant.storage.paths import DataLake, batch_id


@pytest.fixture()
def settings(repo_root: Path) -> Settings:
    return load_settings(repo_root)


@pytest.fixture()
def sandbox_root(tmp_path: Path) -> Path:
    return tmp_path


@pytest.fixture()
def sandbox_settings(sandbox_root: Path) -> Settings:
    """Settings rooted at a temp dir (data/ subdirectory)."""
    return Settings(project_root_dir=sandbox_root, raw={
        "paths": {"data_root": "data"},
        "storage": {
            "parquet": {"compression": "zstd", "compression_level": 1},
            "duckdb": {"schemas": ["bronze", "silver", "gold", "meta"]},
        },
    })


# ------------------------------------------------------------------ DataLake
def test_lake_ensure_all_creates_skeleton(sandbox_root: Path) -> None:
    lake = DataLake(root=sandbox_root / "data")
    lake.ensure_all()
    for domain in ("prices", "security_master", "rba", "abs", "asic", "announcements"):
        assert (lake.bronze_dir / domain).is_dir()
    for layer in ("silver", "gold", "models", "reports"):
        assert (lake.root / layer).is_dir()


def test_bronze_batch_paths_are_append_only(sandbox_root: Path) -> None:
    lake = DataLake(root=sandbox_root / "data")
    p1 = lake.bronze_batch_path("prices", "CBA.AX", "20250101T000000Z")
    p1.touch()
    p2 = lake.bronze_batch_path("prices", "CBA.AX", "20250102T000000Z")
    p2.touch()
    batches = lake.bronze_batches("prices", "CBA.AX")
    assert batches == [p1, p2]  # sorted, oldest first


def test_unknown_bronze_domain_rejected(sandbox_root: Path) -> None:
    lake = DataLake(root=sandbox_root / "data")
    with pytest.raises(ValueError, match="Unknown bronze domain"):
        lake.bronze_domain_dir("not_a_domain")


def test_batch_id_is_sortable(sandbox_root: Path) -> None:
    assert batch_id().endswith("Z") and len(batch_id()) == 16


# ------------------------------------------------------------------- Parquet
def test_parquet_roundtrip_stamps_audit_columns(sandbox_settings: Settings) -> None:
    writer = ParquetWriter(sandbox_settings)
    df = pd.DataFrame({"ticker": ["CBA.AX"], "close": [150.0]})
    path = sandbox_settings.data_root / "test.parquet"
    n = writer.write(df, path, run_id="run-123", dataset_name="test")
    assert n == 1
    back = read_parquet(path)
    assert back["ingestion_run_id"].iloc[0] == "run-123"
    assert back["retrieved_at"].notna().all()


def test_parquet_refuses_silent_overwrite(sandbox_settings: Settings) -> None:
    writer = ParquetWriter(sandbox_settings)
    df = pd.DataFrame({"a": [1]})
    path = sandbox_settings.data_root / "test.parquet"
    writer.write(df, path, run_id="r1")
    with pytest.raises(StorageError, match="overwrite"):
        writer.write(df, path, run_id="r2")
    # explicit overwrite is allowed
    writer.write(df, path, run_id="r3", overwrite=True)
    assert read_parquet(path)["ingestion_run_id"].iloc[0] == "r3"


def test_parquet_refuses_duplicate_columns(sandbox_settings: Settings) -> None:
    writer = ParquetWriter(sandbox_settings)
    df = pd.DataFrame([[1, 2]], columns=["a", "a"])
    with pytest.raises(StorageError, match="duplicate columns"):
        writer.write(df, sandbox_settings.data_root / "dupe.parquet", run_id="r1")


def test_inspect_dataset_reports_schema(sandbox_settings: Settings) -> None:
    writer = ParquetWriter(sandbox_settings)
    path = sandbox_settings.data_root / "meta.parquet"
    writer.write(pd.DataFrame({"x": [1, 2, 3]}), path, run_id="r1")
    info = inspect_dataset(path)
    assert info["num_rows"] == 3
    assert "x" in info["columns"]


def test_read_missing_parquet_raises(sandbox_root: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_parquet(sandbox_root / "ghost.parquet")


# -------------------------------------------------------------------- DuckDB
def test_warehouse_initialises_schemas_and_meta(sandbox_root: Path) -> None:
    wh = Warehouse(sandbox_root / "data" / "warehouse.duckdb")
    with wh:
        rows = wh.query(
            "SELECT schema_name FROM information_schema.schemata "
            "WHERE schema_name IN ('bronze','silver','gold','meta') "
            "ORDER BY schema_name"
        )
        assert list(rows["schema_name"]) == ["bronze", "gold", "meta", "silver"]
        scalar = wh.query_scalar("SELECT COUNT(*) FROM meta.warehouse_meta")
        assert scalar is not None and scalar[0] >= 1


def test_warehouse_register_view_and_query(sandbox_root: Path) -> None:
    lake = DataLake(root=sandbox_root / "data")
    lake.ensure_all()
    df = pd.DataFrame({"ticker": ["A", "B"], "value": [1.0, 2.0]})
    path = lake.silver_dataset_path("probe")
    df.to_parquet(path)

    wh = Warehouse(sandbox_root / "data" / "warehouse.duckdb")
    with wh:
        wh.register_view("silver", "probe", path, layer="silver")
        result = wh.query("SELECT * FROM silver.probe ORDER BY ticker")
        assert list(result["ticker"]) == ["A", "B"]
        reg = wh.query("SELECT * FROM meta.dataset_registry WHERE dataset_name='probe'")
        assert reg.iloc[0]["record_count"] == 2


def test_warehouse_rejects_bad_layer(sandbox_root: Path) -> None:
    wh = Warehouse(sandbox_root / "data" / "warehouse.duckdb")
    with wh:
        with pytest.raises(StorageError, match="Invalid layer"):
            wh.register_view("gold", "x", sandbox_root / "a.parquet", layer="raw")
