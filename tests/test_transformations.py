"""Tests for Bronze → Silver transformations: prices, calendar, coverage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from asx_quant.exceptions import StorageError
from asx_quant.storage.paths import DataLake
from asx_quant.transformations.calendar import (
    build_market_calendar,
    compute_coverage,
)
from asx_quant.transformations.prices import (
    load_bronze_prices,
    standardise_bronze_prices,
    to_silver_schema,
)


def _bronze_row(ticker: str, date: str, close: float, batch: str,
                adj: float | None = None) -> dict:
    return {
        "source_ticker": ticker, "source": "yfinance", "date": date,
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "adj_close": adj if adj is not None else close,
        "volume": 1000, "dividends": 0.0, "split_factor": 1.0,
        "batch_id": batch,
    }


def test_standardise_keeps_newest_batch() -> None:
    frames = {
        "CBA.AX": pd.DataFrame([
            _bronze_row("CBA.AX", "2024-01-02", 100.0, "batch1", adj=99.0),
            _bronze_row("CBA.AX", "2024-01-03", 101.0, "batch1", adj=100.0),
        ]),
    }
    newer = pd.DataFrame([
        _bronze_row("CBA.AX", "2024-01-02", 105.0, "batch2", adj=104.0),  # revised
        _bronze_row("CBA.AX", "2024-01-04", 106.0, "batch2", adj=105.0),
    ])
    result = standardise_bronze_prices({"CBA.AX": pd.concat([frames["CBA.AX"], newer])})
    assert len(result) == 3  # no duplicates
    day2 = result[result["date"] == pd.Timestamp("2024-01-02")]
    assert day2["close"].iloc[0] == 105.0  # newest batch wins


def test_to_silver_schema_daily_return_on_adjusted() -> None:
    prices = pd.DataFrame([
        _bronze_row("A.AX", "2024-01-01", 100.0, "b", adj=100.0),
        _bronze_row("A.AX", "2024-01-02", 110.0, "b", adj=105.0),  # adj +5%
        _bronze_row("A.AX", "2024-01-03", 120.0, "b", adj=110.0),  # adj +4.76%
    ])
    prices["date"] = pd.to_datetime(prices["date"])
    silver = to_silver_schema(prices)
    assert "date_key" in silver.columns
    assert silver["date_key"].iloc[0] == 20240101
    expected = [np.nan, 105 / 100 - 1, 110 / 105 - 1]
    np.testing.assert_allclose(silver["daily_return"].to_numpy(),
                               np.array(expected), rtol=1e-9, equal_nan=True)
    assert silver["split_factor"].notna().all()


def _write_bronze(lake: DataLake, ticker: str, rows: list[dict],
                  batch: str) -> None:
    path = lake.bronze_batch_path("prices", ticker, batch)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path)


def test_load_bronze_prices_tags_batch_id(tmp_path: Path) -> None:
    lake = DataLake(root=tmp_path)
    lake.ensure_all()
    _write_bronze(lake, "A.AX", [_bronze_row("A.AX", "2024-01-01", 100.0, "b1")], "b1")
    _write_bronze(lake, "A.AX", [_bronze_row("A.AX", "2024-01-02", 101.0, "b2")], "b2")
    _write_bronze(lake, "B.AX", [_bronze_row("B.AX", "2024-01-01", 50.0, "b1")], "b1")

    frames = load_bronze_prices(lake)
    assert set(frames) == {"A.AX", "B.AX"}
    assert sorted(frames["A.AX"]["batch_id"].tolist()) == ["b1", "b2"]


def test_load_bronze_empty_raises(tmp_path: Path) -> None:
    lake = DataLake(root=tmp_path)
    lake.ensure_all()
    with pytest.raises(StorageError, match="Run ingest first"):
        load_bronze_prices(lake)


# ------------------------------------------------------------------ calendar
def test_market_calendar_from_benchmark_sessions() -> None:
    sessions = pd.bdate_range("2024-01-01", periods=6)
    prices = pd.DataFrame({
        "ticker": ["IDX_AXJO"] * 4 + ["A.AX"] * 2,
        "date": list(sessions[:4]) + [sessions[0], sessions[5]],
        # A.AX's stray session (not a benchmark session) must be excluded
    })
    cal = build_market_calendar(prices)
    assert len(cal) == 4
    assert cal["trading_day_sequence"].tolist() == [0, 1, 2, 3]
    assert (cal["is_asx_trading_day"]).all()
    assert cal["financial_year"].iloc[0] == 2023  # January → FY2024 starts July


def test_market_calendar_falls_back_to_union() -> None:
    sessions = pd.bdate_range("2024-01-01", periods=3)
    prices = pd.DataFrame({
        "ticker": ["A.AX"] * 3, "date": sessions,
    })
    cal = build_market_calendar(prices)
    assert cal["calendar_source"].eq("union_of_sessions").all()
    assert len(cal) == 3


def test_coverage_measures_gaps_per_security() -> None:
    sessions = pd.bdate_range("2024-01-01", periods=5)
    cal = pd.DataFrame({"calendar_date": sessions})
    prices = pd.DataFrame({
        "ticker": ["A.AX"] * 3 + ["B.AX"] * 5,
        "date": [sessions[0], sessions[2], sessions[4]] + list(sessions),
    })
    coverage = compute_coverage(prices, cal)
    row_a = coverage[coverage["ticker"] == "A.AX"].iloc[0]
    assert row_a["missing_sessions"] == 2
    assert row_a["coverage_status"] == "minor_gaps"
    row_b = coverage[coverage["ticker"] == "B.AX"].iloc[0]
    assert row_b["missing_sessions"] == 0
    assert row_b["coverage_status"] == "ok"


def test_coverage_merges_master_attributes() -> None:
    sessions = pd.bdate_range("2024-01-01", periods=3)
    cal = pd.DataFrame({"calendar_date": sessions})
    prices = pd.DataFrame({"ticker": ["A.AX"] * 3, "date": sessions})
    master = pd.DataFrame({
        "security_key": ["A.AX"], "company_name": ["Alpha Ltd"],
        "sector": ["Financials"], "inclusion_methodology": ["wikipedia"],
    })
    coverage = compute_coverage(prices, cal, master)
    assert coverage.iloc[0]["company_name"] == "Alpha Ltd"


def test_full_silver_pipeline_end_to_end(tmp_path: Path, repo_root: Path) -> None:
    """Bronze batches → silver equity_prices + calendar + coverage, offline."""
    from asx_quant.config.loader import load_settings
    from asx_quant.pipeline.stages import run_transform

    lake = DataLake(root=tmp_path / "data")
    lake.ensure_all()
    sessions = pd.bdate_range("2024-01-01", periods=10)
    rows = []
    for i, d in enumerate(sessions):
        for ticker in ("CBA.AX", "BHP.AX"):
            rows.append(_bronze_row(ticker, str(d.date()), 100 + i, "b1"))
    rows.append(_bronze_row("IDX__AXJO", str(sessions[0].date()), 7500, "b1"))
    _write_bronze(lake, "CBA.AX", rows[:10], "b1")
    _write_bronze(lake, "BHP.AX", rows[10:20], "b1")
    _write_bronze(lake, "IDX__AXJO", [rows[20]], "b1")

    # bronze security master
    master = pd.DataFrame({
        "ticker": ["CBA.AX", "BHP.AX"], "code": ["CBA", "BHP"],
        "company_name": ["CBA Ltd", "BHP Group"], "sector": ["Financials", "Materials"],
        "industry": [None, None], "universe_source": ["wikipedia_asx200"] * 2,
        "security_key": ["CBA.AX", "BHP.AX"], "exchange": ["ASX"] * 2,
        "currency": ["AUD"] * 2, "instrument_type": ["ordinary_share"] * 2,
        "active_flag": [True] * 2, "source_identifier": ["CBA.AX", "BHP.AX"],
        "listing_date": [pd.NaT] * 2, "delisting_date": [pd.NaT] * 2,
        "inclusion_methodology": ["wikipedia_asx200"] * 2,
        "data_quality_status": ["pending_coverage_check"] * 2,
    })
    master_path = lake.bronze_batch_path("security_master", "asx200", "b1")
    master.to_parquet(master_path)

    settings = load_settings(repo_root)
    settings = type(settings)(project_root_dir=tmp_path, raw=settings.raw)
    summary = run_transform(settings)
    assert summary["securities"] == 2
    assert summary["records"] == 21  # 10 CBA + 10 BHP + 1 benchmark session
    assert summary["quality_status"] == "pass"
    assert (lake.silver_dir / "equity_prices.parquet").exists()
    assert (lake.silver_dir / "market_calendar.parquet").exists()
    assert (lake.silver_dir / "data_coverage.parquet").exists()
    assert (lake.silver_dir / "security_master.parquet").exists()
    cov_report = (tmp_path / "data" / "reports" / "data_coverage.json")
    assert cov_report.exists()
