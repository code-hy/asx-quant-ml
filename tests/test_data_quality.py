"""Tests for the data-quality rule engine."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from asx_quant.config.loader import load_settings
from asx_quant.exceptions import DataQualityError
from asx_quant.quality.checks import (
    QualityReport,
    RuleResult,
    run_price_quality_checks,
)


def _price_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": ["CBA.AX"] * 5,
        "date": pd.bdate_range("2024-01-01", periods=5),
        "open": [100.0, 101.0, 102.0, 103.0, 104.0],
        "high": [101.0, 102.0, 103.0, 104.0, 105.0],
        "low": [99.0, 100.0, 101.0, 102.0, 103.0],
        "close": [100.5, 101.5, 102.5, 103.5, 104.5],
        "adj_close": [98.0, 99.0, 100.0, 101.0, 102.0],
        "volume": [1000, 1100, 1200, 1300, 1400],
    })


def _get(report: QualityReport, name: str) -> RuleResult:
    return next(r for r in report.results if r.name == name)


def test_clean_data_passes_all_rules() -> None:
    report = run_price_quality_checks(_price_frame())
    assert report.passed, [r.name for r in report.errors if not r.passed]
    assert report.to_dict()["status"] == "pass"


def test_duplicate_security_date_is_error() -> None:
    prices = pd.concat([_price_frame(), _price_frame().tail(1)])
    report = run_price_quality_checks(prices)
    rule = _get(report, "no_duplicate_security_date")
    assert rule.severity == "error" and not rule.passed
    assert rule.failures == 2  # keep=False counts both originals
    with pytest.raises(DataQualityError):
        report.assert_passed()


def test_price_bounds_rule() -> None:
    prices = _price_frame()
    prices.loc[0, "open"] = 0.0
    prices.loc[1, "close"] = 200000.0
    report = run_price_quality_checks(prices)
    assert not _get(report, "prices_positive_and_bounded").passed


def test_high_below_low_is_error() -> None:
    prices = _price_frame()
    prices.loc[2, "high"] = prices.loc[2, "low"] - 1.0
    report = run_price_quality_checks(prices)
    assert not _get(report, "high_not_below_low").passed


def test_negative_volume_is_error() -> None:
    prices = _price_frame()
    prices.loc[3, "volume"] = -5
    report = run_price_quality_checks(prices)
    assert not _get(report, "volume_non_negative").passed


def test_missing_dates_detected() -> None:
    prices = _price_frame()
    prices.loc[4, "date"] = pd.NaT
    report = run_price_quality_checks(prices)
    assert not _get(report, "trading_dates_valid").passed


def test_extreme_returns_flagged_as_warning_only() -> None:
    prices = _price_frame()
    prices.loc[3, "adj_close"] = prices.loc[2, "adj_close"] * 2.0  # +100%
    report = run_price_quality_checks(prices)
    rule = _get(report, "extreme_single_session_returns")
    assert not rule.passed and rule.severity == "warn"
    assert report.passed  # warn severity never blocks the stage


def test_missing_sessions_measured_not_filled() -> None:
    # Two tickers: a complete one establishes the market-calendar baseline so
    # the incomplete ticker's gap is detectable (impossible with one ticker).
    complete = _price_frame()
    other = complete.copy()
    other["ticker"] = "BHP.AX"
    prices = pd.concat([complete, other], ignore_index=True).drop(index=2)
    report = run_price_quality_checks(prices)
    rule = _get(report, "missing_sessions_measured")
    assert not rule.passed
    assert rule.failures == 1
    assert any("CBA.AX" in s for s in rule.sample)


def test_empty_dataset_is_error() -> None:
    report = run_price_quality_checks(
        pd.DataFrame(columns=["ticker", "date", "open", "high", "low",
                              "close", "adj_close", "volume"])
    )
    assert not report.passed
    assert not _get(report, "non_empty_dataset").passed


def test_quality_report_persistence(tmp_path: Path) -> None:
    report = run_price_quality_checks(_price_frame())
    path = report.save(tmp_path / "data_quality.json")
    assert path.is_file()
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["dataset"] == "bronze_prices"
    assert payload["status"] == "pass"


def test_report_against_real_settings(repo_root: Path) -> None:
    settings = load_settings(repo_root)
    assert float(settings.raw["quality"]["extreme_return_threshold"]) == 0.5
