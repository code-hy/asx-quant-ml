"""Tests for ingestion: Yahoo downloader, universe, normalisation."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from asx_quant.exceptions import IngestionError
from asx_quant.ingestion.universe import (
    _clean_codes,
    add_master_metadata,
    build_universe,
)
from asx_quant.ingestion.yahoo import (
    YahooDownloader,
    append_ingestion_log,
    history_start,
    normalise_raw_frame,
)
from asx_quant.storage.paths import DataLake


def _fake_yf_frame(dates: pd.DatetimeIndex, base: float = 100.0) -> pd.DataFrame:
    n = len(dates)
    return pd.DataFrame(
        {
            "Open": [base] * n,
            "High": [base + 1] * n,
            "Low": [base - 1] * n,
            "Close": [base] * n,
            "Adj Close": [base * 0.98] * n,
            "Volume": [1000] * n,
            "Dividends": [0.0] * n,
            "Stock Splits": [1.0] * n,
        },
        index=dates,
    )


def test_downloader_batches_and_reports_failures() -> None:
    trading_days = pd.bdate_range("2024-01-01", periods=5)
    calls: list[list[str]] = []

    def fake_fn(tickers, start, end, **kwargs):
        calls.append(list(tickers))
        return {
            t: (_fake_yf_frame(trading_days) if t != "BAD.AX" else pd.DataFrame())
            for t in tickers
        }

    downloader = YahooDownloader(
        batch_size=2, batch_pause_seconds=0, max_retries=1,
        retry_backoff_seconds=0, download_fn=fake_fn,
    )
    frames, failed = downloader.download(["A.AX", "B.AX", "BAD.AX"],
                                         date(2024, 1, 1))
    # 2 batches of 2 + individual retry for the failed ticker
    assert len(calls) == 3
    assert frames["A.AX"].shape[0] == 5
    assert failed == ["BAD.AX"]


def test_downloader_retries_then_succeeds() -> None:
    trading_days = pd.bdate_range("2024-01-01", periods=3)
    attempts = {"n": 0}

    def flaky_fn(tickers, start, end, **kwargs):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise IngestionError("provider hiccup")
        return {t: _fake_yf_frame(trading_days) for t in tickers}

    downloader = YahooDownloader(
        batch_size=5, batch_pause_seconds=0, max_retries=3,
        retry_backoff_seconds=0, download_fn=flaky_fn,
    )
    frames, failed = downloader.download(["A.AX"], date(2024, 1, 1))
    assert attempts["n"] == 3 and failed == [] and not frames["A.AX"].empty


def test_normalise_raw_frame_schema_and_padding() -> None:
    idx = pd.DatetimeIndex(["2024-01-02", "2024-01-03"], name="Date")
    frame = _fake_yf_frame(idx)
    for col in ("Open", "High", "Low", "Close", "Adj Close", "Volume"):
        frame.iloc[-1, frame.columns.get_loc(col)] = float("nan")
    out = normalise_raw_frame("CBA.AX", frame)
    assert list(out.columns)[:3] == ["source_ticker", "source", "date"]
    assert out["source_ticker"].eq("CBA.AX").all()
    assert len(out) == 1  # all-NaN padding row dropped
    assert out["split_factor"].eq(1.0).all()


def test_normalise_empty_frame_returns_schema() -> None:
    out = normalise_raw_frame("X.AX", pd.DataFrame())
    assert out.empty and "date" in out.columns


def test_history_start_from_settings(repo_root: Path) -> None:
    from asx_quant.config.loader import load_settings

    settings = load_settings(repo_root)
    start = history_start(settings)
    assert start <= date.today() - timedelta(days=365 * 9)


def test_append_ingestion_log(tmp_path: Path) -> None:
    log_path = tmp_path / "prices" / "ingestion_log.jsonl"
    append_ingestion_log(log_path, [{"ticker": "A.AX", "records": 3}])
    append_ingestion_log(log_path, [{"ticker": "B.AX", "records": 4}])
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["ticker"] == "B.AX"


# ------------------------------------------------------------------ universe
_WIKI_HTML = """
<html><body>
<table class="wikitable">
<tr><th>Code</th><th>Company</th><th>GICS Sector</th><th>GICS Industry</th></tr>
<tr><td>CBA</td><td>Commonwealth Bank</td><td>Financials</td><td>Banks</td></tr>
<tr><td>bhp</td><td>BHP Group</td><td>Materials</td><td>Iron Ore</td></tr>
<tr><td>CBA</td><td>Commonwealth Bank</td><td>Financials</td><td>Banks</td></tr>
<tr><td>XX1</td><td>Not A Real Co</td><td>-</td><td>-</td></tr>
<tr><td>A!</td><td>Bad Code</td><td>-</td><td>-</td></tr>
</table>
</body></html>
"""


def test_clean_codes_normalises_and_dedupes(monkeypatch: pytest.MonkeyPatch) -> None:
    import io

    import pandas as pd

    tables = pd.read_html(io.StringIO(_WIKI_HTML))
    cleaned = _clean_codes(tables[0])
    assert sorted(cleaned["code"]) == ["BHP", "CBA", "XX1"]  # dup + bad code removed
    assert cleaned["sector"].isna().sum() == 1  # "-" placeholder → null


def test_build_universe_from_wikipedia(monkeypatch: pytest.MonkeyPatch,
                                       repo_root: Path) -> None:
    import io

    import pandas as pd

    from asx_quant.config.loader import load_settings

    def fake_fetch(user_agent=None):
        tables = pd.read_html(io.StringIO(_WIKI_HTML))
        return _clean_codes(tables[0])

    monkeypatch.setattr("asx_quant.ingestion.universe.fetch_asx200_constituents",
                        fake_fetch)
    settings = load_settings(repo_root)
    universe = build_universe(settings)
    assert "CBA.AX" in universe["ticker"].tolist()
    assert universe["universe_source"].eq("wikipedia_asx200").all()


def test_build_universe_falls_back_to_seed(monkeypatch: pytest.MonkeyPatch,
                                           repo_root: Path) -> None:
    from asx_quant.config.loader import load_settings

    def broken_fetch(user_agent=None):
        raise IngestionError("network unavailable")

    monkeypatch.setattr("asx_quant.ingestion.universe.fetch_asx200_constituents",
                        broken_fetch)
    settings = load_settings(repo_root)
    universe = build_universe(settings)
    assert universe["universe_source"].eq("config_seed").all()
    assert "CBA.AX" in universe["ticker"].tolist()


def test_add_master_metadata_completes_schema(repo_root: Path) -> None:
    from asx_quant.config.loader import load_settings

    settings = load_settings(repo_root)
    universe = build_universe(settings)
    master = add_master_metadata(universe, settings)
    for col in ("security_key", "exchange", "currency", "instrument_type",
                "active_flag", "source_identifier", "inclusion_methodology",
                "data_quality_status"):
        assert col in master.columns, f"missing {col}"
    assert master["exchange"].eq("ASX").all()
    assert master["security_key"].eq(master["ticker"]).all()


def test_bronze_master_batch_is_append_only(repo_root: Path,
                                            tmp_path: Path) -> None:
    from asx_quant.config.loader import load_settings
    from asx_quant.ingestion.universe import write_bronze_security_master
    from asx_quant.observability.runs import RunRegistry

    lake = DataLake(root=tmp_path / "data")
    lake.ensure_all()
    settings = load_settings(repo_root)
    registry = RunRegistry.start(lake, config_version="v0.1")
    df = pd.DataFrame({
        "ticker": ["CBA.AX"], "code": ["CBA"], "company_name": ["CBA Ltd"],
        "sector": ["Financials"], "industry": [None],
        "universe_source": ["wikipedia_asx200"],
    })
    registry2 = RunRegistry.start(lake, config_version="v0.1")
    p1 = write_bronze_security_master(lake, registry, settings, df)
    p2 = write_bronze_security_master(lake, registry2, settings, df)
    assert p1 != p2 and p1.exists() and p2.exists()  # both batches retained
