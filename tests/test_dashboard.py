"""Phase 9 tests: Streamlit dashboard smoke tests (AppTest).

Each page must render WITHOUT exceptions on both an empty lake (missing
artifacts -> "run stage X first" guidance) and a populated lake (numbers
from real artifacts). These are render-integrity tests: no fabrication,
no crashes, disclaimer always present.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

REPO = Path(__file__).resolve().parents[1]
DASHBOARD = REPO / "dashboard"


@pytest.fixture()
def empty_lake(tmp_path, monkeypatch) -> Path:
    root = tmp_path / "data"
    (root / "reports" / "runs").mkdir(parents=True)
    (root / "gold").mkdir()
    monkeypatch.setenv("ASX_QUANT_DATA_ROOT", str(root))
    monkeypatch.syspath_prepend(str(DASHBOARD))
    return root


def _populate(root: Path) -> None:
    curve = pd.DataFrame({
        "date": pd.bdate_range("2024-01-01", periods=10),
        "nav": [100.0 + i for i in range(10)],
        "cash": [10.0] * 10,
        "positions_value": [90.0 + i for i in range(10)],
        "n_positions": [20] * 10,
        "benchmark_nav": [100.0 + 0.5 * i for i in range(10)],
    })
    curve.to_parquet(root / "gold" / "backtest_equity_curve.parquet",
                     index=False)
    (root / "reports" / "backtest_metrics.json").write_text(json.dumps({
        "n_sessions": 10, "first_date": "2024-01-01",
        "last_date": "2024-01-14", "initial_nav": 100.0, "final_nav": 109.0,
        "total_return": 0.09, "cagr": 0.05, "annual_volatility": 0.15,
        "sharpe_ratio": 0.3, "sortino_ratio": 0.4, "max_drawdown": -0.05,
        "hit_rate": 0.55, "benchmark_total_return": 0.05,
        "excess_return": 0.04, "information_ratio": 0.2, "beta": 0.8,
        "alpha_annualised": 0.03, "cost_scenario": "base",
        "costs": {"brokerage_per_trade_aud": 9.5, "brokerage_percent": 0.0005,
                  "slippage_percent": 0.001},
        "diagnostics": {"skipped_dust_trades": 3},
        "test_window": {"total_return": 0.02, "sharpe_ratio": 0.1,
                        "max_drawdown": -0.02},
    }), encoding="utf-8")
    (root / "reports" / "backtest_quality.json").write_text(json.dumps({
        "dataset": "backtest", "status": "warn", "warn_failures": 3,
        "rules": []}), encoding="utf-8")


def _run(page: Path) -> AppTest:
    app = AppTest.from_file(str(page), default_timeout=60)
    app.run()
    return app


@pytest.mark.parametrize("page", [
    "Home.py",
    "pages/1_Backtest.py",
    "pages/2_Portfolio.py",
    "pages/3_Sensitivity.py",
    "pages/4_Data_Quality.py",
])
def test_pages_render_on_empty_lake(empty_lake, page) -> None:
    """Missing artifacts -> guidance, never fabricated numbers, no crash."""
    app = _run(DASHBOARD / page)
    assert not app.exception, f"page {page} raised: {app.exception}"


def test_home_renders_disclaimer_and_not_run_state(empty_lake) -> None:
    app = _run(DASHBOARD / "Home.py")
    assert not app.exception
    warnings_text = "\n".join(w.body for w in app.warning)
    assert "Not financial advice" in warnings_text


@pytest.mark.parametrize("page", [
    "Home.py",
    "pages/1_Backtest.py",
    "pages/2_Portfolio.py",
    "pages/3_Sensitivity.py",
    "pages/4_Data_Quality.py",
])
def test_pages_render_with_populated_lake(empty_lake, page) -> None:
    _populate(empty_lake)
    app = _run(DASHBOARD / page)
    assert not app.exception, f"page {page} raised: {app.exception}"


def test_backtest_page_shows_real_numbers(empty_lake) -> None:
    _populate(empty_lake)
    app = _run(DASHBOARD / "pages/1_Backtest.py")
    assert not app.exception
    metric_values = " ".join(str(m.value) for m in app.metric)
    assert "109" in metric_values  # final NAV from the artifact
    captions = " ".join(c.body for c in app.caption)
    assert "base" in captions  # cost scenario surfaced


def test_sensitivity_page_shows_not_a_leaderboard(empty_lake) -> None:
    report = {
        "note": ("Sensitivity grid. This table is NOT a leaderboard."),
        "model_name": "hist_gbt_reg", "model_version": "v0.1",
        "signal_name": "predicted_return_3d", "trained_horizon": 3,
        "n_grid_points": 1, "base_reference": None,
        "reference_check_vs_primary": None,
        "skipped_axes": [{"axis": "horizon", "value": 10, "reason": "none"}],
    }
    (empty_lake / "reports" / "sensitivity_report.json").write_text(
        json.dumps(report), encoding="utf-8")
    pd.DataFrame([{
        "horizon": 3, "n_holdings": 20, "frequency": "weekly",
        "max_stock_weight": 0.10, "cost_scenario": "base",
        "sharpe_ratio": 0.289,
    }]).to_parquet(
        empty_lake / "reports" / "sensitivity_results.parquet", index=False)

    app = _run(DASHBOARD / "pages/3_Sensitivity.py")
    assert not app.exception
    error_text = "\n".join(e.body for e in app.error)
    assert "NOT a leaderboard" in error_text
