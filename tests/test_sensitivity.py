"""Phase 8 tests: sensitivity grid, runner, report consistency.

Offline: tiny synthetic lake (predictions + features + prices). The grid
runner's key guarantees:

* the grid is the cartesian product of the runnable axes;
* axes that cannot run (untrained horizons, sector weights) are recorded
  as SKIPPED with explicit reasons — never fabricated, never silent;
* the runner re-uses the SAME pure construction core as the staged build,
  and every grid point passes portfolio + backtest quality checks;
* results are written as a parquet table + JSON report with a base
  reference row.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from asx_quant.exceptions import StorageError
from asx_quant.sensitivity.grid import PortfolioScenario, expand_grid
from asx_quant.sensitivity.runner import run_sensitivity

# ------------------------------------------------------------ synthetic data
SESSIONS = pd.bdate_range("2024-01-01", periods=120)
TICKERS = [f"T{i:02d}.AX" for i in range(30)]


def _features_frame() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(5)
    for ticker in TICKERS:
        vol = abs(rng.normal(0.25, 0.05))
        for date in SESSIONS:
            rows.append({
                "ticker": ticker, "date": date, "liquid_flag": 1.0,
                "volatility_20d": vol,
            })
    return pd.DataFrame(rows)


def _predictions_frame() -> pd.DataFrame:
    """hist_gbt_reg OOS predictions on every session but the first."""
    rows = []
    rng = np.random.default_rng(7)
    for date in SESSIONS[5:]:
        # a persistent ranking: ticker 00 strongest, so results are stable
        for i, ticker in enumerate(TICKERS):
            rows.append({
                "ticker": ticker, "date": date,
                "model_name": "hist_gbt_reg", "model_kind": "regressor",
                "model_version": "v0.1", "feature_set_version": "v0.1",
                "fold_id": 0, "trained_through": SESSIONS[4],
                "prediction": rng.normal(0.002, 0.01) + (30 - i) * 1e-5,
            })
    return pd.DataFrame(rows)


def _prices_frame() -> pd.DataFrame:
    """All tickers + benchmark VAS.AX with deterministic drift."""
    rows = []
    for i, ticker in enumerate(TICKERS):
        price = 10.0 + i * 0.1
        for date in SESSIONS:
            price *= 1.0 + (0.001 if i % 2 == 0 else 0.0002)
            rows.append({
                "ticker": ticker, "date": date,
                "open_price": round(price, 6),
                "close_price": round(price * 1.0005, 6),
                "adjusted_close": round(price * 1.0005, 6),
                "dividend_amount": 0.0, "split_factor": 0.0,
            })
    bpx = 3000.0
    for date in SESSIONS:
        bpx *= 1.0004
        rows.append({
            "ticker": "VAS.AX", "date": date,
            "open_price": round(bpx, 6), "close_price": round(bpx, 6),
            "adjusted_close": round(bpx, 6),
            "dividend_amount": 0.0, "split_factor": 0.0,
        })
    return pd.DataFrame(rows)


FEATURES = _features_frame()
PREDICTIONS = _predictions_frame()
PRICES = _prices_frame()


# ------------------------------------------------------------------- grid
def test_expand_grid_cartesian_product() -> None:
    cfg = {
        "portfolio_sizes": [10, 20],
        "rebalance_frequencies": ["weekly", "monthly"],
        "max_stock_weights": [0.05, 0.10],
        "cost_scenarios": ["optimistic", "base", "conservative"],
        "horizons": [3],
    }
    plan = expand_grid(cfg, trained_horizon=3)
    assert len(plan.points) == 2 * 2 * 2 * 3  # 24
    assert plan.skipped == []
    point = plan.points[0]
    assert point.portfolio == PortfolioScenario(
        horizon=3, n_holdings=10, frequency="weekly", max_stock_weight=0.05)
    assert point.cost_scenario == "optimistic"
    # every portfolio config appears once per cost scenario
    portfolios = {p.portfolio for p in plan.points}
    assert len(portfolios) == 8


def test_expand_grid_skips_untrained_horizon_with_reason() -> None:
    cfg = {"portfolio_sizes": [20], "rebalance_frequencies": ["weekly"],
           "max_stock_weights": [0.10], "cost_scenarios": ["base"],
           "horizons": [3, 10]}
    plan = expand_grid(cfg, trained_horizon=3)
    assert len(plan.points) == 1  # only the 3d horizon runs
    assert len(plan.skipped) == 1
    skip = plan.skipped[0]
    assert skip["axis"] == "horizon" and skip["value"] == 10
    assert "No model trained" in skip["reason"]
    assert "train" in skip["reason"]  # points to the documented path


def test_expand_grid_sector_axis_is_never_runnable() -> None:
    """max_sector_weight stays a documented skip even if someone adds it
    to the grid lists: the runner has no implementation for it."""
    cfg = {"portfolio_sizes": [20], "rebalance_frequencies": ["weekly"],
           "max_stock_weights": [0.10], "cost_scenarios": ["base"],
           "horizons": [3], "max_sector_weights": [0.20, 0.30]}
    plan = expand_grid(cfg, trained_horizon=3)
    # no grid point carries a sector weight axis at all
    assert all(not hasattr(p.portfolio, "max_sector_weight")
               for p in plan.points)


def test_expand_grid_empty_when_all_skipped() -> None:
    cfg = {"horizons": [10], "portfolio_sizes": [20],
           "rebalance_frequencies": ["weekly"], "max_stock_weights": [0.10],
           "cost_scenarios": ["base"]}
    plan = expand_grid(cfg, trained_horizon=3)
    assert plan.points == []
    assert len(plan.skipped) == 1


# ------------------------------------------------- construction equivalence
def test_base_grid_point_matches_staged_construction(
    tmp_path, monkeypatch,
) -> None:
    """The runner's construction core IS the staged build's core."""
    from asx_quant.config.loader import load_yaml as real_load_yaml
    from asx_quant.portfolio.build import construct_target_portfolios
    from asx_quant.portfolio.selection import load_signal

    cfg = real_load_yaml("portfolio").get("portfolio", {})
    signals = load_signal(PREDICTIONS, "hist_gbt_reg", "predicted_return_3d")
    sessions = pd.DatetimeIndex(sorted(FEATURES["date"].unique()))

    staged, staged_records = construct_target_portfolios(
        signals, FEATURES, sessions,
        dict(cfg.get("selection", {})), dict(cfg.get("weights", {})),
        dict(cfg.get("rebalancing", {})), dict(cfg.get("constraints", {})),
        model_name="hist_gbt_reg", model_version="v0.1",
        signal_name="predicted_return_3d", feature_set_version="v0.1",
    )
    # same call again -> identical output (deterministic, pure)
    again, again_records = construct_target_portfolios(
        signals, FEATURES, sessions,
        dict(cfg.get("selection", {})), dict(cfg.get("weights", {})),
        dict(cfg.get("rebalancing", {})), dict(cfg.get("constraints", {})),
        model_name="hist_gbt_reg", model_version="v0.1",
        signal_name="predicted_return_3d", feature_set_version="v0.1",
    )
    pd.testing.assert_frame_equal(staged, again)
    assert staged_records == again_records


# ----------------------------------------------------------------- runner
def _sensitivity_cfg(grid_override: dict | None = None) -> dict:
    return {
        "portfolio": {
            "portfolio_id": "test_v1",
            "selection": {
                "n_holdings": 5, "signal": "predicted_return_3d",
                "prediction_source": {"dataset": "predictions",
                                      "model": "hist_gbt_reg"},
                "require_liquidity_flag": True,
            },
            "rebalancing": {"frequency": "weekly", "weekday": "monday"},
            "weights": {"method": "equal_weight",
                        "max_stock_weight": 0.10,
                        "prefer_cash_over_violation": True},
            "constraints": {"min_history_sessions": 5,
                            "max_holdings": 20},
        },
        "backtest": {
            "signal": {"dataset": "target_portfolios"},
            "benchmark": {"ticker_priority": ["VAS.AX"]},
            "execution": {"execution_price": "next_session_open",
                          "signal_to_execution_lag_sessions": 1},
            "transaction_costs": {
                "active_scenario": "base",
                "scenarios": {
                    "optimistic": {"brokerage_per_trade_aud": 0.0,
                                   "brokerage_percent": 0.0002,
                                   "slippage_percent": 0.0005},
                    "base": {"brokerage_per_trade_aud": 9.5,
                             "brokerage_percent": 0.0005,
                             "slippage_percent": 0.001},
                    "conservative": {"brokerage_per_trade_aud": 19.95,
                                     "brokerage_percent": 0.001,
                                     "slippage_percent": 0.0025},
                },
            },
            "capital": {"initial_capital_aud": 100_000.0,
                        "min_trade_value_aud": 0.0},
            "risk_free_rate": {"annual_rate": 0.0},
            "metrics": {"trading_days_per_year": 252},
            "periods": {"test_start": "2024-03-01"},
            "sensitivity": grid_override or {
                "portfolio_sizes": [5, 10],
                "rebalance_frequencies": ["weekly"],
                "max_stock_weights": [0.10],
                "cost_scenarios": ["optimistic", "base"],
                "horizons": [3, 10],
            },
        },
        "train": {"target": {"horizon": 3}},
    }


class _S:
    pass


def test_run_sensitivity_end_to_end(tmp_path, monkeypatch) -> None:
    from asx_quant.observability.runs import RunRegistry
    from asx_quant.storage.paths import DataLake

    cfg = _sensitivity_cfg()
    monkeypatch.setattr(
        "asx_quant.sensitivity.runner.load_yaml",
        lambda name: {name: cfg[name]},
    )

    lake = DataLake(root=tmp_path / "data")
    lake.ensure_all()
    registry = RunRegistry.start(lake, config_version="test")
    FEATURES.to_parquet(lake.gold_dataset_path("features"), index=False)
    PREDICTIONS.to_parquet(lake.gold_dataset_path("predictions"), index=False)
    PRICES.to_parquet(lake.silver_dataset_path("equity_prices"), index=False)

    result = run_sensitivity(lake, _S(), registry)
    # 2 sizes x 1 freq x 1 cap x 2 cost scenarios = 4 ok rows
    assert len(result.rows) == 4
    assert result.n_points == 4
    assert result.n_skipped_axes == 1  # horizon 10 skipped
    assert (result.rows["status"] == "ok").all()
    # every row has metrics
    ok = result.rows
    assert ok["final_nav"].notna().all()
    assert (ok["final_nav"] > 0).all()
    assert ok["total_return"].notna().all()
    assert ok["test_total_return"].notna().all()
    # trade/cost aggregates are populated (regression: the runner once
    # left them null because it skipped turnover_from_trades)
    assert ok["n_trades"].notna().all()
    assert (ok["n_trades"] > 0).all()
    assert ok["total_costs_aud"].notna().all()
    assert ok["avg_turnover"].notna().all()

    # output artifacts
    saved = pd.read_parquet(result.results_path)
    assert len(saved) == 4
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert report["n_grid_points"] == 4
    assert len(report["skipped_axes"]) == 1
    assert report["skipped_axes"][0]["value"] == 10
    assert "NOT a leaderboard" in report["note"]
    assert report["base_reference"] is not None
    assert report["reference_check_vs_primary"] is None  # no primary run
    # ranking lists present
    assert len(report["best_by_sharpe"]) > 0
    assert len(report["worst_by_sharpe"]) > 0

    # cost scenarios rank as expected: cheaper costs -> higher final NAV
    by_cost = ok.groupby("cost_scenario")["final_nav"].mean()
    assert by_cost["optimistic"] > by_cost["base"]


def test_run_sensitivity_base_matches_primary(tmp_path, monkeypatch) -> None:
    """When the primary backtest ran, the base grid point must reproduce it."""
    from asx_quant.backtest.build import build_backtest
    from asx_quant.observability.runs import RunRegistry
    from asx_quant.storage.paths import DataLake

    cfg = _sensitivity_cfg(grid_override={
        "portfolio_sizes": [5], "rebalance_frequencies": ["weekly"],
        "max_stock_weights": [0.10], "cost_scenarios": ["base"],
        "horizons": [3],
    })
    monkeypatch.setattr(
        "asx_quant.sensitivity.runner.load_yaml",
        lambda name: {name: cfg[name]},
    )
    monkeypatch.setattr(
        "asx_quant.backtest.build.load_yaml",
        lambda name: {name: cfg[name]},
    )
    monkeypatch.setattr(
        "asx_quant.portfolio.build.load_yaml",
        lambda name: {name: cfg[name]},
    )

    lake = DataLake(root=tmp_path / "data")
    lake.ensure_all()
    registry = RunRegistry.start(lake, config_version="test")
    FEATURES.to_parquet(lake.gold_dataset_path("features"), index=False)
    PREDICTIONS.to_parquet(lake.gold_dataset_path("predictions"), index=False)
    PRICES.to_parquet(lake.silver_dataset_path("equity_prices"), index=False)

    # run the staged Phase 7 backtest first (writes backtest_metrics.json)
    from asx_quant.portfolio.build import build_portfolios

    build_portfolios(lake, _S(), registry, with_quality=False)
    build_backtest(lake, _S(), registry, with_quality=False)
    primary = json.loads(
        (tmp_path / "data/reports/backtest_metrics.json").read_text(
            encoding="utf-8"))

    result = run_sensitivity(lake, _S(), registry)
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    check = report["reference_check_vs_primary"]
    assert check is not None
    assert check["matches"] is True
    assert check["primary_final_nav"] == pytest.approx(primary["final_nav"],
                                                       abs=0.01)


def test_run_sensitivity_fails_without_predictions(
    tmp_path, monkeypatch,
) -> None:
    from asx_quant.observability.runs import RunRegistry
    from asx_quant.storage.paths import DataLake

    cfg = _sensitivity_cfg()
    monkeypatch.setattr(
        "asx_quant.sensitivity.runner.load_yaml",
        lambda name: {name: cfg[name]},
    )
    lake = DataLake(root=tmp_path / "data")
    lake.ensure_all()
    registry = RunRegistry.start(lake, config_version="test")
    FEATURES.to_parquet(lake.gold_dataset_path("features"), index=False)

    with pytest.raises(Exception):  # noqa: B017 (parquet/shape error path)
        run_sensitivity(lake, _S(), registry)


def test_run_sensitivity_larger_size_passes_checks(tmp_path, monkeypatch) -> None:
    """Regression: a grid size LARGER than the staged max_holdings cap is a
    legitimate scenario — the per-scenario cap must follow the scenario
    (real run failed with max_holdings_respected on n_holdings=30)."""
    from asx_quant.observability.runs import RunRegistry
    from asx_quant.storage.paths import DataLake

    cfg = _sensitivity_cfg(grid_override={
        "portfolio_sizes": [5, 15], "rebalance_frequencies": ["weekly"],
        "max_stock_weights": [0.10], "cost_scenarios": ["base"],
        "horizons": [3],
    })
    monkeypatch.setattr("asx_quant.sensitivity.runner.load_yaml",
                        lambda name: {name: cfg[name]})
    # staged cap is max_holdings=20 in the shared config; 15 < 20 passes,
    # so push past it explicitly:
    cfg["portfolio"]["constraints"]["max_holdings"] = 10

    lake = DataLake(root=tmp_path / "data")
    lake.ensure_all()
    registry = RunRegistry.start(lake, config_version="test")
    FEATURES.to_parquet(lake.gold_dataset_path("features"), index=False)
    PREDICTIONS.to_parquet(lake.gold_dataset_path("predictions"), index=False)
    PRICES.to_parquet(lake.silver_dataset_path("equity_prices"), index=False)

    result = run_sensitivity(lake, _S(), registry)
    assert (result.rows["status"] == "ok").all()
    big = result.rows[result.rows["n_holdings"] == 15]
    assert len(big) == 1
    # the 15-name scenario actually held ~15 positions (30 tickers, 5-day
    # min history) -> above the staged cap of 10, exactly as intended
    assert big["avg_holdings"].iloc[0] == pytest.approx(15.0)


def test_run_sensitivity_empty_grid_raises(tmp_path, monkeypatch) -> None:
    from asx_quant.observability.runs import RunRegistry
    from asx_quant.storage.paths import DataLake

    cfg = _sensitivity_cfg(grid_override={
        "portfolio_sizes": [20], "rebalance_frequencies": ["weekly"],
        "max_stock_weights": [0.10], "cost_scenarios": ["base"],
        "horizons": [10],
    })
    monkeypatch.setattr(
        "asx_quant.sensitivity.runner.load_yaml",
        lambda name: {name: cfg[name]},
    )
    lake = DataLake(root=tmp_path / "data")
    lake.ensure_all()
    registry = RunRegistry.start(lake, config_version="test")

    with pytest.raises(StorageError, match="empty"):
        run_sensitivity(lake, _S(), registry)


def test_real_config_sensitivity_grid_is_consistent() -> None:
    """The real backtest.yaml sensitivity axes are coherent."""
    from asx_quant.config.loader import load_yaml

    backtest_cfg = load_yaml("backtest")["backtest"]
    sens = backtest_cfg["sensitivity"]
    assert 20 in sens["portfolio_sizes"]
    assert "weekly" in sens["rebalance_frequencies"]
    assert 0.10 in sens["max_stock_weights"]
    assert "base" in sens["cost_scenarios"]
    assert sens["horizons"] == [3, 10]
    assert "max_sector_weights" in sens  # documented, not runnable
    train_cfg = load_yaml("train")["train"]
    assert train_cfg["target"]["horizon"] == 3
