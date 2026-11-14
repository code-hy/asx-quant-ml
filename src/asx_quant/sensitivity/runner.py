"""Sensitivity runner (Phase 8): re-run construction + backtest across a grid.

Design:

* Inputs (OOS predictions, gold features, silver prices) are loaded ONCE;
  every grid point re-uses them in memory. The price pivots for the
  backtest engine are prepared once as well.
* Portfolio grid points call :func:`construct_target_portfolios` — the
  SAME pure core the staged Phase 6 build uses, so the base grid point
  reproduces the primary run exactly (verified by test).
* Each portfolio configuration runs quality checks in memory; each
  backtest runs quality checks in memory; any error-severity failure
  aborts the whole run (a grid must not contain invalid scenarios).
* The base scenario from the staged configs is always included as a
  reference row even if it is not part of the grid.
* NOTHING here informs model or configuration selection: the held-out
  window (periods.test_start) metrics are reported per row as diagnostics
  only. The output is a research table, not a leaderboard to cherry-pick —
  the report text says so explicitly.

Outputs:
* ``data/reports/sensitivity_results.parquet`` — one row per grid point.
* ``data/reports/sensitivity_report.json`` — metadata, skipped axes with
  reasons, and ranked summaries (best/worst by Sharpe).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from asx_quant.backtest.checks import run_backtest_quality_checks
from asx_quant.backtest.engine import (
    benchmark_curve,
    prepare_price_pivots,
    run_backtest,
)
from asx_quant.backtest.metrics import compute_metrics, turnover_from_trades
from asx_quant.config.loader import load_yaml
from asx_quant.exceptions import StorageError
from asx_quant.observability.logging import get_logger
from asx_quant.portfolio import checks as portfolio_checks
from asx_quant.portfolio.build import construct_target_portfolios
from asx_quant.portfolio.selection import load_signal
from asx_quant.sensitivity.grid import expand_grid
from asx_quant.storage.paths import DataLake

logger = get_logger("sensitivity.run")

RESULT_COLUMNS = (
    "horizon", "n_holdings", "frequency", "max_stock_weight",
    "cost_scenario", "status", "skip_reason", "n_rebalances",
    "n_weight_rows", "avg_holdings", "avg_target_turnover", "final_nav",
    "total_return", "cagr", "annual_volatility", "sharpe_ratio",
    "sortino_ratio", "max_drawdown", "benchmark_total_return",
    "excess_return", "information_ratio", "beta", "alpha_annualised",
    "avg_turnover", "total_costs_aud", "n_trades",
    "test_total_return", "test_sharpe", "test_max_drawdown",
)


@dataclass
class SensitivityResult:
    """Paths and row count of one sensitivity run."""

    results_path: Path
    report_path: Path
    rows: pd.DataFrame
    n_points: int
    n_skipped_axes: int


def _row_from_metrics(
    point, metrics: dict, records: list[dict], n_weight_rows: int,
    status: str = "ok", skip_reason: str = "",
) -> dict:
    """Flatten one backtest result into a results-table row."""
    test = metrics.get("test_window", {})
    avg_target_turnover = (
        float(np.mean([r["turnover"] for r in records]))
        if records else float("nan"))
    turnover = metrics.get("turnover", {})
    return {
        "horizon": point.portfolio.horizon,
        "n_holdings": point.portfolio.n_holdings,
        "frequency": point.portfolio.frequency,
        "max_stock_weight": point.portfolio.max_stock_weight,
        "cost_scenario": point.cost_scenario,
        "status": status,
        "skip_reason": skip_reason,
        "n_rebalances": len(records),
        "n_weight_rows": int(n_weight_rows),
        "avg_holdings": round(
            float(np.mean([r["n_holdings"] for r in records])), 2)
        if records else float("nan"),
        "avg_target_turnover": round(avg_target_turnover, 4),
        "final_nav": metrics.get("final_nav"),
        "total_return": metrics.get("total_return"),
        "cagr": metrics.get("cagr"),
        "annual_volatility": metrics.get("annual_volatility"),
        "sharpe_ratio": metrics.get("sharpe_ratio"),
        "sortino_ratio": metrics.get("sortino_ratio"),
        "max_drawdown": metrics.get("max_drawdown"),
        "benchmark_total_return": metrics.get("benchmark_total_return"),
        "excess_return": metrics.get("excess_return"),
        "information_ratio": metrics.get("information_ratio"),
        "beta": metrics.get("beta"),
        "alpha_annualised": metrics.get("alpha_annualised"),
        "avg_turnover": turnover.get("avg_turnover_per_rebalance"),
        "total_costs_aud": turnover.get("total_costs_aud"),
        "n_trades": turnover.get("n_trades"),
        "test_total_return": test.get("total_return"),
        "test_sharpe": test.get("sharpe_ratio"),
        "test_max_drawdown": test.get("max_drawdown"),
    }


def run_sensitivity(
    lake: DataLake,
    settings,  # noqa: ANN001
    registry,  # noqa: ANN001
) -> SensitivityResult:
    """Run the configured sensitivity grid end-to-end in memory."""
    portfolio_cfg = load_yaml("portfolio").get("portfolio", {})
    backtest_cfg = load_yaml("backtest").get("backtest", {})
    train_cfg = load_yaml("train").get("train", {})
    sensitivity_cfg = dict(backtest_cfg.get("sensitivity", {}))

    trained_horizon = int(train_cfg.get("target", {}).get("horizon", 3))
    plan = expand_grid(sensitivity_cfg, trained_horizon)
    if not plan.points:
        raise StorageError(
            "Sensitivity grid is empty: no runnable (horizon, size, "
            "frequency, cap) combination for the trained model."
        )
    logger.info("Sensitivity grid: %d points (%d skipped axes)",
                len(plan.points), len(plan.skipped))

    # ---- shared inputs (loaded once) --------------------------------------
    from asx_quant.storage.parquet import read_parquet

    selection = portfolio_cfg.get("selection", {})
    source = selection.get("prediction_source", {})
    signal_name = str(selection.get("signal", "predicted_return_3d"))
    model_name = str(source.get("model", "hist_gbt_reg"))
    predictions = read_parquet(
        lake.gold_dataset_path(str(source.get("dataset", "predictions"))))
    signals = load_signal(predictions, model_name, signal_name)
    if signals.empty:
        raise StorageError(
            f"No predictions for model '{model_name}'; run the 'train' stage."
        )
    feature_set_version = str(predictions["feature_set_version"].iloc[0])
    model_version = str(predictions["model_version"].iloc[0])

    features = read_parquet(lake.gold_dataset_path("features"))
    sessions = pd.DatetimeIndex(sorted(features["date"].unique()))
    prices = read_parquet(lake.silver_dataset_path("equity_prices"))
    pivots = prepare_price_pivots(prices)

    execution_cfg = dict(backtest_cfg.get("execution", {}))
    capital_cfg = dict(backtest_cfg.get("capital", {}))
    cost_scenarios = {
        name: dict(spec)
        for name, spec in backtest_cfg.get("transaction_costs", {})
        .get("scenarios", {}).items()
    }
    metrics_cfg = dict(backtest_cfg.get("metrics", {}))
    rf_annual = float(backtest_cfg.get("risk_free_rate", {})
                      .get("annual_rate", 0.0425))
    benchmark_priority = list(
        backtest_cfg.get("benchmark", {}).get(
            "ticker_priority", ["VAS.AX", "IDX_AXJO"]))
    initial_capital = float(capital_cfg.get("initial_capital_aud", 100_000.0))

    constraints_cfg = dict(portfolio_cfg.get("constraints", {}))
    max_holdings = int(constraints_cfg.get("max_holdings",
                                           selection.get("n_holdings", 20)))
    require_liquidity = bool(selection.get("require_liquidity_flag", True))
    min_history = int(constraints_cfg.get("min_history_sessions", 250))
    weights_cfg_base = dict(portfolio_cfg.get("weights", {}))
    rebal_cfg_base = dict(portfolio_cfg.get("rebalancing", {}))

    rows: list[dict] = []

    for point in plan.points:
        scenario = point.portfolio
        selection_cfg = dict(selection)
        selection_cfg["n_holdings"] = scenario.n_holdings
        weights_cfg = dict(weights_cfg_base)
        weights_cfg["max_stock_weight"] = scenario.max_stock_weight
        rebal_cfg = dict(rebal_cfg_base)
        rebal_cfg["frequency"] = scenario.frequency

        portfolios, records = construct_target_portfolios(
            signals, features, sessions,
            selection_cfg, weights_cfg, rebal_cfg, constraints_cfg,
            model_name=model_name, model_version=model_version,
            signal_name=signal_name, feature_set_version=feature_set_version,
        )
        # The scenario's own n_holdings IS the intended holding cap: grid
        # points with larger sizes (e.g. 30) are legitimate configurations,
        # not violations of the staged run's platform cap.
        scenario_max_holdings = max(max_holdings, scenario.n_holdings)
        portfolio_checks.run_portfolio_quality_checks(
            portfolios, records,
            max_stock_weight=scenario.max_stock_weight,
            max_holdings=scenario_max_holdings,
            min_history_sessions=min_history,
            require_liquidity_flag=require_liquidity,
            sector_rule_implemented=False,
        ).assert_passed()

        costs = cost_scenarios[point.cost_scenario]
        result = run_backtest(
            portfolios, prices,
            execution_cfg=execution_cfg, costs=costs,
            capital_cfg=capital_cfg, pivots=pivots,
        )
        run_backtest_quality_checks(
            result.equity_curve, result.trades, result.diagnostics,
            max_holdings=scenario_max_holdings,
        ).assert_passed()

        capital = initial_capital
        bench = benchmark_curve(
            prices, pd.DatetimeIndex(result.equity_curve["date"]),
            initial_capital=capital, ticker_priority=benchmark_priority)
        curve = result.equity_curve.copy()
        curve["benchmark_nav"] = bench.to_numpy()
        metrics = compute_metrics(
            curve,
            trading_days=int(metrics_cfg.get("trading_days_per_year", 252)),
            risk_free_annual=rf_annual,
        )
        # test-window block (diagnostics only)
        test_start = backtest_cfg.get("periods", {}).get("test_start")
        if test_start:
            test_curve = curve[curve["date"] >= pd.Timestamp(test_start)]
            if len(test_curve) > 2:
                tw = compute_metrics(
                    test_curve.reset_index(drop=True),
                    trading_days=int(
                        metrics_cfg.get("trading_days_per_year", 252)),
                    risk_free_annual=rf_annual)
                metrics["test_window"] = tw
        metrics["turnover"] = turnover_from_trades(result.trades)

        row = _row_from_metrics(point, metrics, records, len(portfolios))
        rows.append(row)

        logger.info(
            "Grid point done: h%d n%d %s cap%.2f %s -> Sharpe %s, "
            "return %s", scenario.horizon, scenario.n_holdings,
            scenario.frequency, scenario.max_stock_weight,
            point.cost_scenario, metrics.get("sharpe_ratio"),
            metrics.get("total_return"))

    results = pd.DataFrame(rows, columns=list(RESULT_COLUMNS))

    # ---- reference consistency: the base grid point must reproduce the
    # primary staged backtest (Phase 7) when one has already been run.
    reference_check: dict | None = None
    primary_path = lake.report_path("backtest_metrics", ".json")
    base = _base_row(results, portfolio_cfg, backtest_cfg, trained_horizon)
    if Path(primary_path).exists() and base is not None:
        primary = json.loads(
            Path(primary_path).read_text(encoding="utf-8"))
        primary_nav = float(primary.get("final_nav", float("nan")))
        base_nav = float(base.get("final_nav", float("nan")))
        reference_check = {
            "primary_final_nav": round(primary_nav, 2),
            "base_grid_final_nav": round(base_nav, 2),
            "matches": bool(abs(primary_nav - base_nav) <= 0.01),
        }
        if not reference_check["matches"]:
            raise StorageError(
                "Sensitivity base grid point does not reproduce the "
                f"primary backtest (base NAV {base_nav:.2f} vs primary "
                f"{primary_nav:.2f}); the construction core diverged."
            )

    # ---- report ------------------------------------------------------------
    report = {
        "note": (
            "Sensitivity grid over portfolio construction and cost "
            "assumptions. Research diagnostics ONLY: the held-out window "
            "(backtest.yaml periods.test_start) metrics are reported per "
            "row but every selection-relevant decision predates this "
            "stage. This table is NOT a leaderboard to cherry-pick the "
            "best scenario from."),
        "trained_horizon": trained_horizon,
        "model_name": model_name,
        "model_version": model_version,
        "signal_name": signal_name,
        "feature_set_version": feature_set_version,
        "n_grid_points": len(plan.points),
        "n_rows": len(results),
        "skipped_axes": plan.skipped,
        "best_by_sharpe": _top(results, "sharpe_ratio", 5),
        "worst_by_sharpe": _top(results, "sharpe_ratio", 5, ascending=True),
        "best_by_total_return": _top(results, "total_return", 5),
        "base_reference": base,
        "reference_check_vs_primary": reference_check,
    }
    report_path = lake.report_path("sensitivity_report", ".json")
    results_path = lake.report_path("sensitivity_results", ".parquet")
    results.to_parquet(results_path, index=False)
    report_path.write_text(json.dumps(report, indent=2, default=str),
                           encoding="utf-8")

    logger.info("Sensitivity run complete: %d rows -> %s", len(results),
                results_path)
    return SensitivityResult(
        results_path=results_path,
        report_path=report_path,
        rows=results,
        n_points=len(plan.points),
        n_skipped_axes=len(plan.skipped),
    )


def _top(results: pd.DataFrame, column: str, n: int,
         ascending: bool = False) -> list[dict]:
    """Top/bottom n rows by one metric as JSON-safe dicts."""
    if results.empty or column not in results:
        return []
    ranked = results.sort_values(column, ascending=ascending,
                                 na_position="last").head(n)
    return [
        {k: (v if not isinstance(v, float) or np.isfinite(v) else None)
         for k, v in row.items()}
        for row in ranked.to_dict(orient="records")
    ]


def _base_row(results: pd.DataFrame, portfolio_cfg: dict,
              backtest_cfg: dict, trained_horizon: int) -> dict | None:
    """The staged-config grid point, for direct comparison."""
    mask = (
        (results["n_holdings"] == int(portfolio_cfg.get("selection", {})
                                     .get("n_holdings", 20)))
        & (results["frequency"] == str(portfolio_cfg.get("rebalancing", {})
                                       .get("frequency", "weekly")))
        & (results["max_stock_weight"] == float(
            portfolio_cfg.get("weights", {}).get("max_stock_weight", 0.10)))
        & (results["cost_scenario"] == str(
            backtest_cfg.get("transaction_costs", {})
            .get("active_scenario", "base")))
        & (results["horizon"] == trained_horizon)
    )
    matches = results[mask]
    if matches.empty:
        return None
    return matches.iloc[0].to_dict()
