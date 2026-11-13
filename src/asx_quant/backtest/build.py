"""Backtest orchestrator: target portfolios -> equity curve + metrics.

Pipeline:

1. Load ``gold/target_portfolios.parquet`` (Phase 6) and silver equity
   prices (raw open/close, dividends, splits — the backtester never uses
   the adjusted series for execution, only for the benchmark).
2. Pick the active cost scenario from ``backtest.yaml``.
3. Simulate with :func:`asx_quant.backtest.engine.run_backtest`
   (next-session execution, whole shares, costs, dividends, splits).
4. Compute the frictionless benchmark NAV (VAS.AX total-return proxy,
   IDX_AXJO fallback) and performance metrics.
5. Run quality rules; write ``gold/backtest_equity_curve.parquet``,
   ``gold/backtest_trades.parquet``, ``data/reports/backtest_metrics.json``
   and ``data/reports/backtest_quality.json``.

The held-out window (``periods.test_start`` = 2024-07-01) is NOT used for
model selection — backtest metrics over that window are reported separately
in the metrics JSON (``test_window`` block) as a Phase 8 input, and every
selection-relevant decision predates this stage (Phase 5/6 guard).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from asx_quant.backtest.checks import run_backtest_quality_checks
from asx_quant.backtest.engine import benchmark_curve, run_backtest
from asx_quant.backtest.metrics import compute_metrics, turnover_from_trades
from asx_quant.config.loader import load_yaml
from asx_quant.exceptions import StorageError
from asx_quant.observability.logging import get_logger
from asx_quant.quality.checks import QualityReport
from asx_quant.storage.paths import DataLake

logger = get_logger("backtest.build")


@dataclass
class BacktestBuildResult:
    """Paths and headline metrics of one backtest run."""

    curve_path: Path
    trades_path: Path
    metrics_path: Path
    report_path: Path
    report: QualityReport | None
    final_nav: float
    total_return: float
    sharpe: float | None
    max_drawdown: float
    n_trades: int


def build_backtest(
    lake: DataLake,
    settings,  # noqa: ANN001 (Settings; cycle-safe import)
    registry,  # noqa: ANN001 (RunRegistry)
    with_quality: bool = True,
) -> BacktestBuildResult:
    """Run the configured backtest over the constructed portfolios."""
    cfg = load_yaml("backtest").get("backtest", {})
    execution_cfg = dict(cfg.get("execution", {}))
    capital_cfg = dict(cfg.get("capital", {}))
    costs_cfg = dict(cfg.get("transaction_costs", {}))

    scenario_name = str(costs_cfg.get("active_scenario", "base"))
    scenarios = costs_cfg.get("scenarios", {})
    if scenario_name not in scenarios:
        raise ValueError(
            f"Unknown cost scenario '{scenario_name}'. Valid: "
            f"{', '.join(sorted(scenarios))}."
        )
    costs = dict(scenarios[scenario_name])

    # ---- inputs ------------------------------------------------------------
    dataset = str(cfg.get("signal", {}).get("dataset", "target_portfolios"))
    portfolios_path = lake.gold_dataset_path(dataset)
    if not Path(portfolios_path).exists():
        raise StorageError(
            f"gold/{dataset}.parquet not found; run the 'portfolio' stage "
            "first."
        )
    portfolios = pd.read_parquet(portfolios_path)
    prices = pd.read_parquet(lake.silver_dataset_path("equity_prices"))

    # ---- simulate ----------------------------------------------------------
    result = run_backtest(
        portfolios, prices,
        execution_cfg=execution_cfg, costs=costs, capital_cfg=capital_cfg,
    )
    curve = result.equity_curve
    trades = result.trades
    if curve.empty:
        raise StorageError(
            "Backtest produced an empty equity curve; check that the price "
            "history covers the portfolio period."
        )

    capital = float(capital_cfg.get("initial_capital_aud", 100_000.0))
    bench = benchmark_curve(
        prices,
        pd.DatetimeIndex(curve["date"]),
        initial_capital=capital,
        ticker_priority=list(
            cfg.get("benchmark", {}).get(
                "ticker_priority", ["VAS.AX", "IDX_AXJO"])),
    )
    curve = curve.copy()
    curve["benchmark_nav"] = bench.to_numpy()

    rf_cfg = cfg.get("risk_free_rate", {})
    metrics = compute_metrics(
        curve,
        trading_days=int(cfg.get("metrics", {}).get("trading_days_per_year",
                                                    252)),
        risk_free_annual=float(rf_cfg.get("annual_rate", 0.0425)),
    )
    metrics["cost_scenario"] = scenario_name
    metrics["costs"] = costs
    metrics["execution"] = {
        "execution_price": execution_cfg.get("execution_price",
                                             "next_session_open"),
        "lag_sessions": int(execution_cfg.get(
            "signal_to_execution_lag_sessions", 1)),
    }
    metrics["turnover"] = turnover_from_trades(trades)
    metrics["diagnostics"] = result.diagnostics
    metrics["benchmark_priority"] = list(
        cfg.get("benchmark", {}).get("ticker_priority", []))

    # Held-out window metrics: diagnostics only, never used for selection.
    test_start = str(cfg.get("periods", {}).get("test_start", ""))
    if test_start:
        test_curve = curve[curve["date"] >= pd.Timestamp(test_start)]
        if len(test_curve) > 2:
            test_metrics = compute_metrics(
                test_curve.reset_index(drop=True),
                trading_days=int(cfg.get("metrics", {})
                                 .get("trading_days_per_year", 252)),
                risk_free_annual=float(rf_cfg.get("annual_rate", 0.0425)),
            )
            metrics["test_window"] = {
                "start": test_start,
                "note": "held-out diagnostics; never used for selection",
                **{k: v for k, v in test_metrics.items()
                   if k in ("initial_nav", "final_nav", "total_return",
                            "cagr", "annual_volatility", "sharpe_ratio",
                            "sortino_ratio", "max_drawdown", "n_sessions",
                            "benchmark_total_return", "excess_return",
                            "information_ratio", "beta",
                            "alpha_annualised")},
            }

    # ---- quality + persist -------------------------------------------------
    portfolio_cfg = load_yaml("portfolio").get("portfolio", {})
    max_holdings = int(
        portfolio_cfg.get("constraints", {}).get(
            "max_holdings",
            portfolio_cfg.get("selection", {}).get("n_holdings", 20)))

    report = None
    report_path = lake.report_path("backtest_quality", ".json")
    if with_quality:
        report = run_backtest_quality_checks(
            curve, trades, result.diagnostics, max_holdings=max_holdings)
        report_path = report.save(report_path)
        report.assert_passed()
        logger.info("Backtest quality: %s (%d warnings)",
                    report.to_dict()["status"], report.warn_count())

    curve_path = lake.gold_dataset_path("backtest_equity_curve")
    curve.to_parquet(curve_path, index=False)
    trades_path = lake.gold_dataset_path("backtest_trades")
    trades.to_parquet(trades_path, index=False)

    metrics_path = lake.report_path("backtest_metrics", ".json")
    metrics_path.write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    return BacktestBuildResult(
        curve_path=curve_path,
        trades_path=trades_path,
        metrics_path=metrics_path,
        report_path=report_path,
        report=report,
        final_nav=float(curve["nav"].iloc[-1]),
        total_return=float(metrics.get("total_return", float("nan"))),
        sharpe=(float(metrics["sharpe_ratio"])
                if metrics.get("sharpe_ratio") is not None else None),
        max_drawdown=float(metrics.get("max_drawdown", float("nan"))),
        n_trades=int(len(trades)),
    )
