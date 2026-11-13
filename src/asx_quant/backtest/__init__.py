"""Backtesting: target portfolios -> simulated NAV, trades, metrics (Phase 7).

Modules:
    engine   — session-driven simulation: next-session execution, whole
               shares, cost scenario, dividends, splits, cash floor
    metrics  — performance metrics from the NAV curve (+ benchmark, held-out
               window block, turnover/cost aggregates)
    checks   — backtest quality rules (NAV, cash floor, whole shares, lag,
               rebalance coverage) + documented engine counters
    build    — orchestrator: config -> gold/backtest_equity_curve.parquet,
               gold/backtest_trades.parquet, reports
"""
