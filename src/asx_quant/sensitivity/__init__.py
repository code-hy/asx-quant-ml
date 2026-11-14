"""Sensitivity & robustness analysis (Phase 8).

Modules:
    grid   — expand the configured sensitivity axes into runnable grid
             points; horizons/axes that cannot run are recorded as
             explicitly skipped (with reasons), never fabricated
    runner — in-memory re-runs of portfolio construction + backtest over
             the grid; base grid point must reproduce the primary Phase 7
             backtest; outputs sensitivity_results.parquet +
             sensitivity_report.json (research diagnostics, not a
             selection leaderboard)
"""
