"""Portfolio construction: OOS signals -> long-only target portfolios (Phase 6).

Modules:
    schedule   — rebalance calendar on ASX sessions (weekday shift forward)
    selection  — eligibility filters (liquidity, history, look-ahead guard)
                 and signal ranking
    weights    — equal/vol-adjusted weighting with the max-stock cap and a
                 cash residual (prefer_cash_over_violation)
    checks     — portfolio quality rules (long-only, caps, gross exposure,
                 no look-ahead, documented sector-rule gap)
    build      — orchestrator: config -> gold/target_portfolios.parquet
"""
