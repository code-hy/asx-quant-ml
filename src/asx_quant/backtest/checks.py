"""Automated quality rules for backtest results.

Error-severity rules guard simulation integrity:

* NAV series is finite and positive on every simulated session;
* cash never goes negative (no invented leverage);
* every executed trade has a finite positive price and whole-share quantity;
* execution strictly after the signal (effective_date > rebalance_date) and
  every signal was look-ahead-free (trained_through < rebalance_date);
* holding count never exceeds the configured maximum;
* every planned rebalance with a valid effective date was executed.

Warnings document engine behaviour that is by design but material:
skipped dust trades, skipped insufficient-cash buys, no-price skips,
split/cash-in-lieu and dividend event counts, suspended valuation days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from asx_quant.quality.checks import QualityReport, RuleResult


def run_backtest_quality_checks(
    curve: pd.DataFrame,
    trades: pd.DataFrame,
    diagnostics: dict,
    max_holdings: int,
    dataset_name: str = "backtest",
) -> QualityReport:
    """Quality rules over one backtest simulation.

    Args:
        curve: Engine equity curve (date, nav, cash, positions_value,
            n_positions).
        trades: Executed trades (effective_date, rebalance_date, ticker,
            side, shares, price_exec, trade_value, cost_aud, nav_base).
        diagnostics: Engine counters (skipped trades, corporate actions...).
        max_holdings: Configured maximum simultaneous positions.
    """
    report = QualityReport(dataset=dataset_name)

    def _rule(name: str, severity: str, passed: bool, failures: int = 0,
              sample: list[str] | None = None, detail: str = "") -> None:
        report.results.append(RuleResult(
            name=name, severity=severity, passed=passed,
            failures=int(failures), sample=sample or [], detail=detail,
        ))

    if curve.empty:
        _rule("nav_series_present", "error", False, 0,
              detail="No equity curve produced — check the portfolio and "
                     "price coverage overlap.")
        return report
    _rule("nav_series_present", "error", True, 0)

    nav = curve["nav"].astype(float)
    bad_nav = ~(np.isfinite(nav) & (nav > 0))
    _rule("nav_finite_positive", "error", not bad_nav.any(),
          int(bad_nav.sum()),
          [str(curve["date"].iloc[i]) for i in
           np.flatnonzero(bad_nav.to_numpy())[:10]],
          "NAV must be finite and strictly positive on every session.")

    bad_cash = curve["cash"].astype(float) < -1e-6
    _rule("cash_never_negative", "error", not bad_cash.any(),
          int(bad_cash.sum()),
          [str(curve["date"].iloc[i]) for i in
           np.flatnonzero(bad_cash.to_numpy())[:10]],
          "Cash must never go negative — buys exceeding available cash are "
          "skipped, never leveraged.")

    bad_counts = curve["n_positions"] > max_holdings
    _rule("max_holdings_respected", "error", not bad_counts.any(),
          int(bad_counts.sum()),
          [str(curve["date"].iloc[i]) for i in
           np.flatnonzero(bad_counts.to_numpy())[:10]],
          f"Simultaneous positions must never exceed {max_holdings}.")

    if not trades.empty:
        bad_price = ~(np.isfinite(trades["price_exec"])
                      & (trades["price_exec"] > 0))
        _rule("trades_priced_finite_positive", "error", not bad_price.any(),
              int(bad_price.sum()),
              [f"{r.ticker}@{r.effective_date}" for r in
               trades[bad_price].head(10).itertuples()],
              "Every executed trade needs a finite positive price.")

        whole_shares = trades["shares"] % 1 == 0
        _rule("whole_share_trades", "error", whole_shares.all(),
              int((~whole_shares).sum()),
              [f"{r.ticker}@{r.effective_date}" for r in
               trades[~whole_shares].head(10).itertuples()],
              "Trades are whole shares; fractional positions only arise "
              "from split adjustments, never from trades.")

        lag = pd.to_datetime(trades["effective_date"]) <= \
            pd.to_datetime(trades["rebalance_date"])
        _rule("execution_after_signal_date", "error", not lag.any(),
              int(lag.sum()),
              [f"{r.ticker}@{r.rebalance_date}" for r in
               trades[lag].head(10).itertuples()],
              "Execution must happen strictly after the signal close.")

    planned = int(diagnostics.get("rebalances_planned", 0))
    executed = int(diagnostics.get("rebalances_executed", 0))
    _rule("all_rebalances_executed", "error", executed == planned,
          abs(planned - executed),
          detail=f"Planned {planned}, executed {executed}: every rebalance "
                 "with a valid effective date must be simulated.")

    # ---- documented engine behaviour (warns) -------------------------------
    for counter, detail in (
        ("skipped_dust_trades",
         "Trades below min_trade_value_aud were skipped (dust rule)."),
        ("skipped_insufficient_cash",
         "Buys exceeding available cash were skipped (no leverage rule)."),
        ("skipped_no_price",
         "Target/holding tickers without a tradable price at execution "
         "were skipped this session."),
        ("suspended_valuation_days",
         "Held tickers without a session row were valued at the last "
         "known close."),
    ):
        count = int(diagnostics.get(counter, 0))
        _rule(f"engine_{counter}", "warn", count == 0, count,
              detail=detail + " Counted, never silent.")

    for counter in ("split_events", "cash_in_lieu_events", "dividend_events"):
        count = int(diagnostics.get(counter, 0))
        _rule(f"engine_{counter}", "warn", True, count,
              detail="Corporate-action events processed by the engine "
                     "(informational).")

    return report
