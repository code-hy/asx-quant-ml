"""Automated quality rules for constructed target portfolios.

Error-severity rules guard the platform invariants:

* long-only — no negative target weight;
* the max stock weight cap is respected (within float tolerance);
* holding count never exceeds ``max_holdings``;
* gross exposure (stocks + recorded cash) equals 1 within tolerance;
* every selected name was liquidity-eligible (when required) and had
  ``min_history_sessions`` of history;
* NO LOOK-AHEAD — every signal used at a rebalance came from a fold whose
  training ended strictly before the rebalance date;
* rebalance dates are actual ASX sessions.

Warnings document non-violations: rebalances with fewer names than
configured (cash kept instead of padding), and the deliberately skipped
sector-weight rule (no point-in-time sector mapping exists — Phase 4
limitation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from asx_quant.quality.checks import QualityReport, RuleResult

_TOLERANCE = 1e-6


def run_portfolio_quality_checks(
    portfolios: pd.DataFrame,
    rebalance_records: list[dict],
    max_stock_weight: float,
    max_holdings: int,
    min_history_sessions: int,
    require_liquidity_flag: bool,
    sector_rule_implemented: bool = False,
    dataset_name: str = "target_portfolios",
) -> QualityReport:
    """Quality rules over one portfolio-construction run.

    Args:
        portfolios: Long-format weight table (rebalance_date, ticker,
            target_weight, ...).
        rebalance_records: Per-rebalance metadata (cash_weight, turnover,
            n_holdings, min_history, all_liquid, session_flags).
        max_stock_weight: Configured cap.
        max_holdings: Configured cap.
        min_history_sessions: Configured minimum history.
        require_liquidity_flag: Whether liquidity was required.
        sector_rule_implemented: Always False in v0.1 — recorded as a warn
            with an explicit reason.
    """
    report = QualityReport(dataset=dataset_name)

    def _rule(name: str, severity: str, passed: bool, failures: int = 0,
              sample: list[str] | None = None, detail: str = "") -> None:
        report.results.append(RuleResult(
            name=name, severity=severity, passed=passed,
            failures=int(failures), sample=sample or [], detail=detail,
        ))

    if portfolios.empty:
        _rule("non_empty_portfolios", "error", False, 0,
              detail="No target portfolios were constructed — check the "
                     "signal availability window vs the rebalance schedule.")
        return report
    _rule("non_empty_portfolios", "error", True, 0)

    # ---- per-row invariants ----------------------------------------------
    negative = portfolios["target_weight"] < -_TOLERANCE
    _rule("long_only_weights", "error", not negative.any(),
          int(negative.sum()),
          [f"{r.ticker}@{r.rebalance_date.date()}"
           for r in portfolios[negative].head(10).itertuples()],
          "Long-only portfolio: negative target weights are invalid.")

    over_cap = portfolios["target_weight"] > max_stock_weight + _TOLERANCE
    _rule("max_stock_weight_respected", "error", not over_cap.any(),
          int(over_cap.sum()),
          [f"{r.ticker}@{r.rebalance_date.date()}:{r.target_weight:.4f}"
           for r in portfolios[over_cap].head(10).itertuples()],
          f"Per-name cap {max_stock_weight} must never be exceeded.")

    no_lookahead = portfolios["trained_through"] >= portfolios["rebalance_date"]
    _rule("no_lookahead_signals", "error", not no_lookahead.any(),
          int(no_lookahead.sum()),
          [f"{r.ticker}@{r.rebalance_date.date()}"
           for r in portfolios[no_lookahead].head(10).itertuples()],
          "Every signal must come from a fold whose training ended "
          "strictly before the rebalance date.")

    # ---- per-rebalance invariants -----------------------------------------
    if rebalance_records:
        counts = [r["n_holdings"] for r in rebalance_records]
        over_n = [r["rebalance_date"] for r, n in
                  zip(rebalance_records, counts, strict=True) if n > max_holdings]
        _rule("max_holdings_respected", "error", not over_n,
              len(over_n), [str(d) for d in over_n[:10]],
              f"Holding count must never exceed {max_holdings}.")

        gross = [r["cash_weight"] + sum(
            portfolios.loc[portfolios["rebalance_date"] == r["rebalance_date"],
                           "target_weight"])
            for r in rebalance_records]
        bad_gross = [
            r["rebalance_date"] for r, g in
            zip(rebalance_records, gross, strict=True)
            if abs(g - 1.0) > 1e-4
        ]
        _rule("gross_exposure_including_cash_is_one", "error", not bad_gross,
              len(bad_gross), [str(d) for d in bad_gross[:10]],
              "Stocks + recorded cash must sum to 1 per rebalance.")

        if require_liquidity_flag:
            illiquid = [r["rebalance_date"] for r in rebalance_records
                        if not r.get("all_liquid", True)]
            _rule("selected_names_liquid", "error", not illiquid,
                  len(illiquid), [str(d) for d in illiquid[:10]],
                  "Every selected name must carry liquid_flag == 1 at the "
                  "rebalance date.")

        short_history = [
            r["rebalance_date"] for r in rebalance_records
            if r.get("min_history_sessions", np.inf) < min_history_sessions
        ]
        _rule("min_history_respected", "error", not short_history,
              len(short_history), [str(d) for d in short_history[:10]],
              f"Every selected name needs >= {min_history_sessions} "
              "sessions of feature history.")

        not_sessions = [
            r["rebalance_date"] for r in rebalance_records
            if not r.get("is_session", True)
        ]
        _rule("rebalance_dates_are_sessions", "error", not not_sessions,
              len(not_sessions), [str(d) for d in not_sessions[:10]],
              "Portfolios can only be constructed on trading sessions.")

    # ---- documented limitations / coverage warnings -----------------------
    if not sector_rule_implemented:
        _rule("sector_weight_rule_skipped", "warn", False, 1,
              detail="max_sector_weight is NOT enforced: the platform has "
                     "no free point-in-time sector mapping (Phase 4 "
                     "limitation). Reported explicitly, never silently "
                     "ignored.")

    small = [r["rebalance_date"] for r in rebalance_records
             if r.get("n_holdings", 0) < r.get("n_holdings_target", 20)]
    _rule("under_filled_rebalances", "warn", not small, len(small),
          [str(d) for d in small[:10]],
          "Rebalances with fewer eligible names than configured; cash is "
          "kept instead of padding with ineligible stocks.")

    return report
