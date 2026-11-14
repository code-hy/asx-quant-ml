"""Backtest engine: target portfolios -> simulated NAV with real frictions.

Execution model (config/backtest.yaml, every assumption explicit):

* LAG — the signal is computed at the rebalance-date close (session t); all
  execution happens at the NEXT session (``effective_date``, t+1). The engine
  never trades at the close that produced the signal.
* PRICE — next-session open (default) or next-session close, adjusted for
  slippage: buys pay ``price * (1 + slippage)``, sells receive
  ``price * (1 - slippage)``. Brokerage = fixed AUD per trade + percent of
  trade value, from the active cost scenario.
* WHOLE SHARES — trades are whole shares: the target quantity is
  ``floor(target_value / execution_price)`` with ``target_value =
  target_weight * NAV(previous close)``. ``min_trade_value_aud`` dust trades
  are skipped and counted — EXCEPT full exits of dropped names, which always
  execute (a stranded dust position would break the holdings cap; forced
  dust exits are counted in ``forced_dust_exits``).
* CASH — sells execute before buys (sells free capital); a buy that exceeds
  available cash is skipped and counted. Cash never goes negative: no
  leverage is invented.
* DIVIDENDS — entitlement is the share count held at the START of the
  ex-date session (post-split); credited to cash on that session.
* SPLITS — on the ex-split session, the share count is multiplied by the
  recorded split factor; the fractional remainder is converted to cash in
  lieu at that session's price. Raw prices are post-split on that date, so
  position VALUE is continuous across the split.
* SUSPENSION — a held ticker without a session row keeps its last known
  close for valuation and cannot trade that day.
* NAV — cash + sum(quantities * last known close), recorded per session from
  the first effective date onward.

All skipped trades and corporate actions are counted in the diagnostics dict
so the quality rules (checks.py) and the summary report can document them —
nothing is silently ignored.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from asx_quant.observability.logging import get_logger

logger = get_logger("backtest.engine")

_EXECUTION_PRICES = ("next_session_open", "next_session_close")


@dataclass
class BacktestResult:
    """Equity curve, executed trades and engine diagnostics."""

    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    diagnostics: dict = field(default_factory=dict)


@dataclass
class PricePivots:
    """Date x ticker price pivots, prepared once and reused across runs."""

    open_px: pd.DataFrame
    close_px: pd.DataFrame
    div_px: pd.DataFrame
    split_px: pd.DataFrame
    sessions: list


def prepare_price_pivots(prices: pd.DataFrame) -> PricePivots:
    """Pivot the silver price columns once (drop duplicate ticker/date)."""
    clean = prices.drop_duplicates(subset=["ticker", "date"], keep="last")
    open_px = clean.pivot(index="date", columns="ticker",
                          values="open_price")
    close_px = clean.pivot(index="date", columns="ticker",
                           values="close_price")
    div_px = clean.pivot(index="date", columns="ticker",
                         values="dividend_amount")
    split_px = clean.pivot(index="date", columns="ticker",
                           values="split_factor")
    return PricePivots(open_px=open_px, close_px=close_px, div_px=div_px,
                       split_px=split_px, sessions=sorted(open_px.index))


def run_backtest(
    portfolios: pd.DataFrame,
    prices: pd.DataFrame,
    execution_cfg: dict | None = None,
    costs: dict | None = None,
    capital_cfg: dict | None = None,
    pivots: PricePivots | None = None,
) -> BacktestResult:
    """Simulate the target portfolios on the silver price history.

    Args:
        portfolios: ``gold/target_portfolios.parquet`` (long format;
            requires rebalance_date, effective_date, ticker, target_weight).
        prices: Silver equity prices (ticker, date, open_price, close_price,
            dividend_amount, split_factor) including the benchmark series.
        execution_cfg: The ``backtest.execution`` config section.
        costs: ONE cost scenario dict (brokerage_per_trade_aud,
            brokerage_percent, slippage_percent).
        capital_cfg: The ``backtest.capital`` config section.
        pivots: Optional precomputed :class:`PricePivots` (the sensitivity
            runner prepares them once and reuses them across the grid).

    Returns:
        :class:`BacktestResult` with the daily equity curve, one row per
        executed trade, and diagnostics counters.
    """
    execution_cfg = dict(execution_cfg or {})
    costs = dict(costs or {})
    capital_cfg = dict(capital_cfg or {})

    exec_price_mode = str(
        execution_cfg.get("execution_price", "next_session_open"))
    if exec_price_mode not in _EXECUTION_PRICES:
        raise ValueError(
            f"Unknown execution_price '{exec_price_mode}'. "
            f"Valid: {', '.join(_EXECUTION_PRICES)}."
        )
    initial_capital = float(capital_cfg.get("initial_capital_aud", 100_000.0))
    min_trade_value = float(capital_cfg.get("min_trade_value_aud", 0.0))
    slip = float(costs.get("slippage_percent", 0.0))
    fee_fixed = float(costs.get("brokerage_per_trade_aud", 0.0))
    fee_pct = float(costs.get("brokerage_percent", 0.0))

    # ---- inputs ---------------------------------------------------------
    if portfolios.empty:
        raise ValueError("target_portfolios is empty; run the portfolio stage.")
    required = {"rebalance_date", "effective_date", "ticker", "target_weight"}
    missing = required - set(portfolios.columns)
    if missing:
        raise ValueError(
            f"target_portfolios missing columns: {sorted(missing)}."
        )

    plans = portfolios.dropna(subset=["effective_date"]).copy()
    plans["effective_date"] = pd.to_datetime(plans["effective_date"])
    plans["rebalance_date"] = pd.to_datetime(plans["rebalance_date"])
    bad_lag = plans[plans["effective_date"] <= plans["rebalance_date"]]
    if not bad_lag.empty:
        raise ValueError(
            "Execution lag violated: effective_date must be strictly after "
            f"rebalance_date ({len(bad_lag)} rows violate this)."
        )

    targets_by_effective: dict[pd.Timestamp, dict[str, float]] = {}
    rebal_by_effective: dict[pd.Timestamp, pd.Timestamp] = {}
    for eff, group in plans.groupby("effective_date"):
        weights = dict(zip(group["ticker"], group["target_weight"],
                           strict=True))
        if eff in targets_by_effective:  # merged forward-shift collision
            targets_by_effective[eff].update(weights)
        else:
            targets_by_effective[eff] = dict(weights)
        rebal_by_effective[eff] = group["rebalance_date"].iloc[0]

    if pivots is None:
        pivots = prepare_price_pivots(prices)
    open_px = pivots.open_px
    close_px = pivots.close_px
    div_px = pivots.div_px
    split_px = pivots.split_px
    sessions = list(pivots.sessions)

    start = min(targets_by_effective)
    trade_sessions = [s for s in sessions if s >= start]
    if not trade_sessions:
        raise ValueError(
            "No price sessions at/after the first effective date; the price "
            "history does not cover the portfolio period."
        )

    # ---- simulation state -------------------------------------------------
    qty: dict[str, int] = {}
    cash = initial_capital
    last_close: dict[str, float] = {}
    trades_rows: list[dict] = []
    curve_rows: list[dict] = []
    diag = {
        "rebalances_planned": len(targets_by_effective),
        "rebalances_executed": 0,
        "skipped_dust_trades": 0,
        "forced_dust_exits": 0,
        "skipped_insufficient_cash": 0,
        "skipped_no_price": 0,
        "split_events": 0,
        "cash_in_lieu_events": 0,
        "dividend_events": 0,
        "suspended_valuation_days": 0,
        "sessions_simulated": len(trade_sessions),
    }

    def _unit_cost(trade_value: float) -> float:
        return fee_fixed + fee_pct * trade_value

    def _execute(date: pd.Timestamp, nav_base: float,
                 price_lookup: pd.DataFrame) -> None:
        """Morning/close execution block for one rebalance."""
        targets = targets_by_effective[date]
        ticker_set = sorted(set(targets) | set(qty))
        intended: list[tuple[str, int, float]] = []  # ticker, delta_qty, raw px

        for ticker in ticker_set:
            weight = targets.get(ticker, 0.0)
            current = qty.get(ticker, 0)
            try:
                raw = float(price_lookup.at[date, ticker])
            except KeyError:
                raw = float("nan")
            if not raw > 0:
                if current > 0 or weight > 0:
                    diag["skipped_no_price"] += 1
                continue
            buy_price = raw * (1.0 + slip)
            sell_price = raw * (1.0 - slip)
            target_value = weight * nav_base
            if target_value > 0:
                sizing = buy_price if target_value >= current * raw \
                    else sell_price
                target_qty = int(math.floor(target_value / sizing))
            else:
                target_qty = 0  # full exit of a dropped name
            delta = target_qty - current
            if delta == 0:
                continue
            trade_value = abs(delta) * raw
            # Full exits (a dropped name) must ALWAYS execute: skipping them
            # on the dust rule strands the position outside the target
            # portfolio and inflates n_positions past the configured cap
            # (found on monthly/30-name scenarios, 2022-02). The dust rule
            # exists to save fixed brokerage on small adjustments, not to
            # trap exits. Forced dust exits are counted, never silent.
            is_full_exit = current > 0 and target_qty == 0
            if trade_value < min_trade_value and not is_full_exit:
                diag["skipped_dust_trades"] += 1
                continue
            if is_full_exit and trade_value < min_trade_value:
                diag["forced_dust_exits"] += 1
            intended.append((ticker, delta, raw))

        # Sells first (free capital), then buys largest-weight first.
        for ticker, delta, raw in intended:
            if delta >= 0:
                continue
            price = raw * (1.0 - slip)
            proceeds = -delta * price
            cost = _unit_cost(proceeds)
            cash_local[0] += proceeds - cost
            qty[ticker] = qty.get(ticker, 0) + delta
            trades_rows.append(_trade_row(date, ticker, "sell", -delta,
                                          price, proceeds, cost, nav_base))
        for ticker, delta, raw in sorted(
            (t for t in intended if t[1] > 0),
            key=lambda tr: (-targets.get(tr[0], 0.0), tr[0]),
        ):
            price = raw * (1.0 + slip)
            outlay = delta * price
            cost = _unit_cost(outlay)
            if cash_local[0] < outlay + cost:
                diag["skipped_insufficient_cash"] += 1
                continue
            cash_local[0] -= outlay + cost
            qty[ticker] = qty.get(ticker, 0) + delta
            trades_rows.append(_trade_row(date, ticker, "buy", delta,
                                          price, outlay, cost, nav_base))

    def _trade_row(date: pd.Timestamp, ticker: str, side: str, shares: int,
                   price: float, trade_value: float, cost: float,
                   nav_base: float) -> dict:
        return {
            "effective_date": date,
            "rebalance_date": rebal_by_effective.get(date),
            "ticker": ticker,
            "side": side,
            "shares": shares,
            "price_exec": round(price, 6),
            "trade_value": round(trade_value, 2),
            "cost_aud": round(cost, 4),
            "nav_base": round(nav_base, 2),
        }

    cash_local = [cash]  # writable closure state; reconciled after the loop

    # ---- session loop ------------------------------------------------------
    for date in trade_sessions:
        # 1. Splits: multiply quantity by the recorded factor; raw prices on
        #    the ex-split date are post-split, so value is continuous. The
        #    fractional remainder is paid out as cash in lieu.
        if date in split_px.index:
            split_row = split_px.loc[date]
            for ticker in list(qty):
                factor = split_row.get(ticker, float("nan"))
                if factor > 0 and factor != 1.0:
                    new_qty = qty[ticker] * float(factor)
                    whole = int(math.floor(new_qty + 1e-9))
                    try:
                        unit = float(open_px.at[date, ticker])
                    except KeyError:
                        unit = last_close.get(ticker, 0.0)
                    fraction_value = (new_qty - whole) * max(unit, 0.0)
                    if fraction_value > 0:
                        cash_local[0] += fraction_value
                        diag["cash_in_lieu_events"] += 1
                    qty[ticker] = whole
                    diag["split_events"] += 1

        # 2. Dividend entitlement: shares held at the START of the session.
        entitlement = {t: q for t, q in qty.items() if q > 0}

        is_effective = date in targets_by_effective
        nav_base = cash_local[0] + sum(
            q * last_close.get(t, 0.0) for t, q in qty.items())

        # 3. Execution (open mode: before the close update).
        if is_effective and exec_price_mode == "next_session_open":
            _execute(date, nav_base, open_px)
            diag["rebalances_executed"] += 1

        # 4. Close processing: update last known closes; credit dividends.
        if date in close_px.index:
            close_row = close_px.loc[date]
            for ticker in list(qty):
                try:
                    c = float(close_row.at[ticker])
                except KeyError:
                    c = float("nan")
                if c > 0:
                    last_close[ticker] = c
                else:
                    diag["suspended_valuation_days"] += 1
            if date in div_px.index:
                div_row = div_px.loc[date]
                for ticker, held in entitlement.items():
                    try:
                        d = float(div_row.at[ticker])
                    except KeyError:
                        d = float("nan")
                    if d > 0:
                        cash_local[0] += held * d
                        diag["dividend_events"] += 1

        # 3b. Execution (close mode: at the close, after the update).
        if is_effective and exec_price_mode == "next_session_close":
            _execute(date, nav_base, close_px)
            diag["rebalances_executed"] += 1

        # 5. Valuation.
        positions_value = sum(
            q * last_close.get(t, 0.0) for t, q in qty.items())
        curve_rows.append({
            "date": date,
            "nav": cash_local[0] + positions_value,
            "cash": cash_local[0],
            "positions_value": positions_value,
            "n_positions": sum(1 for q in qty.values() if q > 0),
        })

    cash = cash_local[0]

    equity_curve = pd.DataFrame(curve_rows)
    trades = pd.DataFrame(
        trades_rows,
        columns=["effective_date", "rebalance_date", "ticker", "side",
                 "shares", "price_exec", "trade_value", "cost_aud",
                 "nav_base"],
    )
    if not equity_curve.empty:
        equity_curve = equity_curve.sort_values("date").reset_index(drop=True)
    diag["final_cash"] = round(cash, 2)
    diag["n_trades"] = len(trades)
    logger.info(
        "Backtest simulated %d sessions, %d trades, final NAV %.2f",
        diag["sessions_simulated"], diag["n_trades"],
        float(equity_curve["nav"].iloc[-1]) if not equity_curve.empty
        else float("nan"),
    )
    return BacktestResult(
        equity_curve=equity_curve, trades=trades, diagnostics=diag)


def benchmark_curve(
    prices: pd.DataFrame,
    dates: pd.DatetimeIndex,
    initial_capital: float,
    ticker_priority: list[str],
) -> pd.Series:
    """Frictionless buy-and-hold benchmark NAV aligned to the curve dates.

    ``adjusted_close`` growth from the FIRST curve date (VAS.AX adjusted
    close includes distributions -> total-return proxy; IDX_AXJO is the
    price-only fallback). Missing dates are forward-filled.
    """
    present = set(prices["ticker"].unique())
    chosen = next((t for t in ticker_priority if t in present), None)
    if chosen is None:
        raise ValueError(
            f"None of the benchmark tickers {ticker_priority} are present "
            "in the price history."
        )
    bench = (
        prices.loc[prices["ticker"] == chosen, ["date", "adjusted_close"]]
        .drop_duplicates("date")
        .set_index("date")
        .sort_index()
    )
    aligned = bench["adjusted_close"].reindex(dates).ffill()
    base = aligned.iloc[0]
    return (aligned / base * initial_capital).rename("benchmark_nav")
