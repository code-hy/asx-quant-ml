"""Phase 7 tests: backtest engine, metrics, quality checks, orchestrator.

All offline — synthetic price frames and target portfolios in a temporary
lake. The engine's core guarantees are asserted directly:

* execution happens ONLY on the effective date (session after the signal
  close), never at the signal-day close;
* whole-share trades, cost scenario arithmetic, dust-trade skip rule;
* cash never negative; sells free capital before buys;
* splits multiply quantities with value continuity; dividends credit cash;
* NAV = cash + positions at last known close (suspension carried).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from asx_quant.backtest.engine import benchmark_curve, run_backtest
from asx_quant.backtest.metrics import compute_metrics, turnover_from_trades
from asx_quant.exceptions import StorageError

# ------------------------------------------------------------ synthetic data
SESSIONS = pd.bdate_range("2024-01-01", periods=40)


def _prices_frame(returns=(0.0, 0.01), open_gap=0.0) -> pd.DataFrame:
    """Two universe tickers + one benchmark; deterministic prices."""
    rows = []
    for ticker in ("AAA.AX", "BBB.AX"):
        price = 10.0
        for i, date in enumerate(SESSIONS):
            open_px = price * (1.0 + open_gap)
            price = price * (1.0 + returns[i % len(returns)])
            rows.append({
                "ticker": ticker, "date": date,
                "open_price": round(open_px, 6),
                "close_price": round(price, 6),
                "adjusted_close": round(price, 6),
                "dividend_amount": 0.0, "split_factor": 0.0,
            })
    bpx = 3000.0
    for date in SESSIONS:
        rows.append({"ticker": "VAS.AX", "date": date,
                     "open_price": bpx, "close_price": bpx,
                     "adjusted_close": bpx, "dividend_amount": 0.0,
                     "split_factor": 0.0})
        bpx *= 1.0005
    return pd.DataFrame(rows)


def _portfolios_frame() -> pd.DataFrame:
    """One rebalance: 50/50 AAA/BBB at SESSIONS[2], effective SESSIONS[3]."""
    return pd.DataFrame({
        "rebalance_date": [SESSIONS[2]] * 2,
        "effective_date": [SESSIONS[3]] * 2,
        "ticker": ["AAA.AX", "BBB.AX"],
        "target_weight": [0.5, 0.5],
    })


ZERO_COSTS = {"brokerage_per_trade_aud": 0.0, "brokerage_percent": 0.0,
              "slippage_percent": 0.0}
CAPITAL = {"initial_capital_aud": 100_000.0}


# ------------------------------------------------------------------- engine
def test_engine_buys_whole_shares_on_effective_date() -> None:
    result = run_backtest(_portfolios_frame(), _prices_frame(),
                          execution_cfg={"execution_price": "next_session_open"},
                          costs=ZERO_COSTS, capital_cfg=CAPITAL)
    trades = result.trades
    assert len(trades) == 2
    assert (trades["side"] == "buy").all()
    assert (trades["effective_date"] == SESSIONS[3]).all()
    assert (trades["rebalance_date"] == SESSIONS[2]).all()
    # Execution at the effective-date OPEN (10.1), sized off the signal-day
    # close NAV: target 50_000 each -> floor(50_000/10.1) = 4_950 shares.
    aaa = trades[trades["ticker"] == "AAA.AX"].iloc[0]
    assert aaa["shares"] == 4_950
    assert aaa["price_exec"] == pytest.approx(10.1)
    # NAV at the day-3 CLOSE: cash 10 + 2 * 4950 * close(10.1*1.01) = 100,999.9
    day3 = result.equity_curve[result.equity_curve["date"] == SESSIONS[3]]
    assert day3["nav"].iloc[0] == pytest.approx(100_999.9, abs=1e-6)
    assert day3["cash"].iloc[0] == pytest.approx(10.0, abs=1e-6)


def test_engine_never_trades_on_signal_day() -> None:
    result = run_backtest(_portfolios_frame(), _prices_frame(),
                          execution_cfg={"execution_price": "next_session_open"},
                          costs=ZERO_COSTS, capital_cfg=CAPITAL)
    first_trade_day = result.trades["effective_date"].min()
    assert first_trade_day > _portfolios_frame()["rebalance_date"].iloc[0]


def test_engine_execution_lag_violation_raises() -> None:
    bad = _portfolios_frame()
    bad["effective_date"] = bad["rebalance_date"]  # same-day execution
    with pytest.raises(ValueError, match="lag"):
        run_backtest(bad, _prices_frame(), costs=ZERO_COSTS,
                     capital_cfg=CAPITAL)


def test_engine_slippage_direction_and_costs() -> None:
    costs = {"brokerage_per_trade_aud": 9.5, "brokerage_percent": 0.0005,
             "slippage_percent": 0.001}
    pf = _portfolios_frame().assign(target_weight=[0.49, 0.49])
    result = run_backtest(pf, _prices_frame(),
                          execution_cfg={"execution_price": "next_session_open"},
                          costs=costs, capital_cfg=CAPITAL)
    aaa = result.trades[result.trades["ticker"] == "AAA.AX"].iloc[0]
    # buy pays up: 10.1 * 1.001
    assert aaa["price_exec"] == pytest.approx(10.1 * 1.001)
    trade_value = aaa["shares"] * 10.1 * 1.001
    expected_cost = 9.5 + 0.0005 * trade_value
    assert aaa["cost_aud"] == pytest.approx(expected_cost, abs=0.01)
    # costs reduce NAV vs the zero-cost run (100,999.9); both buys filled
    day3 = result.equity_curve[result.equity_curve["date"] == SESSIONS[3]]
    assert day3["nav"].iloc[0] < 100_999.9
    assert len(result.trades) == 2


def test_engine_skips_buy_beyond_available_cash() -> None:
    # 50/50 with fees: the first buy + costs consume enough cash that the
    # second 50% buy no longer fits -> skipped, cash stays positive
    costs = {"brokerage_per_trade_aud": 9.5, "brokerage_percent": 0.0005,
             "slippage_percent": 0.001}
    result = run_backtest(_portfolios_frame(), _prices_frame(),
                          execution_cfg={"execution_price": "next_session_open"},
                          costs=costs, capital_cfg=CAPITAL)
    assert result.diagnostics["skipped_insufficient_cash"] == 1
    assert (result.equity_curve["cash"] > -1e-9).all()
    # only one buy executed
    assert (result.trades["side"] == "buy").all()
    assert len(result.trades) == 1


def test_engine_skips_dust_trades() -> None:
    # tiny portfolio: target value 40 per name < min_trade_value 500
    result = run_backtest(
        _portfolios_frame(), _prices_frame(),
        execution_cfg={"execution_price": "next_session_open"},
        costs=ZERO_COSTS,
        capital_cfg={"initial_capital_aud": 80.0, "min_trade_value_aud": 500.0},
    )
    assert result.trades.empty
    assert result.diagnostics["skipped_dust_trades"] == 2
    # NAV stays flat at initial capital (all cash)
    assert result.equity_curve["nav"].eq(80.0).all()


def test_engine_forced_dust_exit_executes() -> None:
    """Regression (Phase 8 real run): a dropped name whose exit value is
    below min_trade_value MUST still be sold — skipping it strands the
    position and inflates n_positions past the configured cap (monthly
    30-name scenarios peaked at 31 on 2022-02). The forced exit is
    counted, never silent."""
    # AAA crashes to 0.05 before the second rebalance -> the 4,950-share
    # position is worth ~247.50 (dust, < 500) at the exit rebalance.
    rows = []
    for ticker in ("AAA.AX", "BBB.AX"):
        price = 10.0
        for i, date in enumerate(SESSIONS):
            if ticker == "AAA.AX" and i >= 10:
                price = 0.05
            rows.append({
                "ticker": ticker, "date": date,
                "open_price": round(price, 6),
                "close_price": round(price, 6),
                "adjusted_close": round(price, 6),
                "dividend_amount": 0.0, "split_factor": 0.0,
            })
    bpx = 3000.0
    for date in SESSIONS:
        rows.append({"ticker": "VAS.AX", "date": date,
                     "open_price": bpx, "close_price": bpx,
                     "adjusted_close": bpx, "dividend_amount": 0.0,
                     "split_factor": 0.0})
        bpx *= 1.0005
    prices = pd.DataFrame(rows)

    pf = pd.DataFrame({
        "rebalance_date": [SESSIONS[2], SESSIONS[10], SESSIONS[10]],
        "effective_date": [SESSIONS[3], SESSIONS[11], SESSIONS[11]],
        "ticker": ["AAA.AX", "AAA.AX", "BBB.AX"],
        "target_weight": [0.5, 0.0, 0.9],
    })
    result = run_backtest(
        pf, prices, execution_cfg={"execution_price": "next_session_open"},
        costs=ZERO_COSTS,
        capital_cfg={"initial_capital_aud": 100_000.0,
                     "min_trade_value_aud": 500.0},
    )
    sells = result.trades[
        (result.trades["ticker"] == "AAA.AX")
        & (result.trades["effective_date"] == SESSIONS[11])]
    assert len(sells) == 1  # the dust exit executed
    assert result.diagnostics["forced_dust_exits"] == 1
    # the position left the book: n_positions back to target on day 11
    day11 = result.equity_curve[
        result.equity_curve["date"] == SESSIONS[11]]
    assert day11["n_positions"].iloc[0] == 1


def test_engine_cash_never_negative_and_sells_first() -> None:
    prices = _prices_frame()
    # two rebalances: 90% into AAA, then rotate into BBB (sells free cash
    # before the buy; 90% targets leave fee headroom so the buy fills)
    pf = pd.DataFrame({
        "rebalance_date": [SESSIONS[2], SESSIONS[10], SESSIONS[2],
                           SESSIONS[10]],
        "effective_date": [SESSIONS[3], SESSIONS[11], SESSIONS[3],
                           SESSIONS[11]],
        "ticker": ["AAA.AX", "AAA.AX", "BBB.AX", "BBB.AX"],
        "target_weight": [0.9, 0.0, 0.0, 0.9],
    })
    costs = {"brokerage_per_trade_aud": 9.5, "brokerage_percent": 0.0,
             "slippage_percent": 0.0}
    result = run_backtest(pf, prices,
                          execution_cfg={"execution_price": "next_session_open"},
                          costs=costs, capital_cfg=CAPITAL)
    day11 = result.trades[result.trades["effective_date"] == SESSIONS[11]]
    assert set(day11["side"]) == {"buy", "sell"}  # rotate executed
    assert (result.equity_curve["cash"] > -1e-9).all()
    assert result.diagnostics["rebalances_executed"] == 2


def test_engine_exit_dropped_name_completely() -> None:
    pf = pd.DataFrame({
        "rebalance_date": [SESSIONS[2], SESSIONS[10], SESSIONS[10]],
        "effective_date": [SESSIONS[3], SESSIONS[11], SESSIONS[11]],
        "ticker": ["AAA.AX", "AAA.AX", "BBB.AX"],
        "target_weight": [1.0, 0.0, 1.0],
    })
    result = run_backtest(pf, _prices_frame(),
                          execution_cfg={"execution_price": "next_session_open"},
                          costs=ZERO_COSTS, capital_cfg=CAPITAL)
    # after session 11, AAA has zero shares
    day11 = result.equity_curve[result.equity_curve["date"] == SESSIONS[11]]
    # BBB was bought with the AAA proceeds at BBB's session-11 price
    assert day11["nav"].iloc[0] > 0
    sells = result.trades[(result.trades["side"] == "sell")]
    assert len(sells) == 1 and sells["ticker"].iloc[0] == "AAA.AX"
    sells_day = result.trades[result.trades["effective_date"] == SESSIONS[11]]
    assert sells_day["shares"].iloc[0] > 0


def test_engine_splits_preserve_value_and_credit_cash_in_lieu() -> None:
    prices = _prices_frame()
    # AAA consolidates 5:1 on SESSIONS[20]: factor 0.2, price x5 that day
    mask = (prices["ticker"] == "AAA.AX") & (prices["date"] == SESSIONS[20])
    prices.loc[mask, "split_factor"] = 0.2
    day19_close = prices.loc[
        (prices["ticker"] == "AAA.AX") & (prices["date"] == SESSIONS[19]),
        "close_price"].iloc[0]
    prices.loc[mask, "open_price"] = day19_close * 5.0
    prices.loc[mask, "close_price"] = day19_close * 5.0
    prices.loc[mask, "adjusted_close"] = day19_close * 5.0

    result = run_backtest(
        _portfolios_frame(), prices,
        execution_cfg={"execution_price": "next_session_open"},
        costs=ZERO_COSTS, capital_cfg=CAPITAL)
    assert result.diagnostics["split_events"] == 1
    # one-day NAV return across the split is ~0 (value continuity), NOT -80%
    nav19 = result.equity_curve[result.equity_curve["date"] == SESSIONS[19]]["nav"].iloc[0]
    nav20 = result.equity_curve[result.equity_curve["date"] == SESSIONS[20]]["nav"].iloc[0]
    assert nav20 / nav19 == pytest.approx(1.0, abs=0.01)
    # shares were consolidated to whole numbers (5000 -> 1000)
    assert result.diagnostics["cash_in_lieu_events"] in (0, 1)


def test_engine_dividends_credit_cash_to_holders() -> None:
    prices = _prices_frame()
    # AAA pays 0.25/share on SESSIONS[15] (entitlement = holder at start)
    mask = (prices["ticker"] == "AAA.AX") & (prices["date"] == SESSIONS[15])
    prices.loc[mask, "dividend_amount"] = 0.25
    result = run_backtest(
        _portfolios_frame(), prices,
        execution_cfg={"execution_price": "next_session_open"},
        costs=ZERO_COSTS, capital_cfg=CAPITAL)
    assert result.diagnostics["dividend_events"] == 1
    day15 = result.equity_curve[result.equity_curve["date"] == SESSIONS[15]]
    nav14 = result.equity_curve[result.equity_curve["date"] == SESSIONS[14]]["nav"].iloc[0]
    # 4_950 shares held * 0.25 dividend + the ~1% daily drift of both names
    assert day15["nav"].iloc[0] == pytest.approx(
        nav14 * 1.01 + 0.25 * 4_950, rel=1e-3)


def test_engine_suspension_carries_last_close() -> None:
    prices = _prices_frame()
    # AAA suspended for the last 10 sessions (no price rows)
    suspended = prices[~((prices["ticker"] == "AAA.AX")
                         & (prices["date"] >= SESSIONS[30]))]
    result = run_backtest(
        _portfolios_frame(), suspended,
        execution_cfg={"execution_price": "next_session_open"},
        costs=ZERO_COSTS, capital_cfg=CAPITAL)
    assert result.diagnostics["suspended_valuation_days"] == 10
    # NAV still finite/positive and cash unchanged through suspension
    tail = result.equity_curve[result.equity_curve["date"] >= SESSIONS[30]]
    assert np.isfinite(tail["nav"]).all()
    assert (tail["cash"] == tail["cash"].iloc[0]).all()


def test_engine_underfilled_portfolio_keeps_cash() -> None:
    pf = pd.DataFrame({
        "rebalance_date": [SESSIONS[2]],
        "effective_date": [SESSIONS[3]],
        "ticker": ["AAA.AX"],
        "target_weight": [0.5],  # only half invested, rest cash
    })
    result = run_backtest(pf, _prices_frame(),
                          execution_cfg={"execution_price": "next_session_open"},
                          costs=ZERO_COSTS, capital_cfg=CAPITAL)
    day3 = result.equity_curve[result.equity_curve["date"] == SESSIONS[3]]
    # floor(50_000/10.1) = 4_950 shares -> outlay 49_995, cash 50_005
    assert day3["cash"].iloc[0] == pytest.approx(50_005.0, abs=1e-6)


def test_engine_rejects_bad_execution_price_mode() -> None:
    with pytest.raises(ValueError, match="execution_price"):
        run_backtest(_portfolios_frame(), _prices_frame(),
                     execution_cfg={"execution_price": "vwap"},
                     costs=ZERO_COSTS, capital_cfg=CAPITAL)


def test_engine_close_execution_mode() -> None:
    result = run_backtest(_portfolios_frame(), _prices_frame(),
                          execution_cfg={"execution_price": "next_session_close"},
                          costs=ZERO_COSTS, capital_cfg=CAPITAL)
    aaa = result.trades[result.trades["ticker"] == "AAA.AX"].iloc[0]
    # executed at the effective-date CLOSE (10.1 * 1.01)
    assert aaa["price_exec"] == pytest.approx(10.201)


# ----------------------------------------------------------------- benchmark
def test_benchmark_curve_growth_and_priority() -> None:
    prices = _prices_frame()
    dates = pd.DatetimeIndex(SESSIONS[3:])
    nav = benchmark_curve(prices, dates, 100_000.0, ["VAS.AX", "IDX_AXJO"])
    assert nav.iloc[0] == pytest.approx(100_000.0)
    assert nav.iloc[-1] > nav.iloc[0]
    with pytest.raises(ValueError, match="benchmark"):
        benchmark_curve(prices, dates, 100_000.0, ["NOPE.TST"])


# ------------------------------------------------------------------- metrics
def test_metrics_known_series() -> None:
    # alternating +0.2% / +0.1% daily returns: mean 0.0015, std 0.0005
    pattern = np.array([0.002, 0.001] * 126)  # 252 daily returns
    nav = pd.DataFrame({
        "date": pd.bdate_range("2024-01-01", periods=253),
        "nav": 100_000.0 * np.concatenate(
            [[1.0], np.cumprod(1.0 + pattern)]),
    })
    m = compute_metrics(nav, trading_days=252, risk_free_annual=0.0)
    assert m["total_return"] == pytest.approx((1.002 * 1.001) ** 126 - 1,
                                              abs=1e-6)
    # CAGR annualises over n_sessions = 253 rows -> exponent 252/253
    assert m["cagr"] == pytest.approx(
        (1.002 * 1.001) ** (126 * 252 / 253) - 1, rel=1e-6)
    assert m["annual_volatility"] == pytest.approx(0.0005 * np.sqrt(252),
                                                   rel=1e-3)
    assert m["sharpe_ratio"] == pytest.approx(3 * np.sqrt(252), rel=1e-3)
    assert m["max_drawdown"] == pytest.approx(0.0, abs=1e-6)
    assert m["hit_rate"] == pytest.approx(1.0)


def test_metrics_drawdown_and_benchmark_relative() -> None:
    dates = pd.bdate_range("2024-01-01", periods=6)
    curve = pd.DataFrame({
        "date": dates,
        "nav": [100.0, 110.0, 88.0, 99.0, 110.0, 121.0],
        "benchmark_nav": [100.0, 102.0, 101.0, 103.0, 104.0, 105.0],
    })
    m = compute_metrics(curve, trading_days=252, risk_free_annual=0.0)
    assert m["max_drawdown"] == pytest.approx(88.0 / 110.0 - 1.0)
    assert m["max_drawdown_peak_date"] == str(dates[1].date())
    assert m["max_drawdown_trough_date"] == str(dates[2].date())
    assert m["max_drawdown_recovery_date"] == str(dates[4].date())
    assert "benchmark_total_return" in m
    assert m["beta"] > 0
    assert "formula_notes" in m


def test_metrics_guards_short_series() -> None:
    m = compute_metrics(pd.DataFrame({"date": [pd.Timestamp("2024-01-01")],
                                      "nav": [100.0]}))
    assert "error" in m


def test_turnover_from_trades() -> None:
    trades = pd.DataFrame({
        "effective_date": [SESSIONS[3]] * 2,
        "side": ["buy", "buy"],
        "trade_value": [50_000.0, 49_000.0],
        "nav_base": [100_000.0, 100_000.0],
        "cost_aud": [9.5, 9.5],
    })
    t = turnover_from_trades(trades)
    assert t["avg_turnover_per_rebalance"] == pytest.approx(0.495)
    assert t["total_costs_aud"] == pytest.approx(19.0)
    assert t["n_trades"] == 2
    empty = turnover_from_trades(trades.iloc[:0])
    assert empty["n_trades"] == 0


# ------------------------------------------------------------------- checks
def _curve_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.bdate_range("2024-01-01", periods=3),
        "nav": [100_000.0, 101_000.0, 100_500.0],
        "cash": [0.0, 0.0, 0.0],
        "positions_value": [100_000.0, 101_000.0, 100_500.0],
        "n_positions": [2, 2, 2],
    })


def _trades_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "effective_date": [SESSIONS[3]],
        "rebalance_date": [SESSIONS[2]],
        "ticker": ["AAA.AX"],
        "side": ["buy"], "shares": [5000],
        "price_exec": [10.0], "trade_value": [50_000.0],
        "cost_aud": [9.5], "nav_base": [100_000.0],
    })


def test_backtest_quality_clean_passes() -> None:
    from asx_quant.backtest.checks import run_backtest_quality_checks

    report = run_backtest_quality_checks(
        _curve_frame(), _trades_frame(),
        {"rebalances_planned": 1, "rebalances_executed": 1,
         "split_events": 0, "cash_in_lieu_events": 0, "dividend_events": 0},
        max_holdings=20)
    assert report.passed, [r.name for r in report.errors if not r.passed]


@pytest.mark.parametrize("mutate,rule", [
    (lambda c: c.assign(nav=[100_000.0, -1.0, 101.0]), "nav_finite_positive"),
    (lambda c: c.assign(cash=[0.0, -500.0, 0.0]), "cash_never_negative"),
    (lambda c: c.assign(n_positions=[2, 2, 25]), "max_holdings_respected"),
])
def test_backtest_quality_curve_violations(mutate, rule: str) -> None:
    from asx_quant.backtest.checks import run_backtest_quality_checks

    report = run_backtest_quality_checks(
        mutate(_curve_frame()), _trades_frame(),
        {"rebalances_planned": 1, "rebalances_executed": 1},
        max_holdings=20)
    target = next(r for r in report.results if r.name == rule)
    assert target.severity == "error" and not target.passed


def test_backtest_quality_trade_and_coverage_violations() -> None:
    from asx_quant.backtest.checks import run_backtest_quality_checks

    # lag violation
    bad_trades = _trades_frame().assign(
        effective_date=SESSIONS[2], rebalance_date=SESSIONS[2])
    report = run_backtest_quality_checks(
        _curve_frame(), bad_trades,
        {"rebalances_planned": 1, "rebalances_executed": 1}, max_holdings=20)
    rule = next(r for r in report.results
                if r.name == "execution_after_signal_date")
    assert rule.severity == "error" and not rule.passed

    # fractional shares
    frac = _trades_frame().assign(shares=[100.5])
    report2 = run_backtest_quality_checks(
        _curve_frame(), frac,
        {"rebalances_planned": 1, "rebalances_executed": 1}, max_holdings=20)
    rule2 = next(r for r in report2.results if r.name == "whole_share_trades")
    assert not rule2.passed

    # missed rebalance
    report3 = run_backtest_quality_checks(
        _curve_frame(), _trades_frame(),
        {"rebalances_planned": 5, "rebalances_executed": 1}, max_holdings=20)
    rule3 = next(r for r in report3.results
                 if r.name == "all_rebalances_executed")
    assert not rule3.passed


def test_backtest_quality_empty_curve_is_error() -> None:
    from asx_quant.backtest.checks import run_backtest_quality_checks

    report = run_backtest_quality_checks(
        pd.DataFrame(columns=["date", "nav", "cash", "positions_value",
                              "n_positions"]),
        pd.DataFrame(), {}, max_holdings=20)
    assert not next(r for r in report.results
                    if r.name == "nav_series_present").passed


# --------------------------------------------------------- e2e orchestrator
class _S:
    data_root = None  # unused; lake passed explicitly


def test_build_backtest_end_to_end(tmp_path, monkeypatch) -> None:
    from asx_quant.observability.runs import RunRegistry
    from asx_quant.storage.paths import DataLake

    cfg = {
        "backtest": {
            "signal": {"dataset": "target_portfolios"},
            "benchmark": {"ticker_priority": ["VAS.AX"]},
            "execution": {"execution_price": "next_session_open",
                          "signal_to_execution_lag_sessions": 1},
            "transaction_costs": {
                "active_scenario": "base",
                "scenarios": {"base": {
                    "brokerage_per_trade_aud": 9.5,
                    "brokerage_percent": 0.0005,
                    "slippage_percent": 0.001}},
            },
            "capital": {"initial_capital_aud": 100_000.0,
                        "allow_fractional_shares": False,
                        "min_trade_value_aud": 0.0},
            "risk_free_rate": {"annual_rate": 0.0},
            "metrics": {"trading_days_per_year": 252},
            "periods": {"test_start": "2024-02-01"},
        },
        "portfolio": {"constraints": {"max_holdings": 20}},
    }
    monkeypatch.setattr("asx_quant.backtest.build.load_yaml",
                        lambda name: cfg)

    lake = DataLake(root=tmp_path / "data")
    lake.ensure_all()
    registry = RunRegistry.start(lake, config_version="test")

    prices = _prices_frame()
    prices.to_parquet(lake.silver_dataset_path("equity_prices"), index=False)
    # 0.49/0.49 so both buys fit within cash after fees/slippage
    pf = _portfolios_frame().assign(target_weight=[0.49, 0.49])
    pf.to_parquet(lake.gold_dataset_path("target_portfolios"), index=False)

    from asx_quant.backtest.build import build_backtest

    result = build_backtest(lake, _S(), registry)
    assert result.n_trades == 2
    assert result.final_nav > 0

    curve = pd.read_parquet(result.curve_path)
    assert (curve["nav"] > 0).all()
    assert "benchmark_nav" in curve.columns
    assert curve["benchmark_nav"].iloc[0] == pytest.approx(100_000.0)

    trades = pd.read_parquet(result.trades_path)
    assert (trades["side"] == "buy").all()
    assert (trades["cost_aud"] > 0).all()  # base scenario applied

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["cost_scenario"] == "base"
    assert metrics["turnover"]["n_trades"] == 2
    assert "test_window" in metrics  # periods.test_start configured

    quality = json.loads(
        (tmp_path / "data/reports/backtest_quality.json").read_text(
            encoding="utf-8"))
    assert quality["status"] == "pass"


def test_build_backtest_rejects_missing_portfolios(tmp_path,
                                                   monkeypatch) -> None:
    from asx_quant.observability.runs import RunRegistry
    from asx_quant.storage.paths import DataLake

    cfg = {"backtest": {"signal": {"dataset": "target_portfolios"},
                        "transaction_costs": {"active_scenario": "base",
                                              "scenarios": {"base": {}}}}}
    monkeypatch.setattr("asx_quant.backtest.build.load_yaml",
                        lambda name: cfg)
    lake = DataLake(root=tmp_path / "data")
    lake.ensure_all()
    registry = RunRegistry.start(lake, config_version="test")

    from asx_quant.backtest.build import build_backtest

    with pytest.raises(StorageError, match="portfolio"):
        build_backtest(lake, _S(), registry)


def test_backtest_config_file_is_valid() -> None:
    """The real backtest.yaml parses and is internally consistent."""
    from asx_quant.config.loader import load_yaml

    cfg = load_yaml("backtest")["backtest"]
    assert cfg["execution"]["execution_price"] in (
        "next_session_open", "next_session_close")
    assert cfg["execution"]["signal_to_execution_lag_sessions"] >= 1
    costs = cfg["transaction_costs"]
    assert costs["active_scenario"] in costs["scenarios"]
    for scenario in costs["scenarios"].values():
        assert scenario["slippage_percent"] >= 0
        assert scenario["brokerage_percent"] >= 0
    assert cfg["capital"]["initial_capital_aud"] > 0
    assert cfg["benchmark"]["ticker_priority"]
    assert cfg["periods"]["test_start"]
