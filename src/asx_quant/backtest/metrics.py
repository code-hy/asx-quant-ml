"""Backtest performance metrics from a daily NAV series.

Every formula is documented in the docstrings and written verbatim into
``backtest_metrics.json`` (``formula_notes``) so the report is self-describing.
Risk-free rate: constant annual assumption from ``backtest.yaml``
(``risk_free_rate.annual_rate``); the daily equivalent is a simple
``annual / 252`` division (documented simplification).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from asx_quant.observability.logging import get_logger

logger = get_logger("backtest.metrics")

_EPS = 1e-12


def compute_metrics(
    curve: pd.DataFrame,
    trading_days: int = 252,
    risk_free_annual: float = 0.0425,
) -> dict:
    """Performance metrics from an equity-curve frame.

    Args:
        curve: Frame with ``date`` and ``nav`` (sorted, one row per session);
            optional ``benchmark_nav`` enables relative metrics.
        trading_days: Sessions per year (config).
        risk_free_annual: Constant annual risk-free assumption.

    Returns:
        Metrics dict (JSON-safe primitives; None for undefined values).
    """
    nav = curve["nav"].astype(float).reset_index(drop=True)
    if len(nav) < 2 or not np.isfinite(nav).all():
        return {"error": "NAV series too short or non-finite"}

    rf_daily = risk_free_annual / trading_days
    returns = nav.pct_change().dropna()
    n_sessions = len(nav)

    total_return = float(nav.iloc[-1] / nav.iloc[0] - 1.0)
    years = n_sessions / trading_days
    cagr = float((nav.iloc[-1] / nav.iloc[0]) ** (1.0 / years) - 1.0) \
        if years > 0 else None

    mean_ret = float(returns.mean())
    std_ret = float(returns.std(ddof=0))
    annual_vol = std_ret * np.sqrt(trading_days) if std_ret > _EPS else None
    sharpe = ((mean_ret - rf_daily) / std_ret * np.sqrt(trading_days)
              if std_ret > _EPS else None)

    downside = np.minimum(returns.to_numpy() - rf_daily, 0.0)
    downside_dev = float(np.sqrt(np.mean(downside ** 2)))
    sortino = ((mean_ret - rf_daily) * trading_days / (downside_dev
               * np.sqrt(trading_days))
               if downside_dev > _EPS else None)

    running_max = nav.cummax()
    drawdown = nav / running_max - 1.0
    max_dd = float(drawdown.min())
    peak_idx = int(drawdown.idxmin())
    peak_date = curve["date"].iloc[int(nav[:peak_idx + 1].idxmax())]
    trough_date = curve["date"].iloc[peak_idx]
    # Recovery: first session after the trough back at the peak NAV.
    after = nav.iloc[peak_idx:][nav.iloc[peak_idx:] >= running_max.iloc[peak_idx]]
    recovery_date = (curve["date"].iloc[after.index[0]]
                     if len(after) else None)

    out = {
        "n_sessions": int(n_sessions),
        "first_date": str(curve["date"].iloc[0].date()),
        "last_date": str(curve["date"].iloc[-1].date()),
        "initial_nav": round(float(nav.iloc[0]), 2),
        "final_nav": round(float(nav.iloc[-1]), 2),
        "total_return": round(total_return, 6),
        "cagr": round(cagr, 6) if cagr is not None else None,
        "annual_volatility": (round(annual_vol, 6)
                              if annual_vol is not None else None),
        "sharpe_ratio": round(sharpe, 4) if sharpe is not None else None,
        "sortino_ratio": round(sortino, 4) if sortino is not None else None,
        "max_drawdown": round(max_dd, 6),
        "max_drawdown_peak_date": str(pd.Timestamp(peak_date).date()),
        "max_drawdown_trough_date": str(pd.Timestamp(trough_date).date()),
        "max_drawdown_recovery_date": (
            str(pd.Timestamp(recovery_date).date())
            if recovery_date is not None else None),
        "hit_rate": round(float((returns > 0).mean()), 4),
        "best_day": round(float(returns.max()), 6),
        "worst_day": round(float(returns.min()), 6),
    }

    # ---- relative metrics (need the benchmark column) --------------------
    if "benchmark_nav" in curve.columns:
        bench = curve["benchmark_nav"].astype(float).reset_index(drop=True)
        bench_returns = bench.pct_change().dropna()
        common = returns.index.intersection(bench_returns.index)
        pr = returns.loc[common].to_numpy()
        br = bench_returns.loc[common].to_numpy()
        active = pr - br
        active_std = float(active.std(ddof=0))
        bench_total = float(bench.iloc[-1] / bench.iloc[0] - 1.0)
        bench_cagr = ((bench.iloc[-1] / bench.iloc[0]) ** (1.0 / years) - 1.0
                      if years > 0 and bench.iloc[0] > 0 else None)
        beta = (float(np.cov(pr, br, ddof=0)[0, 1] / np.var(br, ddof=0))
                if np.var(br, ddof=0) > _EPS else None)
        bench_mean = float(br.mean())
        alpha_ann = (((pr.mean() - rf_daily) - beta * (bench_mean - rf_daily))
                     * trading_days if beta is not None else None)
        info_ratio = (float(active.mean() * trading_days / active_std)
                      if active_std > _EPS else None)
        out.update({
            "benchmark_total_return": round(bench_total, 6),
            "benchmark_cagr": (round(bench_cagr, 6)
                               if bench_cagr is not None else None),
            "excess_return": round(total_return - bench_total, 6),
            "tracking_error": (round(active_std * np.sqrt(trading_days), 6)
                               if common.size > 1 else None),
            "information_ratio": (round(info_ratio, 4)
                                  if info_ratio is not None else None),
            "beta": round(beta, 4) if beta is not None else None,
            "alpha_annualised": (round(alpha_ann, 6)
                                 if alpha_ann is not None else None),
        })

    out["formula_notes"] = {
        "rf_daily": "risk_free_annual / trading_days (simple division)",
        "sharpe": "(mean(daily) - rf_daily) / std(daily) * sqrt(252)",
        "sortino": "(mean(daily) - rf_daily) * 252 / "
                   "(downside_dev * sqrt(252)); downside_dev = sqrt of the "
                   "mean of squared negative (daily - rf_daily)",
        "max_drawdown": "min(nav / running_max - 1)",
        "cagr": "(final/initial) ** (trading_days / n_sessions) - 1",
        "alpha_annualised": "(daily_excess - beta * bench_excess) * 252",
        "information_ratio": "mean(active) * 252 / std(active)",
        "benchmark": "frictionless buy-and-hold of adjusted_close growth "
                     "(no costs applied to the benchmark)",
    }
    return out


def turnover_from_trades(trades: pd.DataFrame) -> dict:
    """Realised turnover and cost aggregates from executed trades.

    Per-rebalance turnover = 0.5 * sum(|trade_value|) / nav_base — directly
    comparable with the Phase 6 target-weight estimate (0.5/2 conventions
    match: both count a full exit + entry as 1.0).
    """
    if trades.empty:
        return {"avg_turnover_per_rebalance": None, "total_costs_aud": 0.0,
                "cost_drag_pct_of_initial": 0.0, "n_trades": 0}
    per_rebal = (
        trades.groupby("effective_date")
        .apply(lambda g: 0.5 * g["trade_value"].abs().sum()
               / g["nav_base"].iloc[0], include_groups=False)
    )
    total_costs = float(trades["cost_aud"].sum())
    initial = float(trades["nav_base"].iloc[0])
    return {
        "avg_turnover_per_rebalance": round(float(per_rebal.mean()), 4),
        "max_turnover_per_rebalance": round(float(per_rebal.max()), 4),
        "total_costs_aud": round(total_costs, 2),
        "cost_drag_pct_of_initial": round(total_costs / initial, 6),
        "n_trades": int(len(trades)),
        "n_buy_trades": int((trades["side"] == "buy").sum()),
        "n_sell_trades": int((trades["side"] == "sell").sum()),
    }
