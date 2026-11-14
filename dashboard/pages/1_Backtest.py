"""Backtest page: equity curve, drawdown, metrics, diagnostics, trades."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from _shared import DISCLAIMER, equity_curve, json_report, missing_stage, trades

st.set_page_config(page_title="Backtest", page_icon=":chart_with_upwards_trend:",
                   layout="wide")
st.warning(DISCLAIMER)

metrics = json_report("backtest_metrics")
curve = equity_curve()
trade_df = trades()

if metrics is None or curve is None:
    missing_stage("backtest", "backtest")
    st.stop()

st.title("Backtest (Phase 7)")
st.caption(
    "Next-session-open execution with a 1-session lag, whole shares, "
    f"cost scenario `{metrics.get('cost_scenario')}`: "
    f"{metrics.get('costs', {}).get('brokerage_per_trade_aud')} AUD fixed "
    f"+ {metrics.get('costs', {}).get('brokerage_percent')} % + "
    f"{metrics.get('costs', {}).get('slippage_percent')} % slippage.")

# ---- headline metrics ------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Final NAV", f"AUD {metrics.get('final_nav', 0):,.0f}")
c2.metric("Total return", f"{metrics.get('total_return', 0):+.1%}")
c3.metric("CAGR", f"{metrics.get('cagr', 0):+.2%}")
c4.metric("Sharpe", f"{metrics.get('sharpe_ratio', 0):.3f}")
c5.metric("Max drawdown", f"{metrics.get('max_drawdown', 0):.1%}")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Benchmark", f"{metrics.get('benchmark_total_return', 0):+.1%}")
c2.metric("Excess", f"{metrics.get('excess_return', 0):+.1%}")
c3.metric("Information ratio", f"{metrics.get('information_ratio', 0):.3f}")
c4.metric("Beta", f"{metrics.get('beta', 0):.2f}")
c5.metric("Hit rate", f"{metrics.get('hit_rate', 0):.1%}")

# ---- equity curve ----------------------------------------------------------
st.subheader("Equity curve vs benchmark")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=curve["date"], y=curve["nav"], name="Strategy NAV",
    line=dict(color="#1f77b4", width=2)))
if "benchmark_nav" in curve.columns:
    fig.add_trace(go.Scatter(
        x=curve["date"], y=curve["benchmark_nav"], name="VAS.AX (TR proxy)",
        line=dict(color="#d62728", width=1.5, dash="dash")))
fig.update_layout(
    height=420, margin=dict(l=10, r=10, t=10, b=10),
    yaxis_title="NAV (AUD)", legend=dict(orientation="h"))
st.plotly_chart(fig, width='stretch')

# ---- drawdown --------------------------------------------------------------
st.subheader("Drawdown")
nav_series = curve.set_index("date")["nav"]
running_peak = nav_series.cummax()
drawdown = nav_series / running_peak - 1.0
fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(
    x=drawdown.index, y=drawdown.values, name="Drawdown",
    fill="tozeroy", line=dict(color="#8c564b", width=1)))
fig_dd.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
                     yaxis_title="Drawdown", yaxis_tickformat=".0%")
st.plotly_chart(fig_dd, width='stretch')

# ---- metrics table ---------------------------------------------------------
left, right = st.columns(2)
with left:
    st.subheader("Full-period metrics")
    rows = {
        k: metrics.get(k)
        for k in ("n_sessions", "first_date", "last_date", "initial_nav",
                  "final_nav", "total_return", "cagr", "annual_volatility",
                  "sharpe_ratio", "sortino_ratio", "max_drawdown", "hit_rate",
                  "tracking_error", "information_ratio", "beta",
                  "alpha_annualised", "best_day", "worst_day")
    }
    st.dataframe(
        pd.DataFrame({"metric": rows.keys(), "value": rows.values()}),
        width='stretch', hide_index=True)
with right:
    st.subheader("Held-out window (diagnostics ONLY)")
    st.caption(
        "Held-out window (backtest.yaml periods.test_start). Never used "
        "for model or configuration selection.")
    tw = metrics.get("test_window") or {}
    if tw:
        st.dataframe(
            pd.DataFrame({
                "metric": ("total_return", "sharpe_ratio", "max_drawdown"),
                "value": (tw.get("total_return"), tw.get("sharpe_ratio"),
                          tw.get("max_drawdown")),
            }),
            width='stretch', hide_index=True)
    else:
        st.info("No held-out window metrics recorded.")

# ---- diagnostics -----------------------------------------------------------
st.subheader("Engine diagnostics (counted, never silent)")
diag = metrics.get("diagnostics") or {}
diag_df = pd.DataFrame(
    [{"counter": k, "value": v} for k, v in sorted(diag.items())
     if isinstance(v, (int, float)) and v != 0])
if diag_df.empty:
    st.caption("All diagnostic counters are zero.")
else:
    st.dataframe(diag_df, width='stretch', hide_index=True)

# ---- trades ----------------------------------------------------------------
if trade_df is not None and not trade_df.empty:
    st.subheader("Executed trades")
    n_total = len(trade_df)
    st.caption(f"{n_total:,} trades Â· {trade_df['ticker'].nunique()} tickers")
    side_counts = trade_df["side"].value_counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("Buys", int(side_counts.get("buy", 0)))
    c2.metric("Sells", int(side_counts.get("sell", 0)))
    c3.metric("Total costs (AUD)",
              f"{trade_df['cost_aud'].sum():,.0f}")
    with st.expander("Trade table (latest 200 rows)"):
        st.dataframe(trade_df.tail(200), width='stretch',
                     hide_index=True)
