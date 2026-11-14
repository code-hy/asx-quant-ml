"""Portfolio page: rebalance schedule, holdings, cash, turnover."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st
from _shared import DISCLAIMER, json_report, missing_stage, target_portfolios

st.set_page_config(page_title="Portfolio", page_icon=":briefcase:",
                   layout="wide")
st.warning(DISCLAIMER)

summary = json_report("portfolio_summary")
portfolios = target_portfolios()

if summary is None or portfolios is None or portfolios.empty:
    missing_stage("portfolio", "portfolio")
    st.stop()

st.title("Target portfolio construction (Phase 6)")
st.caption(
    f"Signal: `{summary.get('signal_name')}` from model "
    f"`{summary.get('model_name')}` ({summary.get('model_version')}) Â· "
    f"weights: {summary.get('weight_method')} Â· feature set "
    f"`{summary.get('feature_set_version')}` Â· look-ahead guard: "
    "`trained_through` strictly before each rebalance.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rebalances", f"{summary.get('n_rebalances'):,}")
c2.metric("Weight rows", f"{summary.get('n_weight_rows'):,}")
c3.metric("Avg holdings", f"{summary.get('avg_holdings')}")
c4.metric("Avg turnover", f"{summary.get('avg_turnover'):.3f}")

per_rebal = pd.DataFrame(summary.get("per_rebalance", []))
if not per_rebal.empty:
    per_rebal["rebalance_date"] = pd.to_datetime(
        per_rebal["rebalance_date"])

    st.subheader("Holdings, cash and turnover per rebalance")
    import plotly.graph_objects as go  # noqa: E402

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=per_rebal["rebalance_date"], y=per_rebal["n_holdings"],
        name="holdings", marker_color="#1f77b4"))
    fig.add_trace(go.Scatter(
        x=per_rebal["rebalance_date"],
        y=per_rebal["cash_weight"], name="cash weight",
        yaxis="y2", line=dict(color="#ff7f0e", width=1.5)))
    fig.update_layout(
        height=360, margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(title="holdings"),
        yaxis2=dict(title="cash weight", overlaying="y", side="right",
                    range=[0, 1], tickformat=".0%"),
        legend=dict(orientation="h"))
    st.plotly_chart(fig, width='stretch')

    fig_t = px.line(
        per_rebal, x="rebalance_date", y="turnover",
        labels={"rebalance_date": "", "turnover": "turnover vs previous"},
        title="Turnover per rebalance")
    fig_t.update_layout(height=260, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_t, width='stretch')

    # ---- latest holdings ---------------------------------------------------
    latest_date = portfolios["rebalance_date"].max()
    latest = portfolios[portfolios["rebalance_date"] == latest_date]
    st.subheader(f"Latest target portfolio ({latest_date.date()})")
    show = latest[["ticker", "signal", "rank", "target_weight",
                   "trained_through"]].copy()
    show["signal"] = show["signal"].round(6)
    show["target_weight"] = show["target_weight"].round(4)
    c_left, c_right = st.columns([2, 3])
    with c_left:
        st.dataframe(show.sort_values("rank"), width='stretch',
                     hide_index=True)
    with c_right:
        fig_w = px.bar(
            show.sort_values("target_weight", ascending=False),
            x="ticker", y="target_weight",
            labels={"target_weight": "target weight", "ticker": ""})
        fig_w.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_w, width='stretch')

    with st.expander("All rebalance records"):
        st.dataframe(
            per_rebal[["rebalance_date", "n_holdings", "cash_weight",
                       "turnover", "method"]],
            width='stretch', hide_index=True)
