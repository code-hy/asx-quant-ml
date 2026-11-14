"""Sensitivity page: grid diagnostics over construction + cost assumptions.

Deliberately framed as RESEARCH DIAGNOSTICS, not a leaderboard: the report
and this page repeat that selection decisions predate this stage.
"""

from __future__ import annotations

import plotly.express as px
import streamlit as st
from _shared import DISCLAIMER, json_report, missing_stage, sensitivity_results

st.set_page_config(page_title="Sensitivity", page_icon=":microscope:",
                   layout="wide")
st.warning(DISCLAIMER)

report = json_report("sensitivity_report")
results = sensitivity_results()

if report is None or results is None or results.empty:
    missing_stage("sensitivity", "sensitivity")
    st.stop()

st.title("Sensitivity & robustness grid (Phase 8)")
st.error(
    "**This table is NOT a leaderboard.** The held-out window metrics are "
    "reported per row but every selection-relevant decision predates this "
    "stage. Use it to understand how fragile the results are, never to "
    "cherry-pick the best scenario.")

st.caption(
    f"Model `{report.get('model_name')}` ({report.get('model_version')}) Â· "
    f"signal `{report.get('signal_name')}` Â· trained horizon "
    f"{report.get('trained_horizon')} sessions Â· "
    f"{report.get('n_grid_points')} grid points.")

# ---- skipped axes ----------------------------------------------------------
skipped = report.get("skipped_axes", [])
if skipped:
    st.subheader("Skipped axes (documented, never fabricated)")
    for skip in skipped:
        st.markdown(f"- **{skip['axis']} = {skip['value']}**: "
                    f"{skip['reason']}")

# ---- filters ---------------------------------------------------------------
st.subheader("Grid results")
left, right = st.columns(2)
with left:
    cost = st.selectbox(
        "cost scenario", sorted(results["cost_scenario"].unique()),
        index=sorted(results["cost_scenario"].unique()).index("base")
        if "base" in results["cost_scenario"].unique() else 0)
with right:
    frequency = st.multiselect(
        "rebalance frequency", sorted(results["frequency"].unique()),
        default=sorted(results["frequency"].unique()))

view = results[
    (results["cost_scenario"] == cost)
    & (results["frequency"].isin(frequency or
                                 sorted(results["frequency"].unique())))]

# ---- sharpe heat map -------------------------------------------------------
if not view.empty:
    pivot = view.pivot_table(
        index="n_holdings", columns="max_stock_weight",
        values="sharpe_ratio")
    st.caption(f"Sharpe ratio â€” {cost} costs")
    fig = px.imshow(
        pivot.round(3), text_auto=True, aspect="auto",
        color_continuous_scale="RdYlGn", origin="lower",
        labels=dict(x="max stock weight", y="n holdings",
                    color="Sharpe"))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width='stretch')

    # ---- base reference ----------------------------------------------------
    base_ref = report.get("base_reference")
    if base_ref is not None:
        st.info(
            f"Base configuration (staged run): NAV "
            f"{base_ref.get('final_nav'):,.0f} Â· "
            f"Sharpe {base_ref.get('sharpe_ratio'):.3f} Â· "
            f"maxDD {base_ref.get('max_drawdown'):.1%}. The runner verifies "
            "this row reproduces the primary Phase 7 backtest "
            f"(`reference_check_vs_primary.matches` = "
            f"{(report.get('reference_check_vs_primary') or {}).get('matches')})")

    with st.expander("Full results table (all grid points)"):
        wanted = [
            "horizon", "n_holdings", "frequency", "max_stock_weight",
            "cost_scenario", "total_return", "sharpe_ratio",
            "max_drawdown", "n_trades", "total_costs_aud",
            "test_total_return", "test_sharpe"]
        st.dataframe(results[[c for c in wanted if c in results.columns]],
                     width='stretch', hide_index=True)
else:
    st.info("No grid points for the selected filters.")
