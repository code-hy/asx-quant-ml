"""Home page: platform overview, headline results, stage statuses.

Run with:  uv run streamlit run dashboard/Home.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from _shared import DISCLAIMER, json_report, run_log, sensitivity_results, status_badge

st.set_page_config(
    page_title="ASX Quant Platform",
    page_icon=":bar_chart:",
    layout="wide",
)

st.warning(DISCLAIMER)

st.title("ASX Quantitative Research Platform")
st.caption(
    "Long-only, 20-stock ASX universe, weekly rebalance, 3-session "
    "horizon signal â€” walk-forward ML, cost-aware backtest, sensitivity "
    "grid. Local Parquet + DuckDB lake, free data sources only.")

# ---- headline ------------------------------------------------------------
metrics = json_report("backtest_metrics")
col_head, col_bench, col_quality = st.columns(3)
with col_head:
    if metrics is None:
        st.metric("Final NAV", "not run")
    else:
        st.metric(
            "Final NAV",
            f"AUD {metrics.get('final_nav', 0):,.0f}",
            f"{metrics.get('total_return', 0):+.1%} total")
with col_bench:
    if metrics is None:
        st.metric("Benchmark (VAS.AX)", "not run")
    else:
        st.metric(
            "Benchmark (VAS.AX)",
            f"{metrics.get('benchmark_total_return', 0):+.1%}",
            f"excess {metrics.get('excess_return', 0):+.1%}")
with col_quality:
    bt_q = json_report("backtest_quality")
    st.metric(
        "Backtest quality",
        "not run" if bt_q is None else bt_q.get("status", "unknown"),
        f"{(bt_q or {}).get('warn_failures', 0)} warnings")

if metrics is not None:
    st.caption(
        f"Simulated {metrics.get('first_date')} -> "
        f"{metrics.get('last_date')} ({metrics.get('n_sessions')} sessions) "
        f"Â· cost scenario: {metrics.get('cost_scenario')} Â· "
        f"Sharpe {metrics.get('sharpe_ratio')} Â· "
        f"max drawdown {metrics.get('max_drawdown'):.1%}"
        if metrics.get("max_drawdown") is not None else
        f"Simulated {metrics.get('first_date')} -> "
        f"{metrics.get('last_date')} Â· Sharpe {metrics.get('sharpe_ratio')}")

st.divider()

# ---- stage statuses --------------------------------------------------------
st.subheader("Pipeline stage status")
pipeline_q = json_report("data_quality")
features_q = json_report("data_quality_features")
model_q = json_report("model_quality")
portfolio_q = json_report("portfolio_quality")
sens_df = sensitivity_results()

stage_rows = [
    ("ingest / validate", pipeline_q),
    ("transform", json_report("data_coverage")),
    ("features", features_q),
    ("train", model_q),
    ("portfolio", portfolio_q),
    ("backtest", bt_q),
]
rows = []
for name, report in stage_rows:
    rows.append({
        "stage": name,
        "quality": status_badge((report or {}).get("status")),
        "warnings": (report or {}).get("warn_failures", "-") if report else "-",
    })
rows.append({
    "stage": "sensitivity",
    "quality": ":green[DONE]" if sens_df is not None else ":grey[NOT RUN]",
    "warnings": "-" if sens_df is None else f"{len(sens_df)} grid rows",
})
st.markdown(
    "| stage | quality | warnings |\n|---|---|---|\n"
    + "\n".join(f"| {r['stage']} | {r['quality']} | {r['warnings']} |"
                for r in rows))

st.divider()

# ---- latest run ------------------------------------------------------------
log = run_log()
if log:
    latest_run = log[-1]
    st.subheader("Latest recorded stage run")
    st.caption(
        f"run `{latest_run['run_id']}` Â· stage `{latest_run['stage']}` Â· "
        f"{latest_run['status']} Â· started {latest_run['started_at']}")
    with st.expander("Recent run-log entries"):
        st.dataframe(
            pd.DataFrame(log[-20:])[
                ["run_id", "stage", "status", "started_at", "record_count"]],
            width='stretch',
            hide_index=True,
        )
else:
    st.info("No runs recorded yet â€” run `uv run asx_quant initialise`.")

st.divider()
st.caption(
    "Navigation: **Backtest** (equity curve, metrics, trades) Â· "
    "**Portfolio** (rebalances, holdings, turnover) Â· "
    "**Sensitivity** (grid diagnostics, NOT a leaderboard) Â· "
    "**Data quality** (rules, coverage, run log).")
