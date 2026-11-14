"""Data-quality page: rules, coverage, staleness, run log."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st
from _shared import DISCLAIMER, json_report, run_log, status_badge

st.set_page_config(page_title="Data quality", page_icon=":shield:",
                   layout="wide")
st.warning(DISCLAIMER)

st.title("Data quality & run log")

REPORTS = [
    ("ingest", "ingestion_summary"),
    ("bronze validation", "data_quality"),
    ("silver transform", "data_coverage"),
    ("features", "data_quality_features"),
    ("models", "model_quality"),
    ("portfolio", "portfolio_quality"),
    ("backtest", "backtest_quality"),
]

st.subheader("Quality reports by pipeline stage")
rows = []
for stage, name in REPORTS:
    report = json_report(name)
    if report is None:
        rows.append({"stage": stage, "report": name,
                     "status": ":grey[NOT RUN]", "warn": "-"})
        continue
    rows.append({
        "stage": stage,
        "report": name,
        "status": status_badge(report.get("status")),
        "warn": report.get("warn_failures", 0),
    })
st.markdown(
    "| stage | report | status | warnings |\n|---|---|---|---|\n"
    + "\n".join(f"| {r['stage']} | `{r['report']}` | {r['status']} | "
                f"{r['warn']} |" for r in rows))

st.divider()

# ---- failed/warn rules detail ----------------------------------------------
st.subheader("Rule detail")
selected = st.selectbox(
    "report", [name for _stage, name in REPORTS
               if json_report(name) is not None])
detail = json_report(selected)
if detail is not None:
    rules = detail.get("rules", [])
    if rules:
        rule_df = pd.DataFrame(rules)
        cols = [c for c in ("name", "severity", "passed", "failures",
                            "detail") if c in rule_df.columns]
        st.dataframe(rule_df[cols], width='stretch',
                     hide_index=True)
    else:
        st.caption("This report has no per-rule detail (summary report).")
    with st.expander("Raw JSON"):
        st.json(json.dumps(detail, indent=2, default=str))

st.divider()

# ---- run log ---------------------------------------------------------------
st.subheader("Run log")
log = run_log()
if log:
    log_df = pd.DataFrame(log)
    status_style = (
        log_df.assign(
            status=log_df["status"].map(
                lambda s: (":green[success]" if s == "success"
                           else ":red[failed]" if s == "failed"
                           else s))))
    st.dataframe(
        status_style[["run_id", "stage", "status", "started_at",
                      "record_count"]].tail(50),
        width='stretch', hide_index=True)
    latest = log_df.iloc[-1]
    st.caption(
        f"Latest: `{latest['run_id']}` Â· `{latest['stage']}` Â· "
        f"{latest['status']}")
else:
    st.info("No runs recorded yet.")
