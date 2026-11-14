"""Shared helpers for the Streamlit dashboard (Phase 9).

Thin caching layer over :mod:`asx_quant.reporting.artifacts`. Every loader
returns ``None``/empty for artifacts that do not exist yet â€” pages must
render an actionable "run stage X first" message, never fake numbers.

Caches are keyed on the resolved lake root, so pointing the app (or a
test) at a different lake never serves stale frames from the old one.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from asx_quant.reporting import artifacts

DISCLAIMER = (
    "Research-only platform on free historical data. **Not financial "
    "advice.** Simulated performance, no live trading, no slippage-free "
    "promises. All metrics are diagnostics, never a selection leaderboard.")


def _root() -> str:
    return str(artifacts.lake_root())


@st.cache_data(ttl=300, show_spinner=False)
def _json_report_cached(root: str, name: str) -> dict[str, Any] | None:
    return artifacts.load_json_report(name)


@st.cache_data(ttl=300, show_spinner=False)
def _equity_curve_cached(root: str) -> pd.DataFrame | None:
    return artifacts.load_equity_curve()


@st.cache_data(ttl=300, show_spinner=False)
def _trades_cached(root: str) -> pd.DataFrame | None:
    return artifacts.load_trades()


@st.cache_data(ttl=300, show_spinner=False)
def _target_portfolios_cached(root: str) -> pd.DataFrame | None:
    return artifacts.load_target_portfolios()


@st.cache_data(ttl=300, show_spinner=False)
def _sensitivity_results_cached(root: str) -> pd.DataFrame | None:
    return artifacts.load_sensitivity_results()


@st.cache_data(ttl=300, show_spinner=False)
def _run_log_cached(root: str) -> list[dict[str, Any]]:
    return artifacts.load_run_log()


def json_report(name: str) -> dict[str, Any] | None:
    return _json_report_cached(_root(), name)


def equity_curve() -> pd.DataFrame | None:
    return _equity_curve_cached(_root())


def trades() -> pd.DataFrame | None:
    return _trades_cached(_root())


def target_portfolios() -> pd.DataFrame | None:
    return _target_portfolios_cached(_root())


def sensitivity_results() -> pd.DataFrame | None:
    return _sensitivity_results_cached(_root())


def run_log() -> list[dict[str, Any]]:
    return _run_log_cached(_root())


def status_badge(status: str | None) -> str:
    """Markdown pill for a quality status."""
    if status in ("pass", "passed"):
        return ":green[PASS]"
    if status == "warn":
        return ":orange[WARN]"
    if status == "fail":
        return ":red[FAIL]"
    return ":grey[NOT RUN]"


def missing_stage(stage: str, command: str) -> None:
    st.info(f"No artifacts for **{stage}** yet. Run `uv run asx_quant "
            f"{command}` and reload this page.")
