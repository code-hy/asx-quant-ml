"""Read-only artifact loaders for reports and dashboards (Phase 9).

Every pipeline stage already persists its own artifacts (JSON quality
reports, parquet curves/tables) into the data lake. This module provides
SAFE, SIDE-EFFECT-FREE readers over them:

* a missing artifact returns ``None`` / an empty frame — the dashboard
  and the ``report`` stage must never fabricate numbers for stages that
  have not run;
* the lake root can be overridden with the ``ASX_QUANT_DATA_ROOT``
  environment variable (used by tests and by the dashboard when it runs
  from a different working directory);
* :func:`platform_status` consolidates the latest-run state, per-stage
  artifact presence and quality statuses into one dict — the machine-
  readable snapshot the ``report`` stage writes and the Home page renders.
"""

from __future__ import annotations

import json
import os
from datetime import UTC
from pathlib import Path
from typing import Any

import pandas as pd

#: JSON report name -> the pipeline stage that owns it.
STAGE_REPORTS: dict[str, str] = {
    "ingestion_summary": "ingest",
    "data_quality": "validate",
    "data_coverage": "transform",
    "data_quality_features": "features",
    "data_quality_macro": "transform",
    "model_quality": "train",
    "portfolio_quality": "portfolio",
    "backtest_quality": "backtest",
}


def lake_root() -> Path:
    """The data-lake root: env override first, then the configured root."""
    override = os.environ.get("ASX_QUANT_DATA_ROOT")
    if override:
        return Path(override)
    from asx_quant.config.loader import load_settings

    return load_settings().data_root


def _reports_dir() -> Path:
    return lake_root() / "reports"


def _gold_dir() -> Path:
    return lake_root() / "gold"


def load_json_report(name: str) -> dict[str, Any] | None:
    """Load one JSON report from ``data/reports`` (None if absent)."""
    path = _reports_dir() / f"{name}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_parquet_artifact(directory: str, name: str) -> pd.DataFrame | None:
    """Load one parquet artifact from ``bronze/silver/gold`` (None if absent)."""
    path = lake_root() / directory / f"{name}.parquet"
    if not path.is_file():
        return None
    return pd.read_parquet(path)


def load_equity_curve() -> pd.DataFrame | None:
    return load_parquet_artifact("gold", "backtest_equity_curve")


def load_trades() -> pd.DataFrame | None:
    return load_parquet_artifact("gold", "backtest_trades")


def load_target_portfolios() -> pd.DataFrame | None:
    return load_parquet_artifact("gold", "target_portfolios")


def load_sensitivity_results() -> pd.DataFrame | None:
    return load_parquet_artifact("reports", "sensitivity_results")


def load_run_log() -> list[dict[str, Any]]:
    """All stage records across runs, oldest first (empty if no runs)."""
    path = _reports_dir() / "runs" / "run_log.jsonl"
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _artifact_mtime(name: str) -> str | None:
    for directory in ("reports", "gold"):
        path = lake_root() / directory / f"{name}.parquet"
        if not path.is_file():
            path = _reports_dir() / f"{name}.json"
        if path.is_file():
            from datetime import datetime

            mtime = datetime.fromtimestamp(
                path.stat().st_mtime, tz=UTC)
            return mtime.strftime("%Y-%m-%dT%H:%M:%SZ")
    return None


def platform_status() -> dict[str, Any]:
    """Consolidated, machine-readable snapshot of the whole platform.

    Includes: latest run, per-stage artifact presence + quality status +
    staleness, and the headline backtest numbers. Read-only: never runs a
    stage, never writes anything.
    """
    latest = None
    latest_path = _reports_dir() / "runs" / "latest_run.json"
    if latest_path.is_file():
        latest = json.loads(latest_path.read_text(encoding="utf-8"))

    stages: dict[str, dict[str, Any]] = {}
    for report_name, stage in STAGE_REPORTS.items():
        report = load_json_report(report_name)
        entry = stages.setdefault(stage, {"status": "not_run", "reports": {}})
        if report is None:
            entry["reports"][report_name] = "missing"
            continue
        entry["reports"][report_name] = {
            "status": report.get("status", "unknown"),
            "warn_failures": report.get("warn_failures", 0),
        }
        if report.get("status") in ("pass", "passed", "warn"):
            entry["status"] = report.get("status")
        elif report.get("status") == "fail":
            entry["status"] = "fail"

    metrics = load_json_report("backtest_metrics")
    headline: dict[str, Any] = {}
    if metrics is not None:
        headline = {
            "first_date": metrics.get("first_date"),
            "last_date": metrics.get("last_date"),
            "final_nav": metrics.get("final_nav"),
            "total_return": metrics.get("total_return"),
            "sharpe_ratio": metrics.get("sharpe_ratio"),
            "max_drawdown": metrics.get("max_drawdown"),
            "benchmark_total_return": metrics.get("benchmark_total_return"),
            "cost_scenario": metrics.get("cost_scenario"),
        }

    sensitivity = load_sensitivity_results()
    sensitivity_summary: dict[str, Any] | None = None
    if sensitivity is not None:
        sensitivity_summary = {
            "n_rows": int(len(sensitivity)),
            "best_sharpe": (
                float(sensitivity["sharpe_ratio"].max())
                if len(sensitivity) and sensitivity["sharpe_ratio"].notna().any()
                else None),
        }

    return {
        "latest_run": latest,
        "stages": stages,
        "headline": headline,
        "sensitivity": sensitivity_summary,
        "artifacts": {
            "backtest_equity_curve": _artifact_mtime("backtest_equity_curve"),
            "backtest_trades": _artifact_mtime("backtest_trades"),
            "target_portfolios": _artifact_mtime("target_portfolios"),
            "sensitivity_results": _artifact_mtime("sensitivity_results"),
            "backtest_metrics": _artifact_mtime("backtest_metrics"),
        },
        "disclaimer": (
            "Research-only platform. Not financial advice. No live trading. "
            "All metrics are simulated on free historical data."),
    }
