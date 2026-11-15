"""Phase 9 tests: artifact loaders, platform snapshot, report + update stages.

Offline: a temporary lake populated with tiny artifacts. The guarantees:

* loaders return None/empty for MISSING artifacts — the dashboard and the
  report stage must never fabricate numbers for stages that have not run;
* ``platform_status`` consolidates run state, per-stage quality statuses
  and headline metrics into one read-only dict;
* ``run_report_stage`` writes ``platform_summary.json`` without running
  any pipeline stage;
* ``run_update_stage`` chains stages in order, aborts at the first
  failure, and still refreshes the consolidated snapshot.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from asx_quant.pipeline import stages
from asx_quant.reporting import artifacts

# ---------------------------------------------------------------- fixtures


@pytest.fixture()
def lake_root(tmp_path, monkeypatch) -> Path:
    """A temporary lake root, wired through the env override."""
    root = tmp_path / "data"
    (root / "reports" / "runs").mkdir(parents=True)
    (root / "gold").mkdir()
    monkeypatch.setenv("ASX_QUANT_DATA_ROOT", str(root))
    return root


def _write_quality(root: Path, name: str, status: str = "pass",
                   warn: int = 0) -> None:
    path = root / "reports" / f"{name}.json"
    path.write_text(json.dumps({
        "dataset": name, "status": status, "warn_failures": warn,
        "rules": [],
    }), encoding="utf-8")


# ------------------------------------------------------------ artifact loaders
def test_missing_artifacts_read_as_none(lake_root) -> None:
    assert artifacts.load_json_report("backtest_metrics") is None
    assert artifacts.load_equity_curve() is None
    assert artifacts.load_trades() is None
    assert artifacts.load_target_portfolios() is None
    assert artifacts.load_sensitivity_results() is None
    assert artifacts.load_run_log() == []
    status = artifacts.platform_status()
    assert status["headline"] == {}
    assert status["sensitivity"] is None
    assert all(e["status"] == "not_run" for e in status["stages"].values())


def test_loaders_return_written_artifacts(lake_root) -> None:
    curve = pd.DataFrame({
        "date": pd.bdate_range("2024-01-01", periods=5),
        "nav": [100.0, 101.0, 100.5, 102.0, 103.0],
    })
    curve.to_parquet(lake_root / "gold" / "backtest_equity_curve.parquet",
                     index=False)
    loaded = artifacts.load_equity_curve()
    assert loaded is not None
    assert len(loaded) == 5
    assert loaded["nav"].iloc[-1] == pytest.approx(103.0)

    (lake_root / "reports" / "backtest_metrics.json").write_text(
        json.dumps({"final_nav": 103.0, "total_return": 0.03}),
        encoding="utf-8")
    status = artifacts.platform_status()
    assert status["headline"]["final_nav"] == pytest.approx(103.0)
    assert status["artifacts"]["backtest_equity_curve"] is not None


def test_platform_status_aggregates_quality(lake_root) -> None:
    _write_quality(lake_root, "backtest_quality", status="warn", warn=3)
    _write_quality(lake_root, "portfolio_quality", status="pass")
    status = artifacts.platform_status()
    assert status["stages"]["backtest"]["status"] == "warn"
    assert status["stages"]["backtest"]["reports"]["backtest_quality"][
        "warn_failures"] == 3
    assert status["stages"]["portfolio"]["status"] == "pass"
    assert status["stages"]["ingest"]["status"] == "not_run"
    assert "not financial advice" in status["disclaimer"].lower()


def test_run_log_reads_jsonl(lake_root) -> None:
    log_path = lake_root / "reports" / "runs" / "run_log.jsonl"
    log_path.write_text(
        json.dumps({"run_id": "r1", "stage": "ingest", "status": "success",
                    "started_at": "2026-01-01T00:00:00Z"}) + "\n"
        + json.dumps({"run_id": "r1", "stage": "validate",
                      "status": "success",
                      "started_at": "2026-01-01T00:05:00Z"}) + "\n",
        encoding="utf-8")
    log = artifacts.load_run_log()
    assert len(log) == 2
    assert log[0]["stage"] == "ingest"
    assert log[1]["stage"] == "validate"


# ------------------------------------------------------------- report stage
class _FakeSettings:
    project_version = "test"


def test_run_report_stage_writes_summary(lake_root, monkeypatch) -> None:
    _write_quality(lake_root, "backtest_quality", status="warn", warn=2)
    (lake_root / "reports" / "backtest_metrics.json").write_text(
        json.dumps({"final_nav": 159_395.37, "total_return": 0.6527,
                    "sharpe_ratio": 0.289, "max_drawdown": -0.4118}),
        encoding="utf-8")

    monkeypatch.setattr(
        stages.DataLake, "from_settings",
        classmethod(lambda cls, settings: _make_lake(lake_root)))

    summary = stages.run_report_stage(_FakeSettings())
    assert summary["final_nav"] == pytest.approx(159_395.37)
    assert "backtest" in summary["stages_run"]
    assert "ingest" not in summary["stages_run"]  # no artifact written

    written = json.loads(
        (lake_root / "reports" / "platform_summary.json").read_text(
            encoding="utf-8"))
    assert written["headline"]["final_nav"] == pytest.approx(159_395.37)
    assert written["stages"]["backtest"]["status"] == "warn"
    # a run-registry record was appended
    log = artifacts.load_run_log()
    assert log[-1]["stage"] == "report"
    assert log[-1]["status"] == "success"


def _make_lake(root: Path):
    from asx_quant.storage.paths import DataLake

    return DataLake(root=root)


# ------------------------------------------------------------- update stage
def test_run_update_stage_chains_in_order(lake_root, monkeypatch) -> None:
    calls: list[str] = []

    def fake_ingest(settings, mode="historical"):
        calls.append(f"ingest:{mode}")
        return {"records": 1}

    for name in ("validate", "transform", "features", "train", "portfolio",
                 "backtest"):
        def fake(settings, mode="historical", _name=name):
            calls.append(_name)
            return {"ok": True}

        monkeypatch.setattr(stages, f"run_{name}", fake)

    monkeypatch.setattr(
        stages.DataLake, "from_settings",
        classmethod(lambda cls, settings: _make_lake(lake_root)))
    monkeypatch.setattr(stages, "run_ingest", fake_ingest)

    summary = stages.run_update_stage(_FakeSettings(), mode="incremental")
    assert summary["failed"] is None
    assert summary["stages"]["report"]["stages_run"] is not None
    assert calls[0] == "ingest:incremental"
    assert calls[-1] == "backtest"
    # the report stage ran last inside the update
    assert "report" in summary["stages"]
    assert summary["stages"]["report"]["final_nav"] is None  # no artifacts


# ------------------------------------------------------------ phase 10 docs
def test_phase10_documents_exist_and_are_honest(repo_root) -> None:
    """The final report, presentation and beginner explainer exist, and the
    explainer carries the not-financial-advice framing."""
    docs = repo_root / "docs"
    for name in ("final_report.md", "presentation.md", "explainer.md",
                 "phase_explainer.md"):
        path = docs / name
        assert path.is_file(), f"missing Phase 10 deliverable: {name}"
        assert path.stat().st_size > 2_000, f"suspiciously small: {name}"

    explainer = (docs / "explainer.md").read_text(encoding="utf-8")
    assert "not financial advice" in explainer.lower()
    assert "machine learning" in explainer.lower()
    report = (docs / "final_report.md").read_text(encoding="utf-8")
    assert "cost drag" in report.lower()
    assert "NOT a leaderboard" in report
    # the strategy-vs-benchmark headline must be stated, not buried
    assert "65.27%" in report and "79.02%" in report

    phase_explainer = (docs / "phase_explainer.md").read_text(
        encoding="utf-8")
    # every phase 1..10 must have its own section
    for phase_no in range(1, 11):
        assert f"# Phase {phase_no}" in phase_explainer, (
            f"phase_explainer.md missing the Phase {phase_no} section")
    # the explainer must teach the fundamentals, not assume them
    for term in ("feature", "backtest", "Sharpe", "drawdown",
                 "walk-forward"):
        assert term in phase_explainer


def test_phase10_notebooks_exist_and_are_executed(repo_root) -> None:
    """Both deliverable notebooks exist with real executed outputs: the EDA
    notebook answers the EDA criteria (target description, feature-vs-target
    correlations, per-ticker prediction/precision analysis) and the workflow
    notebook demonstrates the whole pipeline via the package's own stage
    functions."""
    notebooks = repo_root / "notebooks"
    for name in ("01_eda.ipynb", "02_end_to_end_workflow.ipynb"):
        path = notebooks / name
        assert path.is_file(), f"missing notebook deliverable: {name}"

    eda = (notebooks / "01_eda.ipynb").read_text(encoding="utf-8")
    # EDA criteria: describe TO_PREDICT, correlate features with it, and
    # analyse predictions/precision by ticker.
    for marker in ("TO_PREDICT", "future_return_3d", "spearman", "corrwith",
                   "per-ticker", "precision"):
        assert marker in eda, f"01_eda.ipynb missing EDA content: {marker}"

    workflow = (notebooks / "02_end_to_end_workflow.ipynb").read_text(
        encoding="utf-8")
    # the workflow notebook must call every pipeline stage function and show
    # the latest trades
    for stage in ("run_ingest", "run_validate", "run_transform", "run_train",
                  "run_portfolio", "run_backtest", "run_report_stage",
                  "load_target_portfolios", "load_trades"):
        assert stage in workflow, f"02_end_to_end_workflow.ipynb missing: {stage}"

    # both notebooks must have been executed: every code cell carries an
    # execution count (nbclient stamps 1..n on executed cells)
    for name in ("01_eda.ipynb", "02_end_to_end_workflow.ipynb"):
        raw = (notebooks / name).read_text(encoding="utf-8")
        assert '"execution_count": null' not in raw, (
            f"{name} contains an unexecuted code cell")


def test_run_update_stage_aborts_on_failure(lake_root, monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        stages.DataLake, "from_settings",
        classmethod(lambda cls, settings: _make_lake(lake_root)))

    def fake_ingest(settings, mode="historical"):
        calls.append("ingest")
        return {"records": 1}

    def fake_validate(settings, mode="historical"):
        calls.append("validate")
        raise RuntimeError("bronze data is bad")

    def fail_if_called(settings, mode="historical"):  # pragma: no cover
        calls.append("SHOULD NOT RUN")
        raise AssertionError("stage ran after failure")

    monkeypatch.setattr(stages, "run_ingest", fake_ingest)
    monkeypatch.setattr(stages, "run_validate", fake_validate)
    monkeypatch.setattr(stages, "run_transform", fail_if_called)

    summary = stages.run_update_stage(_FakeSettings())
    assert summary["failed"] == "validate"
    assert calls == ["ingest", "validate"]  # chain aborted, nothing after
    assert "error" in summary["stages"]["validate"]
    # snapshot still refreshed (reflects the actual lake state)
    assert "report" in summary["stages"]
