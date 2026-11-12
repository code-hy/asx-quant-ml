"""Pipeline run registry.

Every pipeline stage records: run ID, stage name, record counts, status,
errors, data watermark, model version and configuration version into
``data/reports/runs/run_log.jsonl``. This is the backbone of restartability
and reproducibility (AC31).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from asx_quant.observability.logging import get_logger
from asx_quant.storage.paths import DataLake

logger = get_logger("runs")

_RUN_LOG_FILENAME = "run_log.jsonl"
_RUN_META_FILENAME = "latest_run.json"


@dataclass
class StageRecord:
    """One stage execution within a pipeline run."""

    run_id: str
    stage: str
    started_at: str
    finished_at: str | None = None
    status: str = "running"  # running | success | failed | skipped
    record_count: int | None = None
    error: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunRegistry:
    """Append-only registry of pipeline runs and their stages."""

    log_dir: Path
    run_id: str
    config_version: str
    model_version: str | None = None
    _current_stage: StageRecord | None = field(default=None, repr=False)

    # ------------------------------------------------------------------ runs
    @classmethod
    def start(
        cls,
        lake: DataLake,
        config_version: str,
        model_version: str | None = None,
        run_id: str | None = None,
    ) -> RunRegistry:
        """Create a new run and mark it as the latest run on disk."""
        registry = cls(
            log_dir=lake.reports_dir / "runs",
            run_id=run_id or _new_run_id(),
            config_version=config_version,
            model_version=model_version,
        )
        registry.log_dir.mkdir(parents=True, exist_ok=True)
        (registry.log_dir / _RUN_META_FILENAME).write_text(
            json.dumps(
                {
                    "run_id": registry.run_id,
                    "started_at": _now_iso(),
                    "config_version": config_version,
                    "model_version": model_version,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("Run started: %s", registry.run_id)
        return registry

    def read_latest(self) -> dict[str, Any] | None:
        """Return the latest run's metadata, or ``None`` if no run has started."""
        path = self.log_dir / _RUN_META_FILENAME
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    # ---------------------------------------------------------------- stages
    def begin_stage(self, stage: str) -> StageRecord:
        """Mark the start of a pipeline stage."""
        record = StageRecord(
            run_id=self.run_id, stage=stage, started_at=_now_iso()
        )
        self._current_stage = record
        logger.info("Stage started: %s", stage)
        return record

    def finish_stage(
        self,
        record: StageRecord,
        status: str = "success",
        record_count: int | None = None,
        error: str | None = None,
        **details: Any,
    ) -> None:
        """Finalise a stage record and append it to the run log."""
        record.finished_at = _now_iso()
        record.status = status
        record.record_count = record_count
        record.error = error
        record.details = details
        self._append(record)
        level = logger.info if status == "success" else logger.error
        level("Stage %s: %s", record.stage, status)

    # ------------------------------------------------------------- internals
    def _append(self, record: StageRecord) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with (self.log_dir / _RUN_LOG_FILENAME).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(record), default=str) + "\n")

    def read_log(self) -> list[dict[str, Any]]:
        """Return all stage records across all runs, oldest first."""
        path = self.log_dir / _RUN_LOG_FILENAME
        if not path.is_file():
            return []
        with path.open("r", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _new_run_id() -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def get_run_registry(lake: DataLake, config_version: str) -> RunRegistry:
    """Convenience factory: create a registry bound to the data lake."""
    return RunRegistry.start(lake=lake, config_version=config_version)
