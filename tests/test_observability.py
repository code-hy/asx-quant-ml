"""Tests for observability: structured logging and run registry."""

from __future__ import annotations

import logging
from pathlib import Path

from asx_quant.observability.logging import configure_logging, get_logger
from asx_quant.observability.runs import RunRegistry
from asx_quant.storage.paths import DataLake


def test_get_logger_namespaces_unknown_names() -> None:
    assert get_logger("foo").name == "asx_quant.foo"
    assert get_logger("asx_quant.bar").name == "asx_quant.bar"


def test_configure_logging_is_idempotent(tmp_path: Path) -> None:
    logger = configure_logging(log_file=tmp_path / "a.jsonl")
    n_first = len(logger.handlers)
    configure_logging(log_file=tmp_path / "a.jsonl")
    assert len(logger.handlers) == n_first  # no duplicated handlers


def test_json_file_logging(tmp_path: Path) -> None:
    import json

    log_path = tmp_path / "pipe.jsonl"
    configure_logging(log_file=log_path)
    get_logger("probe").info("hello %s", "world")
    for handler in logging.getLogger("asx_quant").handlers:
        handler.flush()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    payload = json.loads(lines[-1])
    assert payload["message"] == "hello world"


def test_run_registry_records_stages(tmp_path: Path) -> None:
    lake = DataLake(root=tmp_path)
    registry = RunRegistry.start(lake, config_version="v0.1")

    ok = registry.begin_stage("stage_ok")
    registry.finish_stage(ok, record_count=10)
    bad = registry.begin_stage("stage_bad")
    registry.finish_stage(bad, status="failed", error="boom")

    entries = registry.read_log()
    assert [e["status"] for e in entries] == ["success", "failed"]
    assert entries[1]["error"] == "boom"
    latest = registry.read_latest()
    assert latest["run_id"] == registry.run_id
    assert latest["config_version"] == "v0.1"


def test_run_registry_read_log_empty(tmp_path: Path) -> None:
    lake = DataLake(root=tmp_path)
    registry = RunRegistry.start(lake, config_version="v0.1")
    assert registry.read_log() == []
