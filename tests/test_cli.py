"""Tests for the CLI and the health-check pipeline."""

from __future__ import annotations

import pytest

from asx_quant import cli
from asx_quant.pipeline.health import run_health_check


def test_cli_health_succeeds(repo_root) -> None:
    exit_code = cli.main(["health"])
    assert exit_code == 0


def test_cli_version() -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--version"])
    assert excinfo.value.code == 0


def test_cli_phased_commands_report_not_implemented() -> None:
    # Implemented: ingest/validate/transform (Phase 2-3), features (Phase 4),
    # train (Phase 5), portfolio (Phase 6). The rest are later phases and
    # must NOT pretend to run.
    assert cli.main(["predict"]) == 2
    assert cli.main(["backtest"]) == 2


def test_health_report_all_probes_pass() -> None:
    report = run_health_check()
    failed = [c for c in report.checks if not c.passed]
    assert not failed, f"failing probes: {[(c.name, c.detail) for c in failed]}"
    assert len(report.checks) == 5


@pytest.mark.parametrize("command", ["health", "initialise"])
def test_cli_commands_are_registered(command: str) -> None:
    parser = cli.build_parser()
    args = parser.parse_args([command])
    assert args.command == command
