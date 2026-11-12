"""Tests for configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from asx_quant.config.loader import (
    REQUIRED_CONFIG_FILES,
    load_settings,
    load_yaml,
    project_root,
)
from asx_quant.exceptions import ConfigError


def test_project_root_discovered_from_repo(repo_root: Path) -> None:
    assert project_root(repo_root) == repo_root


def test_project_root_raises_outside_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the ConfigError without depending on the real filesystem tree
    (a pyproject.toml in a parent directory of tmp would otherwise be found)."""
    real_is_file = Path.is_file

    def no_pyproject(self: Path) -> bool:
        if self.name == "pyproject.toml":
            return False
        return real_is_file(self)

    monkeypatch.setattr(Path, "is_file", no_pyproject)
    with pytest.raises(ConfigError, match="pyproject.toml"):
        project_root(tmp_path)


def test_all_required_config_files_exist(repo_root: Path) -> None:
    for stem in REQUIRED_CONFIG_FILES:
        path = repo_root / "config" / f"{stem}.yaml"
        assert path.is_file(), f"missing config file: {path}"


def test_load_settings_validates_and_returns_paths(repo_root: Path) -> None:
    settings = load_settings(repo_root)
    assert settings.project_name == "asx-quant"
    assert settings.data_root == repo_root / "data"
    assert settings.duckdb_path.name == "warehouse.duckdb"
    assert settings.parquet_compression == "zstd"
    assert {"bronze", "silver", "gold", "meta"} <= set(
        settings.raw["storage"]["duckdb"]["schemas"]
    )


def test_missing_config_file_raises(repo_root: Path) -> None:
    with pytest.raises(ConfigError, match="missing"):
        load_yaml("does_not_exist", repo_root / "config")


def test_malformed_yaml_raises(tmp_path: Path) -> None:
    bad = tmp_path / "broken.yaml"
    bad.write_text("key: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="Could not parse YAML"):
        load_yaml("broken", tmp_path)


def test_settings_missing_required_section_raises(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "settings.yaml").write_text(
        yaml.safe_dump({"project": {"name": "x"}}), encoding="utf-8"
    )
    from unittest import mock

    with mock.patch("asx_quant.config.loader.project_root", return_value=tmp_path):
        with pytest.raises(ConfigError, match="paths"):
            load_settings(tmp_path)


def test_portfolio_signal_matches_declared_target(repo_root: Path) -> None:
    """Cross-file consistency: portfolio signal must be a configured target."""
    portfolio = load_yaml("portfolio", repo_root / "config")
    signal = portfolio["portfolio"]["selection"]["signal"]
    # Prediction columns map onto configured targets via the model's output
    # naming: predicted_return_{h}d is the model's estimate of future_return_{h}d.
    valid = {
        "future_return_3d",
        "future_return_10d",
        "future_direction_3d",
        "future_direction_10d",
        "predicted_return_3d",
        "predicted_return_10d",
    }
    assert signal in valid, f"portfolio.signal '{signal}' is not a declared target"


def test_backtest_test_period_is_in_the_past(repo_root: Path) -> None:
    """The untouched test window must already be historical (anti-tuning guard)."""
    from datetime import date

    backtest = load_yaml("backtest", repo_root / "config")
    test_start = date.fromisoformat(backtest["backtest"]["periods"]["test_start"])
    assert test_start < date.today(), "test period must not extend into the future"
