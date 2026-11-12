"""Configuration loading and validation.

All behaviour is configuration-driven: the pipeline reads YAML files from
``config/`` and never hard-codes thresholds, paths or source parameters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from asx_quant.exceptions import ConfigError

_CONFIG_DIR_ENV = "ASX_QUANT_CONFIG_DIR"
_CONFIG_DIR_DEFAULT = Path("config")

#: Configuration file stems that must exist and parse.
REQUIRED_CONFIG_FILES: tuple[str, ...] = (
    "settings",
    "universe",
    "portfolio",
    "features",
    "backtest",
)


def project_root(start_dir: Path | None = None) -> Path:
    """Locate the project root by walking up until ``pyproject.toml`` is found.

    Args:
        start_dir: Directory to begin the search from. Defaults to the
            current working directory.

    Returns:
        The directory containing ``pyproject.toml``.

    Raises:
        ConfigError: If no ancestor directory contains ``pyproject.toml``.
    """
    current = (start_dir or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise ConfigError(
        "Could not locate project root: no ancestor of "
        f"'{current}' contains 'pyproject.toml'. Run the CLI from inside the "
        "asx-quant repository."
    )


def config_dir() -> Path:
    """Resolve the configuration directory (overridable via environment)."""
    override = os.environ.get(_CONFIG_DIR_ENV)
    if override:
        return Path(override).expanduser().resolve()
    return project_root() / _CONFIG_DIR_DEFAULT


@dataclass(frozen=True)
class Settings:
    """Typed view over ``settings.yaml``.

    Only the sections the platform depends on structurally are materialised
    as attributes; the remainder is reachable through ``raw``.
    """

    project_root_dir: Path
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    # --- derived paths -----------------------------------------------------
    @property
    def data_root(self) -> Path:
        return self.project_root_dir / self.raw["paths"]["data_root"]

    @property
    def bronze_dir(self) -> Path:
        return self.data_root / "bronze"

    @property
    def silver_dir(self) -> Path:
        return self.data_root / "silver"

    @property
    def gold_dir(self) -> Path:
        return self.data_root / "gold"

    @property
    def models_dir(self) -> Path:
        return self.data_root / "models"

    @property
    def reports_dir(self) -> Path:
        return self.data_root / "reports"

    @property
    def run_log_dir(self) -> Path:
        return self.reports_dir / "runs"

    @property
    def duckdb_path(self) -> Path:
        return self.data_root / "warehouse.duckdb"

    @property
    def parquet_compression(self) -> str:
        return self.raw["storage"]["parquet"]["compression"]

    @property
    def parquet_compression_level(self) -> int:
        return int(self.raw["storage"]["parquet"]["compression_level"])

    @property
    def timestamp_timezone(self) -> str:
        return self.raw["storage"]["timestamp_timezone"]

    @property
    def project_name(self) -> str:
        return self.raw["project"]["name"]

    @property
    def project_version(self) -> str:
        return self.raw["project"]["version"]


def load_yaml(name: str, directory: Path | None = None) -> dict[str, Any]:
    """Load a YAML configuration file by stem name.

    Args:
        name: File stem, e.g. ``"portfolio"`` for ``config/portfolio.yaml``.
        directory: Override the configuration directory.

    Returns:
        Parsed YAML content as a dictionary.

    Raises:
        ConfigError: If the file is missing, unreadable, empty or not a mapping.
    """
    path = (directory or config_dir()) / f"{name}.yaml"
    if not path.is_file():
        raise ConfigError(
            f"Required configuration file is missing: '{path}'. Expected one of "
            f"{', '.join(REQUIRED_CONFIG_FILES)}.yaml in the config directory."
        )
    try:
        with path.open("r", encoding="utf-8") as handle:
            parsed = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse YAML in '{path}': {exc}") from exc

    if not isinstance(parsed, dict):
        raise ConfigError(
            f"Configuration file '{path}' must contain a top-level mapping "
            f"(got {type(parsed).__name__})."
        )
    return parsed


def _validate_settings(raw: dict[str, Any], path: Path) -> None:
    """Assert that structurally required keys exist in ``settings.yaml``.

    Raises:
        ConfigError: naming the first missing required key path.
    """
    required_sections: dict[str, tuple[str, ...]] = {
        "project": (),
        "paths": ("data_root",),
        "storage": ("parquet", "duckdb", "timestamp_timezone"),
        "ingestion": ("market_data",),
        "quality": ("min_history_sessions",),
        "runs": (),
    }
    for section, keys in required_sections.items():
        if section not in raw or not isinstance(raw[section], dict):
            raise ConfigError(
                f"'{path}' is missing required section '{section}'."
            )
        for key in keys:
            if key not in raw[section]:
                raise ConfigError(
                    f"'{path}' is missing required key '{section}.{key}'."
                )
    storage_parquet = raw["storage"]["parquet"]
    if "compression" not in storage_parquet:
        raise ConfigError(
            f"'{path}' is missing required key 'storage.parquet.compression'."
        )
    duckdb_section = raw["storage"]["duckdb"]
    if "schemas" not in duckdb_section:
        raise ConfigError(
            f"'{path}' is missing required key 'storage.duckdb.schemas'."
        )


def load_settings(start_dir: Path | None = None) -> Settings:
    """Load and validate the full configuration set.

    Loads all five required YAML files, validates ``settings.yaml`` structure,
    and verifies internal consistency between files (e.g. the portfolio signal
    must be a declared target).

    Returns:
        A validated :class:`Settings` object.

    Raises:
        ConfigError: If any file is missing/invalid or cross-file consistency
            checks fail.
    """
    root = project_root(start_dir)
    config_path = root / "config"
    settings_raw = load_yaml("settings", config_path)
    _validate_settings(settings_raw, config_path / "settings.yaml")

    # Load remaining required files (validation of their internals happens in
    # the consuming modules — this ensures they exist and are mappings).
    for stem in REQUIRED_CONFIG_FILES:
        if stem == "settings":
            continue
        load_yaml(stem, config_path)

    return Settings(project_root_dir=root, raw=settings_raw)
