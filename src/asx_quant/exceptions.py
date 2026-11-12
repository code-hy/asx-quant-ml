"""Custom exceptions with meaningful, actionable messages."""

from __future__ import annotations


class AsxQuantError(Exception):
    """Base exception for all platform errors."""


class ConfigError(AsxQuantError):
    """Raised when configuration files are missing, malformed or invalid."""


class StorageError(AsxQuantError):
    """Raised when the Parquet/DuckDB storage layer cannot be initialised or used."""


class DataQualityError(AsxQuantError):
    """Raised when a data-quality rule of severity 'error' fails."""


class IngestionError(AsxQuantError):
    """Raised when a data source cannot be retrieved after retries."""


class NotFittedError(AsxQuantError):
    """Raised when a pipeline stage depends on artefacts that do not exist yet."""


class MissingDependencyError(AsxQuantError):
    """Raised when an optional component's dependency is unavailable."""
