"""Data lake directory layout and conventions.

The lake follows a medallion architecture:

* ``bronze/`` — immutable raw source data, partitioned by ingest batch
* ``silver/`` — cleaned, conformed, deduplicated datasets
* ``gold/``   — analytical datasets (features, labels, predictions, backtests)

Rules (enforced by convention and documented in docs/architecture.md):

1. Bronze data is append-only: re-ingesting writes a NEW batch file and never
   overwrites an existing one.
2. Silver/Gold datasets are reproducible from Bronze; deleting them is safe.
3. Every Parquet dataset records ``ingestion_run_id`` and ``retrieved_at``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

_BRONZE_DOMAINS = (
    "prices",
    "security_master",
    "rba",
    "abs",
    "asic",
    "announcements",
)


def utc_now_iso() -> str:
    """Current UTC time as an ISO-8601 string (single canonical helper)."""
    return datetime.now(UTC).isoformat()


def batch_id(moment: datetime | None = None) -> str:
    """Build a sortable batch identifier: ``YYYYMMDDTHHMMSSZ``."""
    moment = moment or datetime.now(UTC)
    return moment.strftime("%Y%m%dT%H%M%SZ")


@dataclass(frozen=True)
class DataLake:
    """Typed access to the project's data directories.

    Attributes:
        root: The data root directory (default ``<project>/data``).
    """

    root: Path

    # ---------------------------------------------------------------- layout
    @classmethod
    def from_settings(cls, settings) -> DataLake:  # noqa: ANN001 (cycle-safe)
        """Build a lake rooted at the configured ``paths.data_root``."""
        return cls(root=settings.data_root)

    # bronze -------------------------------------------------------------
    @property
    def bronze_dir(self) -> Path:
        return self.root / "bronze"

    def bronze_domain_dir(self, domain: str) -> Path:
        """Directory for a bronze domain (prices, rba, abs, asic, ...)."""
        if domain not in _BRONZE_DOMAINS:
            raise ValueError(
                f"Unknown bronze domain '{domain}'. Valid domains: "
                f"{', '.join(_BRONZE_DOMAINS)}."
            )
        return self.bronze_dir / domain

    def bronze_batch_path(self, domain: str, tick: str, batch: str) -> Path:
        """Path for one immutable bronze batch file.

        Format: ``bronze/<domain>/<tick>/<batch>.parquet`` where ``tick`` is a
        source ticker (or series id for macro/ASIC domains). Each batch file is
        append-only; re-ingestion creates a new ``batch`` id.
        """
        path = self.bronze_domain_dir(domain) / _safe_name(tick)
        path.mkdir(parents=True, exist_ok=True)
        return path / f"{batch}.parquet"

    def bronze_batches(self, domain: str, tick: str) -> list[Path]:
        """List existing bronze batch files for a domain/ticker, oldest first."""
        path = self.bronze_domain_dir(domain) / _safe_name(tick)
        if not path.is_dir():
            return []
        return sorted(path.glob("*.parquet"))

    # silver -------------------------------------------------------------
    @property
    def silver_dir(self) -> Path:
        return self.root / "silver"

    def silver_dataset_path(self, name: str) -> Path:
        """Path of a single-file silver dataset (e.g. ``security_master``)."""
        self.silver_dir.mkdir(parents=True, exist_ok=True)
        return self.silver_dir / f"{name}.parquet"

    # gold ---------------------------------------------------------------
    @property
    def gold_dir(self) -> Path:
        return self.root / "gold"

    def gold_dataset_path(self, name: str) -> Path:
        """Path of a single-file gold dataset (e.g. ``target_labels``)."""
        self.gold_dir.mkdir(parents=True, exist_ok=True)
        return self.gold_dir / f"{name}.parquet"

    # models / reports -----------------------------------------------------
    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def reports_dir(self) -> Path:
        return self.root / "reports"

    def report_path(self, name: str, extension: str = ".json") -> Path:
        """Path of a generated report artefact."""
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        return self.reports_dir / f"{name}{extension}"

    # housekeeping ---------------------------------------------------------
    def ensure_all(self) -> None:
        """Create the full directory skeleton (idempotent)."""
        for domain in _BRONZE_DOMAINS:
            self.bronze_domain_dir(domain).mkdir(parents=True, exist_ok=True)
        for layer in (self.silver_dir, self.gold_dir, self.models_dir, self.reports_dir):
            layer.mkdir(parents=True, exist_ok=True)
        (self.reports_dir / "runs").mkdir(parents=True, exist_ok=True)


def _safe_name(identifier: str) -> str:
    """Sanitise a ticker/series id for use as a directory name."""
    return identifier.replace("^", "IDX_").replace("/", "_").replace(":", "_")
