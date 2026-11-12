"""Project health check.

A fast, dependency-light verification that the foundation is correctly set
up: configuration loads, directories resolve, DuckDB initialises and Parquet
round-trips. Executed via ``python -m asx_quant.cli health`` and by pytest.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from asx_quant.config.loader import Settings, load_settings
from asx_quant.exceptions import AsxQuantError
from asx_quant.observability.logging import configure_logging, get_logger
from asx_quant.observability.runs import RunRegistry
from asx_quant.storage.duckdb import Warehouse
from asx_quant.storage.parquet import ParquetWriter, read_parquet
from asx_quant.storage.paths import DataLake

logger = get_logger("health")


@dataclass
class CheckResult:
    """Outcome of one health-check probe."""

    name: str
    passed: bool
    detail: str = ""


@dataclass
class HealthReport:
    """Aggregated health-check report."""

    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def summary(self) -> str:
        lines = [f"{'PASS' if c.passed else 'FAIL'}  {c.name}" + (f"  — {c.detail}" if c.detail else "") for c in self.checks]
        status = "HEALTHY" if self.passed else "UNHEALTHY"
        return f"Health check: {status}\n" + "\n".join(lines)


def run_health_check(settings: Settings | None = None) -> HealthReport:
    """Run all foundation checks against a temporary sandbox.

    Uses a temporary data root so the check never mutates real data. Returns
    a :class:`HealthReport` with one :class:`CheckResult` per probe.
    """
    report = HealthReport()
    configure_logging()

    def _check(name: str, fn) -> None:  # noqa: ANN001
        try:
            detail = fn()
            report.checks.append(CheckResult(name=name, passed=True, detail=detail or ""))
            logger.info("Health check passed: %s", name)
        except Exception as exc:  # noqa: BLE001 - health check must capture all
            report.checks.append(CheckResult(name=name, passed=False, detail=str(exc)))
            logger.error("Health check failed: %s — %s", name, exc)

    _check("config: five YAML files load and validate", lambda: _check_config(settings))
    _check("storage: directory skeleton resolves", lambda: _check_dirs(settings))
    _check("storage: DuckDB schemas + metadata initialise", _check_duckdb)
    _check("storage: Parquet write/read round-trip", _check_parquet_roundtrip)
    _check("observability: run registry appends and reads", _check_run_registry)
    return report


# --------------------------------------------------------------------- probes
def _check_config(settings: Settings | None) -> str:
    loaded = settings or load_settings()
    names = ", ".join(sorted(loaded.raw["storage"]["duckdb"]["schemas"]))
    return f"project='{loaded.project_name}' v{loaded.project_version}; duckdb schemas: {names}"


def _check_dirs(settings: Settings | None) -> str:
    loaded = settings or load_settings()
    lake = DataLake.from_settings(loaded)
    lake.ensure_all()
    domains = [p.name for p in lake.bronze_dir.iterdir() if p.is_dir()]
    if len(domains) < 6:
        raise AsxQuantError(f"Expected 6 bronze domains, found {len(domains)}: {domains}")
    return f"bronze domains: {', '.join(sorted(domains))}"


def _check_duckdb() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        warehouse = Warehouse(Path(tmp) / "health.duckdb")
        with warehouse:
            schemas = warehouse.query(
                "SELECT schema_name FROM information_schema.schemata "
                "WHERE schema_name IN ('bronze','silver','gold','meta') "
                "ORDER BY schema_name"
            )
            got = list(schemas["schema_name"])
            if got != ["bronze", "gold", "meta", "silver"]:
                raise AsxQuantError(f"Unexpected schemas: {got}")
        return f"schemas: {got}"


def _check_parquet_roundtrip() -> str:
    with tempfile.TemporaryDirectory() as tmp:

        sandbox = _sandbox_settings(Path(tmp))
        writer = ParquetWriter(sandbox)
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        path = Path(tmp) / "probe.parquet"
        writer.write(df, path, run_id="health-probe", dataset_name="probe")
        back = read_parquet(path)
        if len(back) != 2 or list(back["a"]) != [1, 2]:
            raise AsxQuantError(f"Round-trip mismatch: {back.shape} / {back['a'].tolist()}")
        audit_ok = {"ingestion_run_id", "retrieved_at"}.issubset(back.columns)
        if not audit_ok:
            raise AsxQuantError("Audit columns missing after round-trip")
        return "2 rows; audit columns stamped"


def _check_run_registry() -> str:
    with tempfile.TemporaryDirectory() as tmp:
        lake = DataLake(root=Path(tmp))
        registry = RunRegistry.start(lake, config_version="v0.1")
        record = registry.begin_stage("probe_stage")
        registry.finish_stage(record, record_count=7, detail="ok")
        entries = registry.read_log()
        if len(entries) != 1 or entries[0]["stage"] != "probe_stage":
            raise AsxQuantError(f"Run registry log unexpected: {entries}")
        latest = registry.read_latest()
        if latest is None or latest["run_id"] != registry.run_id:
            raise AsxQuantError("latest_run.json missing or stale")
        return f"run {registry.run_id}; stage recorded"


def _sandbox_settings(root: Path):
    """Minimal settings stand-in exposing only what ParquetWriter needs."""
    from types import SimpleNamespace

    return SimpleNamespace(
        parquet_compression="zstd",
        parquet_compression_level=1,
    )
