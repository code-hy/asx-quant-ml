"""Re-ingest ASIC short-position reports only (no price re-download).

Creates a fresh run ID, downloads the report index + one YTD file per
configured year, and writes new bronze batches (bronze is append-only, so
previous batches remain intact). Skipped years are logged as failures and
never fabricated.

Usage::

    uv run python scripts/ingest_asic_only.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asx_quant.config.loader import load_settings  # noqa: E402
from asx_quant.observability.logging import configure_logging, get_logger  # noqa: E402
from asx_quant.observability.runs import RunRegistry  # noqa: E402
from asx_quant.pipeline.stages import ingest_asic_bronze  # noqa: E402
from asx_quant.storage.paths import DataLake  # noqa: E402

logger = get_logger("scripts.ingest_asic_only")


def main() -> int:
    configure_logging()
    settings = load_settings()
    lake = DataLake.from_settings(settings)
    registry = RunRegistry.start(
        lake, config_version=f"v{settings.project_version}"
    )
    record = registry.begin_stage("ingest_asic_only")
    log_records: list[dict[str, object]] = []
    try:
        summary = ingest_asic_bronze(lake, settings, registry, log_records)
        registry.finish_stage(
            record,
            status="success" if summary["status"] == "ok" else "partial",
            record_count=int(summary["records"]),
            securities=int(summary["securities"]),  # type: ignore[arg-type]
            years=str(summary["years"]),
            failed_reports=str(summary["failed_reports"]),
        )
        print(f"ASIC-only ingestion complete (run {registry.run_id}):")
        print(f"  status:            {summary['status']}")
        print(f"  records:           {summary['records']:,}")
        print(f"  securities:        {summary['securities']}")
        print(f"  years ingested:    {summary['years']}")
        if summary["failed_reports"]:
            print(f"  FAILED reports:    {summary['failed_reports']}")
        return 0
    except Exception as exc:  # noqa: BLE001
        registry.finish_stage(record, status="failed", error=str(exc))
        print(f"ASIC-only ingestion FAILED: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
