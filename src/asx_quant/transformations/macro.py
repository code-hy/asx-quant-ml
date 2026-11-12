"""Bronze -> Silver transformations for macro and short-position data.

Conformance rules:

* ``macro_observations`` — one row per (series_name, observation_date) with
  the point-in-time publication columns carried through from Bronze. The
  newest batch wins on duplicates (provider revisions).
* ``short_positions`` — ASIC codes are mapped to the standard ``.AX`` ticker
  format; the newest batch wins on (ticker, trade_date).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from asx_quant.exceptions import StorageError
from asx_quant.observability.logging import get_logger
from asx_quant.quality.checks import (
    QualityReport,
    run_macro_quality_checks,
    run_short_position_quality_checks,
)
from asx_quant.storage.parquet import ParquetWriter, read_parquet

logger = get_logger("transformations.macro")

MACRO_OBSERVATION_COLUMNS = (
    "series_name", "source_series_id", "observation_date", "value",
    "publication_date", "publication_date_method", "potentially_revised",
    "frequency", "unit", "source", "source_table",
    "ingestion_run_id", "retrieved_at",
)

SHORT_POSITION_COLUMNS = (
    "ticker", "code", "product_name", "trade_date",
    "reported_short_positions", "pct_of_product_in_issue",
    "report_date", "publication_date", "publication_date_method",
    "ingestion_run_id", "retrieved_at",
)


def _load_bronze_domain(
    lake,  # noqa: ANN001
    domain: str,
    tick_filter=None,
) -> dict[str, pd.DataFrame]:
    """Load every bronze batch in a domain, tagged with its batch id.

    ``tick_filter`` optionally restricts which tick directories are loaded
    (e.g. the ASIC domain mixes per-year report batches with the report
    index batch; only the former are position data).
    """
    base = lake.bronze_domain_dir(domain)
    if not base.is_dir():
        raise StorageError(
            f"No bronze '{domain}' domain directory at '{base}'. Run ingest first."
        )
    frames: dict[str, pd.DataFrame] = {}
    for tick_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        if tick_filter is not None and not tick_filter(tick_dir.name):
            continue
        batches = sorted(tick_dir.glob("*.parquet"))
        if not batches:
            continue
        parts = []
        for batch_file in batches:
            frame = read_parquet(batch_file)
            frame["batch_id"] = batch_file.stem
            parts.append(frame)
        frames[tick_dir.name] = pd.concat(parts, ignore_index=True)
    if not frames:
        raise StorageError(
            f"No bronze batches found under '{base}'. Run ingest first."
        )
    return frames


# -------------------------------------------------------------------- macro
def transform_macro_to_silver(
    lake,  # noqa: ANN001
    settings,  # noqa: ANN001
    registry,  # noqa: ANN001
    with_quality: bool = True,
) -> tuple[Path, QualityReport | None]:
    """Build ``silver/macro_observations.parquet`` from bronze RBA batches."""
    frames = _load_bronze_domain(lake, "rba")
    parts = []
    for _tick, frame in frames.items():
        if frame.empty:
            continue
        parts.append(frame)
    if not parts:
        raise StorageError("No RBA observations could be loaded from bronze.")
    macro = pd.concat(parts, ignore_index=True)
    macro["observation_date"] = pd.to_datetime(macro["observation_date"])
    macro["publication_date"] = pd.to_datetime(macro["publication_date"])
    macro["value"] = pd.to_numeric(macro["value"], errors="coerce")
    macro["potentially_revised"] = macro["potentially_revised"].astype(bool)
    macro = macro.sort_values(["series_name", "observation_date", "batch_id"])
    macro = macro.drop_duplicates(
        subset=["series_name", "observation_date"], keep="last"
    )

    report = run_macro_quality_checks(macro) if with_quality else None
    if report is not None:
        # Save the report BEFORE asserting so a failure is always persisted.
        report.save(lake.report_path("data_quality_macro", ".json"))
        report.assert_passed()
        logger.info("Macro quality: %s (%d warnings)",
                    report.to_dict()["status"], report.warn_count())

    out = macro[[c for c in MACRO_OBSERVATION_COLUMNS if c in macro.columns]]
    for col in ("ingestion_run_id", "retrieved_at"):
        if col not in out.columns:
            out[col] = None
    out = out[list(MACRO_OBSERVATION_COLUMNS)]
    path = lake.silver_dataset_path("macro_observations")
    ParquetWriter(settings).write(
        out, path, run_id=registry.run_id, overwrite=True,
        dataset_name="macro_observations",
    )
    return path, report


# ----------------------------------------------------------- short positions
def transform_short_positions_to_silver(
    lake,  # noqa: ANN001
    settings,  # noqa: ANN001
    registry,  # noqa: ANN001
    with_quality: bool = True,
) -> tuple[Path, QualityReport | None]:
    """Build ``silver/short_positions.parquet`` from bronze ASIC batches."""
    frames = _load_bronze_domain(
        lake, "asic", tick_filter=lambda name: name.startswith("ytd_")
    )
    parts = [f for f in frames.values() if not f.empty]
    if not parts:
        raise StorageError("No ASIC positions could be loaded from bronze.")
    positions = pd.concat(parts, ignore_index=True)

    positions["ticker"] = positions["code"].astype(str) + ".AX"
    positions["trade_date"] = pd.to_datetime(positions["trade_date"])
    positions["publication_date"] = pd.to_datetime(positions["publication_date"])
    positions["report_date"] = pd.to_datetime(positions["report_date"])
    positions = positions.sort_values(["ticker", "trade_date", "batch_id"])
    positions = positions.drop_duplicates(
        subset=["ticker", "trade_date"], keep="last"
    )

    report = run_short_position_quality_checks(positions) if with_quality else None
    if report is not None:
        # Save the report BEFORE asserting so a failure is always persisted.
        report.save(lake.report_path("data_quality_short_positions", ".json"))
        report.assert_passed()
        logger.info("Short-position quality: %s (%d warnings)",
                    report.to_dict()["status"], report.warn_count())

    out = positions[[c for c in SHORT_POSITION_COLUMNS if c in positions.columns]]
    for col in ("ingestion_run_id", "retrieved_at"):
        if col not in out.columns:
            out[col] = None
    out = out[list(SHORT_POSITION_COLUMNS)]
    path = lake.silver_dataset_path("short_positions")
    ParquetWriter(settings).write(
        out, path, run_id=registry.run_id, overwrite=True,
        dataset_name="short_positions",
    )
    return path, report
