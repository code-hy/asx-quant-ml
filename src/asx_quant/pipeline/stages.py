"""Pipeline stages: ingest, validate, transform (Phase 2).

Each stage is restartable and records itself in the run registry. Stage
functions are the only place where registry, lake, config and quality
modules are composed together.
"""

from __future__ import annotations

import json

import pandas as pd

from asx_quant.config.loader import Settings, load_yaml
from asx_quant.exceptions import IngestionError
from asx_quant.ingestion.universe import (
    add_master_metadata,
    build_universe,
    write_bronze_security_master,
)
from asx_quant.ingestion.yahoo import (
    YahooDownloader,
    append_ingestion_log,
    history_start,
    normalise_raw_frame,
)
from asx_quant.observability.logging import get_logger
from asx_quant.observability.runs import RunRegistry, StageRecord
from asx_quant.quality.checks import QualityReport, run_price_quality_checks
from asx_quant.storage.parquet import ParquetWriter, read_parquet
from asx_quant.storage.paths import DataLake, batch_id
from asx_quant.transformations.calendar import (
    build_market_calendar,
    compute_coverage,
)
from asx_quant.transformations.prices import (
    load_bronze_prices,
    standardise_bronze_prices,
    transform_prices_to_silver,
)

logger = get_logger("pipeline.stages")

INGESTION_LOG_NAME = "ingestion_log.jsonl"


def _make_registry(settings: Settings, lake: DataLake) -> RunRegistry:
    return RunRegistry.start(
        lake, config_version=f"v{settings.project_version}"
    )


def _benchmark_tickers() -> list[str]:
    benchmarks = load_yaml("universe")["benchmarks"]
    tickers = []
    for spec in benchmarks.values():
        if isinstance(spec, dict) and "source_ticker" in spec:
            tickers.append(spec["source_ticker"])
    return tickers


# ------------------------------------------------------------------- ingest
def run_ingest(settings: Settings, mode: str = "historical") -> dict:
    """Stage 1: ingest universe + historical prices into Bronze."""
    if mode != "historical":
        raise IngestionError(
            f"Mode '{mode}' is implemented from Phase 9; use --mode historical."
        )
    lake = DataLake.from_settings(settings)
    lake.ensure_all()
    registry = _make_registry(settings, lake)
    record: StageRecord | None = None
    log_records: list[dict[str, object]] = []

    try:
        # -- universe ----------------------------------------------------
        record = registry.begin_stage("ingest_universe")
        universe = build_universe(settings)
        master_bronze = write_bronze_security_master(
            lake, registry, settings, add_master_metadata(universe, settings)
        )
        tickers = sorted(universe["ticker"].tolist())
        registry.finish_stage(
            record, record_count=len(tickers),
            master_path=str(master_bronze), source=str(universe["universe_source"].iloc[0]),
        )
        log_records.append({
            "stage": "ingest_universe", "run_id": registry.run_id,
            "batch_id": batch_id(), "source": "wikipedia_asx200_or_seed",
            "record_count": len(tickers), "status": "success",
        })

        # -- prices ------------------------------------------------------
        record = registry.begin_stage("ingest_prices")
        all_tickers = sorted(set(tickers + _benchmark_tickers()))
        downloader = YahooDownloader(
            batch_size=int(settings.raw["ingestion"]["market_data"]["batch_size"]),
            batch_pause_seconds=float(
                settings.raw["ingestion"]["market_data"]["batch_pause_seconds"]),
            max_retries=int(settings.raw["ingestion"]["market_data"]["max_retries"]),
            retry_backoff_seconds=float(
                settings.raw["ingestion"]["market_data"]["retry_backoff_seconds"]),
        )
        frames, failed = downloader.download(all_tickers, history_start(settings))

        writer = ParquetWriter(settings)
        batch = registry.run_id  # unique per run; timestamps keep it sortable
        written: dict[str, int] = {}
        for ticker, frame in frames.items():
            if frame.empty:
                continue
            normalised = normalise_raw_frame(ticker, frame)
            if normalised.empty:
                continue
            path = lake.bronze_batch_path("prices", ticker, batch)
            writer.write(normalised, path, run_id=registry.run_id,
                         dataset_name=f"bronze_prices_{ticker}")
            written[ticker] = len(normalised)
            log_records.append({
                "stage": "ingest_prices", "run_id": registry.run_id,
                "batch_id": batch, "source": "yfinance",
                "source_ticker": ticker, "record_count": len(normalised),
                "status": "success",
                "date_range": [
                    str(normalised["date"].min().date()),
                    str(normalised["date"].max().date()),
                ] if not normalised.empty else None,
            })

        if failed:
            log_records.append({
                "stage": "ingest_prices", "run_id": registry.run_id,
                "batch_id": batch, "source": "yfinance",
                "record_count": 0, "status": "failed",
                "error": f"No data for: {', '.join(failed)}",
            })
        append_ingestion_log(
            lake.bronze_domain_dir("prices") / INGESTION_LOG_NAME, log_records
        )
        registry.finish_stage(
            record, record_count=sum(written.values()),
            tickers_written=len(written), failed_tickers=failed,
        )

        summary = {
            "run_id": registry.run_id,
            "universe_source": str(universe["universe_source"].iloc[0]),
            "universe_size": len(tickers),
            "tickers_written": len(written),
            "records_written": sum(written.values()),
            "failed_tickers": failed,
        }
        registry.log_dir.parent.joinpath("ingestion_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary
    except Exception as exc:
        if record is not None:
            registry.finish_stage(record, status="failed", error=str(exc))
        raise


# ----------------------------------------------------------------- validate
def run_validate(settings: Settings) -> QualityReport:
    """Stage 2: run data-quality checks on the Bronze price data."""
    lake = DataLake.from_settings(settings)
    registry = _make_registry(settings, lake)
    record = registry.begin_stage("validate")
    try:
        frames = load_bronze_prices(lake)
        standardised = standardise_bronze_prices(frames)
        report = run_price_quality_checks(
            standardised,
            dataset_name="bronze_prices",
            quality_cfg=settings.raw["quality"],
        )
        report_path = report.save(lake.report_path("data_quality", ".json"))
        status = report.to_dict()["status"]
        registry.finish_stage(
            record,
            status="success" if report.passed else "failed",
            record_count=len(standardised),
            error=None if report.passed else
            ", ".join(r.name for r in report.errors if not r.passed),
            report_path=str(report_path),
        )
        logger.info(
            "Validation complete: status=%s, %d warnings",
            status, report.warn_count(),
        )
        return report
    except Exception as exc:
        registry.finish_stage(record, status="failed", error=str(exc))
        raise


# ---------------------------------------------------------------- transform
def run_transform(settings: Settings) -> dict:
    """Stage 3: Bronze → Silver (prices, master, calendar, coverage)."""
    lake = DataLake.from_settings(settings)
    registry = _make_registry(settings, lake)
    record = registry.begin_stage("transform")
    try:
        prices_path, quality = transform_prices_to_silver(lake, settings, registry)
        silver = read_parquet(prices_path)

        calendar = build_market_calendar(silver)
        calendar_path = lake.silver_dataset_path("market_calendar")
        ParquetWriter(settings).write(
            calendar, calendar_path, run_id=registry.run_id,
            overwrite=True, dataset_name="market_calendar",
        )

        master = _latest_security_master(lake)
        coverage = compute_coverage(silver, calendar, master)
        coverage_path = lake.silver_dataset_path("data_coverage")
        ParquetWriter(settings).write(
            coverage, coverage_path, run_id=registry.run_id,
            overwrite=True, dataset_name="data_coverage",
        )
        _write_coverage_json(lake, registry, coverage, quality)

        silver_master = _build_silver_master(coverage, master)
        master_path = lake.silver_dataset_path("security_master")
        ParquetWriter(settings).write(
            silver_master, master_path, run_id=registry.run_id,
            overwrite=True, dataset_name="security_master",
        )

        _register_warehouse_views(settings, lake)

        summary = {
            "run_id": registry.run_id,
            "records": len(silver),
            "securities": int(
                silver.loc[~silver["ticker"].str.startswith("IDX_"),
                           "ticker"].nunique()
            ),
            "calendar_sessions": len(calendar),
            "first_date": str(silver["date"].min().date()),
            "last_date": str(silver["date"].max().date()),
            "quality_status": quality.to_dict()["status"] if quality else "skipped",
        }
        registry.finish_stage(record, record_count=len(silver), **{
            k: v for k, v in summary.items() if k != "run_id"
        })
        return summary
    except Exception as exc:
        registry.finish_stage(record, status="failed", error=str(exc))
        raise


def _latest_security_master(lake: DataLake) -> pd.DataFrame:
    batches = lake.bronze_batches("security_master", "asx200")
    if not batches:
        raise IngestionError(
            "No bronze security master batch found. Run ingest first."
        )
    return read_parquet(batches[-1])


def _build_silver_master(
    coverage: pd.DataFrame,
    bronze_master: pd.DataFrame,
) -> pd.DataFrame:
    master = bronze_master.copy()
    cov = coverage[["ticker", "first_observed_date", "last_observed_date",
                    "n_sessions", "coverage_status"]]
    master = master.merge(cov, left_on="ticker", right_on="ticker", how="left")
    master["data_quality_status"] = master["coverage_status"].fillna("no_price_data")
    return master


def _write_coverage_json(
    lake: DataLake,
    registry: RunRegistry,
    coverage: pd.DataFrame,
    quality: QualityReport | None,
) -> None:
    benchmark_coverage = coverage[coverage["ticker"].str.startswith(("IDX_",))]
    universe_coverage = coverage[~coverage["ticker"].str.startswith(("IDX_",))]
    usable = universe_coverage[universe_coverage["n_sessions"] >= 250]
    payload = {
        "run_id": registry.run_id,
        "total_securities": len(universe_coverage),
        "securities_with_usable_history": len(usable),
        "benchmarks": len(benchmark_coverage),
        "earliest_date": str(coverage["first_observed_date"].min()),
        "latest_date": str(coverage["last_observed_date"].max()),
        "total_unique_security_session_records": int(coverage["n_sessions"].sum()),
        "securities_with_missing_sessions": int(
            (coverage["missing_sessions"] > 0).sum()
        ),
        "quality_status": quality.to_dict()["status"] if quality else "skipped",
        "per_security_top10_missing": coverage.sort_values(
            "missing_sessions", ascending=False
        ).head(10)[["ticker", "missing_sessions", "n_sessions"]]
        .to_dict(orient="records"),
        "unusable_or_failed": coverage[coverage["n_sessions"] < 250][
            ["ticker", "n_sessions", "coverage_status"]
        ].to_dict(orient="records"),
    }
    lake.report_path("data_coverage", ".json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )


def _register_warehouse_views(settings: Settings, lake: DataLake) -> None:
    from asx_quant.storage.duckdb import get_warehouse

    warehouse = get_warehouse(settings)
    try:
        warehouse.connect()
        warehouse.register_view(
            "silver", "equity_prices",
            lake.silver_dataset_path("equity_prices"), layer="silver",
        )
        warehouse.register_view(
            "silver", "security_master",
            lake.silver_dataset_path("security_master"), layer="silver",
        )
        warehouse.register_view(
            "silver", "market_calendar",
            lake.silver_dataset_path("market_calendar"), layer="silver",
        )
        warehouse.register_view(
            "silver", "data_coverage",
            lake.silver_dataset_path("data_coverage"), layer="silver",
        )
    finally:
        warehouse.close()
