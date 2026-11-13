"""Pipeline stages: ingest, validate, transform (Phase 2).

Each stage is restartable and records itself in the run registry. Stage
functions are the only place where registry, lake, config and quality
modules are composed together.
"""

from __future__ import annotations

import json

import pandas as pd

from asx_quant.config.loader import Settings, load_yaml
from asx_quant.exceptions import IngestionError, StorageError
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
from asx_quant.quality.checks import (
    QualityReport,
    run_macro_quality_checks,
    run_price_quality_checks,
    run_short_position_quality_checks,
)
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

        # -- macro (RBA tables incl. ABS-via-RBA + cash rate) -------------
        macro_summary: dict[str, object] = {"status": "disabled"}
        if settings.raw["ingestion"]["macro"].get("rba_enabled", False):
            record = registry.begin_stage("ingest_macro_rba")
            try:
                macro_summary = ingest_rba_bronze(lake, settings, registry,
                                                  log_records)
                registry.finish_stage(
                    record, record_count=int(macro_summary["records"]),  # type: ignore[arg-type]
                    **{k: v for k, v in macro_summary.items() if k != "status"},
                )
            except Exception as exc:  # noqa: BLE001 - optional source
                logger.error("RBA macro ingestion failed (prices are "
                             "unaffected): %s", exc)
                macro_summary = {"status": "failed", "error": str(exc)}
                registry.finish_stage(record, status="failed", error=str(exc))
                record = None

        # -- ASIC aggregated short positions ------------------------------
        asic_summary: dict[str, object] = {"status": "disabled"}
        if settings.raw["ingestion"]["asic_short_positions"].get("enabled",
                                                                 False):
            record = registry.begin_stage("ingest_asic")
            try:
                asic_summary = ingest_asic_bronze(lake, settings, registry,
                                                  log_records)
                registry.finish_stage(
                    record, record_count=int(asic_summary["records"]),  # type: ignore[arg-type]
                    **{k: v for k, v in asic_summary.items() if k != "status"},
                )
            except Exception as exc:  # noqa: BLE001 - optional source
                logger.error("ASIC ingestion failed (prices are unaffected): %s",
                             exc)
                asic_summary = {"status": "failed", "error": str(exc)}
                registry.finish_stage(record, status="failed", error=str(exc))
                record = None

        summary = {
            "run_id": registry.run_id,
            "universe_source": str(universe["universe_source"].iloc[0]),
            "universe_size": len(tickers),
            "tickers_written": len(written),
            "records_written": sum(written.values()),
            "failed_tickers": failed,
            "macro": macro_summary,
            "asic": asic_summary,
        }
        registry.log_dir.parent.joinpath("ingestion_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary
    except Exception as exc:
        if record is not None:
            registry.finish_stage(record, status="failed", error=str(exc))
        raise


def ingest_rba_bronze(
    lake: DataLake,
    settings: Settings,
    registry: RunRegistry,
    log_records: list[dict[str, object]],
) -> dict[str, object]:
    """Ingest RBA statistical tables + cash rate into Bronze batches."""
    from asx_quant.ingestion.rba import build_macro_observations

    macro = build_macro_observations(settings)
    writer = ParquetWriter(settings)
    batch = registry.run_id
    tables: list[str] = []
    for table_key, group in macro.groupby("source_table"):
        path = lake.bronze_batch_path("rba", str(table_key), batch)
        writer.write(group, path, run_id=registry.run_id,
                     dataset_name=f"bronze_rba_{table_key}")
        tables.append(str(table_key))
        log_records.append({
            "stage": "ingest_macro_rba", "run_id": registry.run_id,
            "batch_id": batch, "source": "rba",
            "source_table": table_key, "record_count": len(group),
            "status": "success",
            "series": sorted(group["series_name"].unique().tolist()),
            "date_range": [
                str(group["observation_date"].min().date()),
                str(group["observation_date"].max().date()),
            ],
        })
    return {
        "status": "ok", "records": len(macro),
        "series": int(macro["series_name"].nunique()), "tables": tables,
    }


def ingest_asic_bronze(
    lake: DataLake,
    settings: Settings,
    registry: RunRegistry,
    log_records: list[dict[str, object]],
) -> dict[str, object]:
    """Ingest ASIC aggregated short positions into Bronze batches."""
    from asx_quant.ingestion.asic import (
        default_asic_http_get,
        ingest_asic_reports,
        write_ingestion_log,
    )

    index, positions, asic_log = ingest_asic_reports(
        settings, default_asic_http_get()
    )
    writer = ParquetWriter(settings)
    batch = registry.run_id
    idx_path = lake.bronze_batch_path("asic", "report_index", batch)
    writer.write(index, idx_path, run_id=registry.run_id,
                 dataset_name="bronze_asic_report_index")
    years: list[int] = []
    for year, group in positions.groupby("report_year"):
        path = lake.bronze_batch_path("asic", f"ytd_{int(year)}", batch)
        writer.write(group, path, run_id=registry.run_id,
                     dataset_name=f"bronze_asic_ytd_{int(year)}")
        years.append(int(year))
    write_ingestion_log(
        lake.bronze_domain_dir("asic") / INGESTION_LOG_NAME, asic_log
    )
    for record in asic_log:
        log_records.append({"run_id": registry.run_id, **record})
    failed_years = [r["source_report"] for r in asic_log if r.get("status") != "success"]
    return {
        "status": "ok" if not failed_years else "partial",
        "records": len(positions),
        "securities": int(positions["code"].nunique()),
        "years": years,
        "failed_reports": failed_years,
    }


# ----------------------------------------------------------------- validate
def run_validate(settings: Settings) -> QualityReport:
    """Stage 2: run data-quality checks on all Bronze data.

    Prices are always validated; macro and short-position checks run when
    their bronze data exists (older builds may predate Phase 3).
    """
    lake = DataLake.from_settings(settings)
    registry = _make_registry(settings, lake)
    record = registry.begin_stage("validate")
    try:
        combined = QualityReport(dataset="bronze_all")
        frames = load_bronze_prices(lake)
        standardised = standardise_bronze_prices(frames)
        price_report = run_price_quality_checks(
            standardised,
            dataset_name="bronze_prices",
            quality_cfg=settings.raw["quality"],
        )
        combined.results.extend(price_report.results)
        record_count = len(standardised)

        from asx_quant.transformations.macro import _load_bronze_domain

        try:
            bronze_rba = _load_bronze_domain(lake, "rba")
            macro = pd.concat([f for f in bronze_rba.values() if not f.empty],
                              ignore_index=True)
            macro["observation_date"] = pd.to_datetime(macro["observation_date"])
            macro["publication_date"] = pd.to_datetime(macro["publication_date"])
            macro["value"] = pd.to_numeric(macro["value"], errors="coerce")
            macro_report = run_macro_quality_checks(
                macro, dataset_name="macro_observations")
            combined.results.extend(macro_report.results)
            record_count += len(macro)
        except StorageError:
            logger.info("No bronze RBA data present; macro validation skipped.")

        try:
            bronze_asic = _load_bronze_domain(
                lake, "asic", tick_filter=lambda name: name.startswith("ytd_"))
            positions = pd.concat(
                [f for f in bronze_asic.values() if not f.empty],
                ignore_index=True)
            for col in ("trade_date", "publication_date"):
                positions[col] = pd.to_datetime(positions[col])
            positions["ticker"] = positions["code"].astype(str) + ".AX"
            positions_report = run_short_position_quality_checks(
                positions, dataset_name="short_positions")
            combined.results.extend(positions_report.results)
            record_count += len(positions)
        except StorageError:
            logger.info("No bronze ASIC data present; short-position "
                        "validation skipped.")

        report = combined
        report_path = report.save(lake.report_path("data_quality", ".json"))
        status = report.to_dict()["status"]
        registry.finish_stage(
            record,
            status="success" if report.passed else "failed",
            record_count=record_count,
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

        # -- macro + short positions (Phase 3; skipped if bronze absent) --
        macro_status = "skipped"
        short_status = "skipped"
        macro_detail: dict[str, object] = {}
        try:
            from asx_quant.transformations.macro import (
                transform_macro_to_silver,
                transform_short_positions_to_silver,
            )

            macro_path, macro_quality = transform_macro_to_silver(
                lake, settings, registry)
            macro_silver = read_parquet(macro_path)
            macro_status = (
                macro_quality.to_dict()["status"] if macro_quality else "ok"
            )
            macro_detail = {
                "records": len(macro_silver),
                "series": int(macro_silver["series_name"].nunique()),
                "series_names": sorted(
                    macro_silver["series_name"].unique().tolist()),
                "first_publication_date": str(
                    macro_silver["publication_date"].min().date()),
                "quality_status": macro_status,
            }
            logger.info("Macro silver: %s", macro_detail)
        except StorageError as exc:
            logger.info("Macro silver skipped: %s", exc)

        try:
            from asx_quant.transformations.macro import (
                transform_short_positions_to_silver,
            )

            short_path, short_quality = transform_short_positions_to_silver(
                lake, settings, registry)
            short_silver = read_parquet(short_path)
            short_status = (
                short_quality.to_dict()["status"] if short_quality else "ok"
            )
            short_detail = {
                "records": len(short_silver),
                "securities": int(short_silver["ticker"].nunique()),
                "first_trade_date": str(short_silver["trade_date"].min().date()),
                "last_trade_date": str(short_silver["trade_date"].max().date()),
                "quality_status": short_status,
            }
            logger.info("Short-position silver: %s", short_detail)
        except StorageError as exc:
            logger.info("Short-position silver skipped: %s", exc)

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
            "macro_status": macro_status,
            "short_positions_status": short_status,
        }
        registry.finish_stage(record, record_count=len(silver), **{
            k: v for k, v in summary.items() if k != "run_id"
        })
        return summary
    except Exception as exc:
        registry.finish_stage(record, status="failed", error=str(exc))
        raise


# ------------------------------------------------------------------ features
def run_features(settings: Settings) -> dict:
    """Stage 4: Silver → Gold feature set + target labels (Phase 4)."""
    from asx_quant.features.build import build_feature_set

    lake = DataLake.from_settings(settings)
    registry = _make_registry(settings, lake)
    record = registry.begin_stage("features")
    try:
        result = build_feature_set(lake, settings, registry)
        features = read_parquet(result.features_path)
        targets = read_parquet(result.targets_path)
        summary = {
            "run_id": registry.run_id,
            "feature_set_version": result.feature_set_version,
            "n_features": result.n_features,
            "n_rows": result.n_rows,
            "securities": int(features["ticker"].nunique()),
            "first_date": str(features["date"].min().date()),
            "last_date": str(features["date"].max().date()),
            "quality_status": (result.report.to_dict()["status"]
                               if result.report else "skipped"),
            "warn_failures": (result.report.warn_count()
                              if result.report else 0),
            "label_horizons": "3d/10d",
            "embargo_sessions": 3,
            "labelled_rows": int(targets["future_return_3d"].notna().sum()),
        }
        registry.finish_stage(record, record_count=result.n_rows, **{
            k: v for k, v in summary.items() if k != "run_id"
        })
        return summary
    except Exception as exc:
        registry.finish_stage(record, status="failed", error=str(exc))
        raise


# --------------------------------------------------------------------- train
def run_train(settings: Settings) -> dict:
    """Stage 5: Gold features -> walk-forward trained models (Phase 5)."""
    from asx_quant.models.build import train_models

    lake = DataLake.from_settings(settings)
    registry = _make_registry(settings, lake)
    record = registry.begin_stage("train")
    try:
        result = train_models(lake, settings, registry)
        predictions = read_parquet(result.predictions_path)
        eval_auc = {
            name: round(metrics.get("roc_auc", float("nan")), 4)
            for name, metrics in result.evaluation_metrics.items()
            if "roc_auc" in metrics
        }
        eval_ic = {
            name: round(metrics.get("spearman", float("nan")), 4)
            for name, metrics in result.evaluation_metrics.items()
            if "spearman" in metrics
        }
        summary = {
            "run_id": registry.run_id,
            "model_version": result.model_version,
            "n_folds": result.n_folds,
            "n_models": result.n_models,
            "n_prediction_rows": len(predictions),
            "evaluation_auc": eval_auc,
            "evaluation_spearman": eval_ic,
            "quality_status": (result.report.to_dict()["status"]
                               if result.report else "skipped"),
            "warn_failures": (result.report.warn_count()
                              if result.report else 0),
        }
        registry.finish_stage(record, record_count=len(predictions), **{
            k: v for k, v in summary.items() if k != "run_id"
        })
        return summary
    except Exception as exc:
        registry.finish_stage(record, status="failed", error=str(exc))
        raise


# ----------------------------------------------------------------- portfolio
def run_portfolio(settings: Settings) -> dict:
    """Stage 6: OOS predictions -> long-only target portfolios (Phase 6)."""
    from asx_quant.portfolio.build import build_portfolios

    lake = DataLake.from_settings(settings)
    registry = _make_registry(settings, lake)
    record = registry.begin_stage("portfolio")
    try:
        result = build_portfolios(lake, settings, registry)
        portfolios = read_parquet(result.portfolios_path)
        summary = {
            "run_id": registry.run_id,
            "n_rebalances": result.n_rebalances,
            "n_weight_rows": len(portfolios),
            "avg_turnover": result.avg_turnover,
            "quality_status": (result.report.to_dict()["status"]
                               if result.report else "skipped"),
            "warn_failures": (result.report.warn_count()
                              if result.report else 0),
        }
        registry.finish_stage(record, record_count=len(portfolios), **{
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
        for name in ("equity_prices", "security_master", "market_calendar",
                     "data_coverage", "macro_observations", "short_positions"):
            path = lake.silver_dataset_path(name)
            if path.exists():
                warehouse.register_view(
                    "silver", name, path, layer="silver",
                )
    finally:
        warehouse.close()
