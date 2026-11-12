"""Bronze → Silver price transformation and data-coverage reporting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from asx_quant.exceptions import StorageError
from asx_quant.observability.logging import get_logger
from asx_quant.quality.checks import QualityReport, run_price_quality_checks
from asx_quant.storage.parquet import ParquetWriter, read_parquet

logger = get_logger("transformations.prices")

SILVER_PRICE_COLUMNS = (
    "ticker", "date", "date_key", "open_price", "high_price", "low_price",
    "close_price", "adjusted_close", "volume", "dividend_amount",
    "split_factor", "daily_return", "source", "ingestion_run_id",
    "retrieved_at",
)


def load_bronze_prices(lake) -> dict[str, pd.DataFrame]:  # noqa: ANN001
    """Load ALL bronze price batches per ticker (oldest → newest batch order).

    Each row gains a ``batch_id`` column (from the filename) so the caller can
    deduplicate keeping the NEWEST batch — which implements provider-revision
    reconciliation for free.
    """
    frames: dict[str, pd.DataFrame] = {}
    domain = lake.bronze_domain_dir("prices")
    if not domain.is_dir():
        raise StorageError(
            f"No bronze price domain directory at '{domain}'. Run ingest first."
        )
    for ticker_dir in sorted(domain.iterdir()):
        if not ticker_dir.is_dir():
            continue
        batches = sorted(ticker_dir.glob("*.parquet"))
        if not batches:
            continue
        parts = []
        for batch_file in batches:
            frame = read_parquet(batch_file)
            frame["batch_id"] = batch_file.stem
            parts.append(frame)
        frames[ticker_dir.name] = pd.concat(parts, ignore_index=True)
    if not frames:
        raise StorageError(
            f"No bronze price batches found under '{domain}'. Run ingest first."
        )
    return frames


def standardise_bronze_prices(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Conform every ticker's bronze batches and reconcile duplicates.

    Duplicate (ticker, date) rows are resolved by keeping the record from the
    NEWEST batch (highest batch_id), so re-ingesting recent history silently
    replaces any revised values — and never the other way round.
    """
    parts: list[pd.DataFrame] = []
    for ticker, frame in frames.items():
        if frame.empty:
            continue
        out = frame.copy()
        if "source_ticker" not in out.columns:
            out.insert(0, "source_ticker", ticker)
        parts.append(out)
    if not parts:
        raise StorageError("No price records could be standardised from bronze.")

    prices = pd.concat(parts, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    for col in ("open", "high", "low", "close", "adj_close", "volume",
                "dividends", "split_factor"):
        prices[col] = pd.to_numeric(prices[col], errors="coerce")
    prices = prices.sort_values(["source_ticker", "date", "batch_id"])
    prices = prices.drop_duplicates(subset=["source_ticker", "date"], keep="last")
    prices = prices.rename(columns={"source_ticker": "ticker"})
    return prices.reset_index(drop=True)


def standard_ticker(source_ticker: str) -> str:
    """Map a source ticker to the standard Silver ticker format.

    Index tickers with Yahoo's ``^`` prefix become ``IDX_<name>`` (e.g.
    ``^AXJO`` → ``IDX_AXJO``); equity tickers pass through unchanged.
    """
    if source_ticker.startswith("^"):
        return "IDX_" + source_ticker[1:]
    return source_ticker


def to_silver_schema(prices: pd.DataFrame) -> pd.DataFrame:
    """Project standardised prices onto the documented Silver schema."""
    out = prices.rename(columns={"source_ticker": "ticker"})
    out["ticker"] = out["ticker"].map(standard_ticker)
    out = out.rename(columns={
        "open": "open_price",
        "high": "high_price",
        "low": "low_price",
        "close": "close_price",
        "adj_close": "adjusted_close",
        "dividends": "dividend_amount",
        "split_factor": "split_factor",
    })
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["date_key"] = out["date"].dt.strftime("%Y%m%d").astype(int)
    # Daily return on ADJUSTED close — the documented convention shared with
    # target labels (dividends accrue to the holder).
    out = out.sort_values(["ticker", "date"])
    out["daily_return"] = out.groupby("ticker")["adjusted_close"].pct_change()
    out["split_factor"] = out["split_factor"].fillna(1.0)
    out["dividend_amount"] = out["dividend_amount"].fillna(0.0)
    # Audit columns are stamped by ParquetWriter at write time when absent.
    for col in ("ingestion_run_id", "retrieved_at"):
        if col not in out.columns:
            out[col] = None
    return out[list(SILVER_PRICE_COLUMNS)]


def transform_prices_to_silver(
    lake,  # noqa: ANN001
    settings,  # noqa: ANN001
    registry,  # noqa: ANN001
    quality_report: bool = True,
) -> tuple[Path, QualityReport | None]:
    """Build ``silver/equity_prices.parquet`` from all bronze batches.

    Returns:
        ``(silver_path, quality_report)`` — the report is ``None`` when
        quality checks are disabled.
    """
    frames = load_bronze_prices(lake)
    standardised = standardise_bronze_prices(frames)
    report = run_price_quality_checks(
        standardised,
        quality_cfg=settings.raw["quality"],
    ) if quality_report else None
    if report is not None:
        report.assert_passed()
        report.save(lake.report_path("data_quality", ".json"))
        logger.info(
            "Quality report: %d rules, status %s (%d warnings)",
            len(report.results), report.to_dict()["status"], report.warn_count(),
        )

    silver = to_silver_schema(standardised)
    writer = ParquetWriter(settings)
    path = lake.silver_dataset_path("equity_prices")
    writer.write(silver, path, run_id=registry.run_id, overwrite=True,
                 dataset_name="equity_prices")
    return path, report
