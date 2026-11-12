"""Yahoo Finance daily price ingestion (free research source).

Wraps ``yfinance`` download behaviour behind an injectable ``download_fn``
so every aspect of ingestion (batching, retries, Bronze persistence) is
unit-testable without network access.

Conventions:
* Tickers use Yahoo's ``.AX`` suffix for ASX listings.
* The download keeps ``Close`` (raw) and ``Adj Close`` (split + dividend
  adjusted) as separate columns — the target/label layer needs adjusted
  prices while backtest execution needs raw prices.
* Dividends are recorded per ex-date; split factors default to 1.0 on
  sessions without a split (canonical semantics, not silent null-filling).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from asx_quant.exceptions import IngestionError
from asx_quant.observability.logging import get_logger

logger = get_logger("ingestion.yahoo")

RawDownloadFn = Callable[..., dict[str, pd.DataFrame]]

#: Price/volume fields a usable raw record must contain.
OHLCV_FIELDS = ("Open", "High", "Low", "Close", "Adj Close", "Volume")


def _default_download_fn(
    tickers: list[str],
    start: str,
    end: str | None,
    **kwargs: object,
) -> dict[str, pd.DataFrame]:
    """Single batch call to yfinance (single import site for the library)."""
    import yfinance as yf

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,   # keep raw Close AND Adj Close
        actions=True,        # include Dividends and Stock Splits columns
        group_by="ticker",
        threads=True,
        progress=False,
    )
    result: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return {t: pd.DataFrame() for t in tickers}
    for ticker in tickers:
        if isinstance(raw.columns, pd.MultiIndex):
            if ticker not in raw.columns.get_level_values(0):
                result[ticker] = pd.DataFrame()
                continue
            frame = raw[ticker].copy()
        else:  # single-ticker download returns a flat column index
            frame = raw.copy()
        result[ticker] = frame
    return result


class YahooDownloader:
    """Batched, retrying downloader for daily Yahoo Finance history."""

    def __init__(
        self,
        batch_size: int = 50,
        batch_pause_seconds: float = 2.0,
        max_retries: int = 3,
        retry_backoff_seconds: float = 5.0,
        download_fn: RawDownloadFn | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self._batch_size = batch_size
        self._batch_pause = batch_pause_seconds
        self._max_retries = max_retries
        self._backoff = retry_backoff_seconds
        self._download_fn = download_fn or _default_download_fn

    def download(
        self,
        tickers: list[str],
        start: date,
        end: date | None = None,
    ) -> tuple[dict[str, pd.DataFrame], list[str]]:
        """Download daily history for all tickers.

        Args:
            tickers: Yahoo-format tickers (e.g. ``CBA.AX``).
            start: First date to fetch (inclusive; provider-dependent).
            end: Optional exclusive end date (defaults to today).

        Returns:
            ``(frames, failed)`` where ``frames`` maps each ticker to a raw
            dataframe (possibly empty for missing histories) and ``failed``
            lists tickers for which no usable data could be retrieved.
        """
        frames: dict[str, pd.DataFrame] = {}
        batches = [tickers[i:i + self._batch_size]
                   for i in range(0, len(tickers), self._batch_size)]
        logger.info(
            "Downloading %d tickers in %d batches from %s",
            len(tickers), len(batches), start,
        )
        for index, batch in enumerate(batches, start=1):
            frames.update(self._download_with_retry(batch, start, end))
            if index < len(batches) and self._batch_pause > 0:
                time.sleep(self._batch_pause)

        # Individual retry for tickers that failed in their batch — single-name
        # downloads are often more reliable than batched ones.
        failed = sorted(t for t, frame in frames.items() if _is_empty(frame))
        if failed:
            logger.warning("Retrying %d failed tickers individually", len(failed))
            for ticker in failed:
                single = self._download_with_retry([ticker], start, end)
                if not _is_empty(single.get(ticker, pd.DataFrame())):
                    frames[ticker] = single[ticker]
                time.sleep(self._backoff) if self._backoff > 0 else None

        final_failed = sorted(t for t, frame in frames.items() if _is_empty(frame))
        succeeded = [t for t in tickers if not _is_empty(frames.get(t, pd.DataFrame()))]
        logger.info("Download complete: %d succeeded, %d failed",
                    len(succeeded), len(final_failed))
        return frames, final_failed

    def _download_with_retry(
        self,
        batch: list[str],
        start: date,
        end: date | None,
    ) -> dict[str, pd.DataFrame]:
        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                frames = self._download_fn(
                    batch,
                    start.isoformat(),
                    end.isoformat() if end else None,
                )
                if any(not _is_empty(f) for f in frames.values()):
                    return frames
                last_error = IngestionError(
                    f"Batch download returned no data for {batch[:3]}..."
                )
            except Exception as exc:  # noqa: BLE001 - provider errors vary
                last_error = exc
            logger.warning(
                "Batch download attempt %d/%d failed (%s)",
                attempt, self._max_retries, last_error,
            )
            if attempt < self._max_retries and self._backoff > 0:
                time.sleep(self._backoff)
        logger.error("Giving up on batch after %d attempts", self._max_retries)
        return {t: pd.DataFrame() for t in batch}


def normalise_raw_frame(ticker: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise one raw yfinance frame to the Bronze schema.

    Output columns: ``date, open, high, low, close, adj_close, volume,
    dividends, split_factor`` plus source metadata. Rows whose OHLCV are all
    missing are dropped (provider padding artefacts), never counted as data.
    """
    if _is_empty(frame):
        return pd.DataFrame(
            columns=["date", "open", "high", "low", "close", "adj_close",
                     "volume", "dividends", "split_factor"]
        )
    out = frame.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    rename = {"adj_close": "adj_close", "stock_splits": "split_factor"}
    out = out.rename(columns=rename)
    required = {"open", "high", "low", "close", "volume"}
    missing = required - set(out.columns)
    if missing:
        raise IngestionError(
            f"Ticker '{ticker}': download is missing columns {sorted(missing)}."
        )
    keep = ["open", "high", "low", "close", "adj_close", "volume",
            "dividends", "split_factor"]
    for col in ("adj_close", "dividends", "split_factor"):
        if col not in out.columns:
            out[col] = 0.0 if col != "split_factor" else 1.0
    out = out[keep]

    index = pd.to_datetime(out.index)
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)
    out.index = index.normalize()
    out = out[~out[list(required)].isna().all(axis=1)]
    out = out.reset_index()
    # The first column is whatever the provider named its date index
    # ("Date", "index", ...); rename it to the canonical 'date'.
    first_col = out.columns[0]
    if first_col != "date":
        out = out.rename(columns={first_col: "date"})
    out.insert(0, "source_ticker", ticker)
    out.insert(1, "source", "yfinance")
    out["split_factor"] = out["split_factor"].fillna(1.0)
    out["dividends"] = out["dividends"].fillna(0.0)
    return out


def append_ingestion_log(
    log_path: Path,
    records: list[dict[str, object]],
) -> None:
    """Append ingestion metadata records to the JSONL ingestion log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, default=str) + "\n")


def history_start(settings) -> date:  # noqa: ANN001
    """Compute the historical start date from ``settings``."""
    years = int(settings.raw["ingestion"]["market_data"]["history_years"])
    return date.today() - timedelta(days=365 * years)


def _is_empty(frame: pd.DataFrame) -> bool:
    return frame is None or frame.empty
