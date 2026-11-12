"""ASIC short-position report ingestion (free research source).

ASIC publishes aggregated daily net short positions per stock as
year-to-date CSV files on ``download.asic.gov.au/short-selling/``. The report
index (``short-selling-data.json``) lists every published report date and
version; the page's JS builds per-report URLs as::

    RR{YYYYMMDD}-{version}-SSDailyYTD.csv      (year-to-date pivot)
    RR{YYYYMMDD}-{version}-SSDailyAggShortPos.csv  (single day)

The YTD pivot layout (verified against a live file): row 0 holds trade dates
(``DD/MM/YYYY``, one date per TWO columns), row 1 the sub-headers
(``Product``, ``Product Code``, then repeating ``Reported Short Positions`` /
``% of Total Product in Issue``), and the remaining rows one line per stock.

**Publication-date rule (AC07/AC08).** A report dated ``D`` aggregates the
positions reported to ASIC on or before ``D``; it is published on ``D``. For
each trade date ``T`` we set ``publication_date`` to the EARLIEST report date
in the index that is >= ``T`` — the first report through which the market
could have seen ``T``'s position. This is data-driven and conservative; it is
flagged ``estimated_from_report_calendar`` (daily reports are normally
published the next business day, so this estimate is tight but never early).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from asx_quant.exceptions import IngestionError
from asx_quant.observability.logging import get_logger

logger = get_logger("ingestion.asic")

ASIC_BASE_URL = "https://download.asic.gov.au/short-selling/"
INDEX_FILENAME = "short-selling-data.json"

PublicationIndexFn = object  # typed loosely; the index is a plain dataframe


def ytd_report_url(report_date: pd.Timestamp, version: str) -> str:
    """Build the YTD CSV URL for one report date/version."""
    stamp = report_date.strftime("%Y%m%d")
    return f"{ASIC_BASE_URL}RR{stamp}-{version}-SSDailyYTD.csv"


def fetch_report_index(http_get) -> pd.DataFrame:  # noqa: ANN001
    """Fetch and normalise the report index (``report_date, version``)."""
    content = http_get(ASIC_BASE_URL + INDEX_FILENAME)
    text = content.decode("utf-8-sig", errors="replace")
    entries = json.loads(text)
    if not isinstance(entries, list) or not entries:
        raise IngestionError("ASIC report index JSON is empty or malformed.")
    index = pd.DataFrame([
        {
            "report_date": pd.to_datetime(str(e["date"]), format="%Y%m%d",
                                          errors="coerce"),
            "version": str(e.get("version", "001")),
        }
        for e in entries
    ]).dropna(subset=["report_date"]).sort_values("report_date")
    if index.empty:
        raise IngestionError("ASIC report index contained no valid dates.")
    logger.info("ASIC report index: %d reports (%s -> %s)", len(index),
                index["report_date"].min().date(),
                index["report_date"].max().date())
    return index.reset_index(drop=True)


def pick_ytd_reports(
    index: pd.DataFrame,
    start_year: int,
) -> list[dict[str, object]]:
    """Choose one YTD report per year: the LATEST report date with a YTD file.

    The page's aggregate table lists the max date per year; earlier versions
    of the same date are superseded (highest version wins).
    """
    index = index[index["report_date"].dt.year >= start_year]
    picks: list[dict[str, object]] = []
    for year, group in index.groupby(index["report_date"].dt.year):
        latest = group.sort_values(["report_date", "version"]).iloc[-1]
        picks.append({
            "year": int(year),
            "report_date": latest["report_date"],
            "version": latest["version"],
        })
    picks.sort(key=lambda p: int(p["year"]))  # type: ignore[arg-type]
    return picks


def _decode_report_csv(content: bytes) -> tuple[str, str]:
    """Decode report bytes, returning ``(text, delimiter)``.

    Two file vintages exist: current reports are UTF-8 comma-separated;
    pre-2022 reports are UTF-16LE tab-separated. Encoding is detected from
    the BOM, delimiter from content.
    """
    if content.startswith(b"\xff\xfe") or content.startswith(b"\xfe\xff"):
        text = content.decode("utf-16")
    else:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = content.decode("utf-16")
    delimiter = "\t" if "\t" in text.splitlines()[0] else ","
    return text, delimiter


def _parse_date(cell: str) -> pd.Timestamp:
    """Parse one trade-date cell from either file vintage.

    The pre-2022 vintage uses ISO dates (``2018-11-30``); the current vintage
    uses ``dd/mm/yyyy``. ISO is tried first so the dayfirst fallback (which
    emits a pandas UserWarning on ISO strings) is only reached for the
    current vintage.
    """
    parsed = pd.to_datetime(cell, format="%Y-%m-%d", errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(cell, dayfirst=True, errors="coerce")
    return parsed


def parse_ytd_pivot(content: bytes) -> pd.DataFrame:
    """Parse one ASIC YTD pivot CSV (either file vintage) into long format.

    Layout, comma-separated current vintage::

        row 0: '', 'Trade Date', d, d, d', d', ...   (date per pos/pct pair)
        row 1: 'Product', 'Product Code', 'Reported Short Positions',
               '% of Total Product in Issue ...', ... (repeating pairs)
        row 2+: '<product name>', '<code>', pos, pct, pos, pct, ...

    Layout, tab-separated pre-2022 vintage::

        row 0: title row containing 'Trade Date'
        row 1: dates (ISO format) at pair-start columns, gaps between
        row 2: 'Reported Short Positions' / '% of ...' pairs
        row 3: 'Product', 'Product Code' label row
        row 4+: data rows, same pair alignment

    In both vintages the date for pair ``i`` sits at column ``2 + 2i``, the
    position at the same column and the pct one column later.

    Returns:
        ``trade_date, code, product_name, reported_short_positions,
        pct_of_product_in_issue, report_date``.
    """

    import csv
    import io

    text, delimiter = _decode_report_csv(content)
    # Use a real CSV reader: product names can be quoted and CONTAIN the
    # delimiter (e.g. '"GI DYNAMICS<tab> INC CDI US PROHIBITED"' in old
    # vintage files, embedded commas in current ones). Naive splitting
    # misparses such rows and shifts every subsequent column.
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    rows = [r for r in reader if any(c.strip() for c in r)]
    if len(rows) < 4:
        raise IngestionError("ASIC YTD CSV has too few rows to be valid.")

    # Locate the date row ('Trade Date' marker) and the 'Product' label row.
    date_row_idx = next(
        (i for i, r in enumerate(rows)
         if any("trade date" in c.strip().lower() for c in r)),
        None,
    )
    product_row_idx = next(
        (i for i, r in enumerate(rows)
         if r and r[0].strip() == "Product"
         and len(r) > 1 and "code" in r[1].strip().lower()),
        None,
    )
    if date_row_idx is None or product_row_idx is None:
        raise IngestionError(
            "ASIC YTD CSV layout changed: could not locate the 'Trade Date' "
            "or 'Product'/'Product Code' rows."
        )

    # The dates may be in the marker row itself (current vintage, where the
    # second column IS 'Trade Date') or the row after it (pre-2022 vintage).
    marker_row = rows[date_row_idx]
    if len(marker_row) > 2 and marker_row[1].strip().lower() == "trade date":
        date_fields = marker_row
    else:
        date_fields = rows[date_row_idx + 1]

    n_pairs = (len(date_fields) - 1) // 2
    if n_pairs < 1:
        raise IngestionError("ASIC YTD CSV contains no trade-date columns.")
    dates = [_parse_date(date_fields[2 + 2 * i]) for i in range(n_pairs)]

    def _num(cell: str) -> float:
        cell = (cell or "").strip()
        if cell in {"", "-", "\u2013"}:
            return float("nan")
        return pd.to_numeric(cell.replace(",", ""), errors="coerce")

    def _field(row: list[str], index: int) -> str:
        return row[index] if index < len(row) else ""

    records: list[dict[str, object]] = []
    for row in rows[product_row_idx + 1:]:
        if len(row) < 3 or not row[1].strip():
            continue
        code = row[1].strip().upper()
        product = row[0].strip()
        if not code or code in {"PRODUCT CODE", "-"}:
            continue
        for i in range(n_pairs):
            trade_date = dates[i]
            if pd.isna(trade_date):
                continue
            pos = _num(_field(row, 2 + 2 * i))
            pct = _num(_field(row, 3 + 2 * i))
            if pd.isna(pos) and pd.isna(pct):
                continue  # no position reported that day for this stock
            records.append({
                "trade_date": trade_date,
                "code": code,
                "product_name": product,
                "reported_short_positions": pos,
                "pct_of_product_in_issue": pct,
            })
    if not records:
        raise IngestionError("ASIC YTD CSV parsed to zero usable records.")
    frame = pd.DataFrame(records)
    frame["report_date"] = dates[0]  # the file's report date (latest session)
    return frame


def ingest_asic_reports(
    settings,  # noqa: ANN001
    http_get,  # noqa: ANN001
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, object]]]:
    """Download the report index + one YTD file per configured year.

    Returns:
        ``(index, positions, log_records)`` — the index dataframe (bronze
        metadata), the long positions dataframe, and JSONL log records.
    """
    cfg = settings.raw["ingestion"]["asic_short_positions"]
    start_year = int(cfg.get("start_year", 2015))
    pause = float(cfg.get("batch_pause_seconds", 1.5))
    retries = int(cfg.get("max_retries", 3))
    backoff = float(cfg.get("retry_backoff_seconds", 5.0))

    index = fetch_report_index(http_get)
    picks = pick_ytd_reports(index, start_year)
    logger.info("Downloading %d ASIC YTD reports (%d -> %d)", len(picks),
                picks[0]["year"], picks[-1]["year"])

    frames: list[pd.DataFrame] = []
    log_records: list[dict[str, object]] = []
    for pick in picks:
        url = ytd_report_url(pick["report_date"], str(pick["version"]))  # type: ignore[arg-type]
        year = int(pick["year"])  # type: ignore[arg-type]
        try:
            content = _retry(url, retries, backoff, http_get)
            parsed = parse_ytd_pivot(content)
            parsed["report_year"] = year
            parsed["source_url"] = url
            frames.append(parsed)
            log_records.append({
                "stage": "ingest_asic", "source": "asic",
                "source_report": f"ytd_{year}", "url": url,
                "record_count": len(parsed), "status": "success",
            })
            logger.info("ASIC YTD %d: %d records", year, len(parsed))
        except Exception as exc:  # noqa: BLE001 - report per-year failures
            logger.error("ASIC YTD %d failed: %s", year, exc)
            log_records.append({
                "stage": "ingest_asic", "source": "asic",
                "source_report": f"ytd_{year}", "url": url,
                "record_count": 0, "status": "failed", "error": str(exc),
            })
        if pause > 0:
            time.sleep(pause)
    if not frames:
        raise IngestionError(
            "No ASIC YTD report could be ingested; see ingestion log."
        )
    positions = pd.concat(frames, ignore_index=True)

    # Point-in-time publication date: earliest index report date >= trade date.
    report_dates = index["report_date"].sort_values().to_numpy()
    trade_dates = positions["trade_date"].to_numpy()
    pos_idx = pd.Index(report_dates).searchsorted(trade_dates, side="left")
    pos_idx = pd.Series(pos_idx).clip(upper=len(report_dates) - 1).to_numpy()
    publication = pd.Series(report_dates)[pos_idx].to_numpy()
    publication = pd.to_datetime(publication).to_numpy()
    trade_dates_dt = pd.to_datetime(trade_dates).to_numpy()
    # If the matched report date is BEFORE the trade date (can only happen at
    # the very end of the index), fall back to the next calendar day.
    fallback = publication < trade_dates_dt
    publication[fallback] = trade_dates_dt[fallback] + pd.Timedelta(days=1)
    positions["publication_date"] = publication
    positions["publication_date_method"] = "estimated_from_report_calendar"
    logger.info("ASIC positions built: %d rows covering %d securities",
                len(positions), positions["code"].nunique())
    return index, positions, log_records


def _retry(
    url: str,
    max_retries: int,
    backoff_seconds: float,
    http_get,  # noqa: ANN001
) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return http_get(url)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning("ASIC fetch attempt %d/%d failed (%s)",
                           attempt, max_retries, exc)
            if attempt < max_retries and backoff_seconds > 0:
                time.sleep(backoff_seconds)
    raise IngestionError(f"Could not fetch '{url}': {last_error}")


def default_asic_http_get() -> object:  # noqa: ANN201
    """Factory for the HTTP getter used for ASIC downloads."""
    import os

    import requests

    session = requests.Session()
    session.headers.update({
        "User-Agent": os.environ.get("ASX_QUANT_USER_AGENT")
        or "asx-quant-research/0.1 (educational; local)",
    })

    def _get(url: str) -> bytes:
        response = session.get(url, timeout=60)
        response.raise_for_status()
        return response.content

    return _get


def write_ingestion_log(path: Path, records: list[dict[str, object]]) -> None:
    """Append ASIC ingestion records to the domain's JSONL log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, default=str) + "\n")
