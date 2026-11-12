"""RBA statistical-table ingestion (free research source).

Ingests the Reserve Bank of Australia's full-history statistical CSVs and the
cash-rate change history:

* Tables F2 (daily government bond yields), F11/F11.1 (FX), G1 (CPI — ABS
  data republished by the RBA), H5 (labour force — ABS), I2 (commodity
  prices). Endpoint pattern:
  ``https://www.rba.gov.au/statistics/tables/csv/<table>-data.csv``
* Cash-rate changes from ``https://www.rba.gov.au/statistics/cash-rate/``
  (HTML table of effective dates).

**Point-in-time publication rules (AC07).** The CSV ``Publication date``
metadata row is the CURRENT vintage only — it is NOT the publication date of
each historical observation and must never be used per-row. Instead:

* daily series  -> ``publication_date = observation_date`` (``same_day_publication``)
* monthly / quarterly series -> ``observation_date + configured lag days``
  (``estimated_publication_lag``, flagged ``potentially_revised``)
* cash rate -> the RBA announces at 2:30pm on the decision date and the new
  rate is EFFECTIVE the following weekday, so the observation is dated at the
  ANNOUNCEMENT date (``effective_date - 1 day``, ``same_day_publication``);
  the original effective date is kept for provenance.

These are conservative estimates; every estimated value is flagged so no
downstream code can mistake it for an exact vintage date.
"""

from __future__ import annotations

import io
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from asx_quant.exceptions import IngestionError
from asx_quant.observability.logging import get_logger

logger = get_logger("ingestion.rba")

TABLES_BASE_URL = "https://www.rba.gov.au/statistics/tables/csv/"
CASH_RATE_URL = "https://www.rba.gov.au/statistics/cash-rate/"

DEFAULT_USER_AGENT = "asx-quant-research/0.1 (educational; local)"
_FETCH_TIMEOUT_SECONDS = 60

#: Frequency-level publication-lag defaults (per-series config overrides).
DEFAULT_PUBLICATION_LAG_DAYS = {"monthly": 15, "quarterly": 35}

HTTPGet = Callable[[str], bytes]


def _default_http_get(url: str) -> bytes:
    """Single import site for HTTP retrieval of RBA resources."""
    import requests

    response = requests.get(
        url,
        headers={"User-Agent": os.environ.get("ASX_QUANT_USER_AGENT")
                 or DEFAULT_USER_AGENT},
        timeout=_FETCH_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.content


@dataclass(frozen=True)
class SeriesSpec:
    """One configured macro series."""

    series_id: str          # RBA series ID, e.g. FCMYGBAG2D
    name: str               # canonical feature name, e.g. au_gov_yield_2y
    unit: str
    table: str              # source table key, e.g. F2
    csv_name: str
    frequency: str          # daily | weekly | monthly | quarterly
    original_publisher: str = "rba"
    publication_lag_days: int | None = None


def load_series_specs(settings) -> list[SeriesSpec]:  # noqa: ANN001
    """Build the configured series list from ``ingestion.macro.rba.tables``."""
    cfg = settings.raw["ingestion"]["macro"]["rba"]
    freq_defaults = DEFAULT_PUBLICATION_LAG_DAYS | cfg.get(
        "publication_lag_days", {}
    )
    specs: list[SeriesSpec] = []
    for table_key, table_cfg in cfg["tables"].items():
        for series_id, meta in table_cfg["series"].items():
            specs.append(SeriesSpec(
                series_id=series_id,
                name=meta["name"],
                unit=meta["unit"],
                table=table_key,
                csv_name=table_cfg["csv_name"],
                frequency=table_cfg["frequency"],
                original_publisher=meta.get("original_publisher", "rba"),
                publication_lag_days=(
                    int(meta["publication_lag_days"])
                    if "publication_lag_days" in meta
                    else int(freq_defaults.get(table_cfg["frequency"], 0))
                ),
            ))
    return specs


def fetch_with_retry(
    url: str,
    max_retries: int = 3,
    backoff_seconds: float = 5.0,
    http_get: HTTPGet | None = None,
) -> bytes:
    """Fetch a URL as bytes with retries (ASIC/RBA endpoints can be flaky)."""
    getter = http_get or _default_http_get
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return getter(url)
        except Exception as exc:  # noqa: BLE001 - provider errors vary
            last_error = exc
            logger.warning("Fetch attempt %d/%d failed for %s (%s)",
                           attempt, max_retries, url, exc)
            if attempt < max_retries and backoff_seconds > 0:
                time.sleep(backoff_seconds)
    raise IngestionError(f"Could not fetch '{url}': {last_error}") from last_error


# ------------------------------------------------------------------ parsing
def parse_rba_csv(content: bytes) -> pd.DataFrame:
    """Parse a full-history RBA statistical CSV into long format.

    Layout (verified against live files): metadata preamble rows, then a row
    starting with ``Series ID`` whose remaining cells are the series IDs; the
    FIRST column holds dates in mixed ``DD-Mon-YYYY`` / ``DD/MM/YYYY`` forms.
    Forward-looking placeholder rows (future dates with NaN values) are
    dropped — never counted as data.

    Returns:
        Long dataframe: ``observation_date, series_id, value``.
    """
    text = content.decode("utf-8-sig", errors="replace")
    lines = text.splitlines()
    header_idx = next(
        (i for i, line in enumerate(lines)
         if line.split(",")[0].strip().strip('"').lower() == "series id"),
        None,
    )
    if header_idx is None:
        raise IngestionError(
            "RBA CSV is missing its 'Series ID' header row — layout may have "
            "changed; refusing to guess."
        )
    raw = pd.read_csv(io.StringIO(text), skiprows=header_idx, dtype=str)
    if raw.columns[0] != "Series ID" or len(raw.columns) < 2:
        raise IngestionError("Unexpected RBA CSV header layout.")
    raw = raw.rename(columns={"Series ID": "observation_date"})

    long = raw.melt(id_vars=["observation_date"], var_name="series_id",
                    value_name="value")
    long["observation_date"] = pd.to_datetime(
        long["observation_date"], dayfirst=True, format="mixed", errors="coerce"
    )
    long["value"] = pd.to_numeric(
        long["value"].str.replace(",", "", regex=False), errors="coerce"
    )
    long = long.dropna(subset=["observation_date", "value"])
    if long.empty:
        raise IngestionError("RBA CSV parsed to zero usable observations.")
    return long[["observation_date", "series_id", "value"]].reset_index(drop=True)


def parse_cash_rate_html(content: bytes, min_rows: int = 5) -> pd.DataFrame:
    """Parse the RBA cash-rate change-history HTML table.

    Args:
        content: Raw HTML bytes of the /statistics/cash-rate/ page.
        min_rows: Minimum row count to accept a table as the change history;
            lower it for test fixtures. The real history table has 400+ rows,
            and candidate tables must also match both expected column names,
            so a small threshold stays safe.

    Returns:
        ``effective_date, cash_rate_target`` — the announcement date is one
        weekday before the effective date (see module docstring).
    """
    tables = pd.read_html(io.BytesIO(content))
    for table in tables:
        cols = {str(c).strip().lower(): c for c in table.columns}
        eff_col = next((c for k, c in cols.items()
                        if "effective" in k and "date" in k), None)
        target_col = next((c for k, c in cols.items()
                           if "cash rate target" in k), None)
        if eff_col is None or target_col is None:
            continue
        out = pd.DataFrame({
            "effective_date": table[eff_col].astype(str).str.strip(),
            "cash_rate_target": pd.to_numeric(
                table[target_col].astype(str).str.replace(r"[^0-9.\-]", "",
                                                          regex=True),
                errors="coerce",
            ),
        })
        out["effective_date"] = pd.to_datetime(
            out["effective_date"], format="%d %b %Y", errors="coerce"
        )
        out = out.dropna().reset_index(drop=True)
        if len(out) >= min_rows:  # the real change-history table is long
            return out
    raise IngestionError(
        "Could not parse the cash-rate change table from the RBA page."
    )


def assign_publication_dates(
    observations: pd.DataFrame,
    frequency: str,
    lag_days: int,
    method: str | None = None,
) -> pd.DataFrame:
    """Assign publication dates + method flags to one table's observations.

    ``observations`` must already contain ``observation_date``. Adds:
    ``publication_date, publication_date_method, potentially_revised``.
    """
    out = observations.copy()
    if method is None:
        method = ("same_day_publication" if frequency == "daily"
                  else "estimated_publication_lag")
    if method == "same_day_publication":
        out["publication_date"] = out["observation_date"]
        out["potentially_revised"] = False
    elif method == "estimated_publication_lag":
        out["publication_date"] = out["observation_date"] + pd.Timedelta(days=lag_days)
        out["potentially_revised"] = True
    else:  # announcement_date: effective date minus one weekday-approximate day
        out["publication_date"] = out["observation_date"] - pd.Timedelta(days=1)
        out["potentially_revised"] = False
    out["publication_date_method"] = method
    return out


def build_macro_observations(
    settings: object,
    http_get: HTTPGet | None = None,
    max_retries: int = 3,
    backoff_seconds: float = 5.0,
) -> pd.DataFrame:
    """Download and parse every configured RBA table into one long frame.

    All columns needed downstream are attached here (Bronze rows carry the
    publication-date derivation so Silver is pure conformance).
    """
    specs = load_series_specs(settings)
    by_table: dict[str, list[SeriesSpec]] = {}
    for spec in specs:
        by_table.setdefault(spec.table, []).append(spec)

    frames: list[pd.DataFrame] = []
    for table_key, table_specs in by_table.items():
        url = TABLES_BASE_URL + table_specs[0].csv_name
        logger.info("Fetching RBA table %s from %s", table_key, url)
        content = fetch_with_retry(url, max_retries, backoff_seconds, http_get)
        parsed = parse_rba_csv(content)
        wanted = {s.series_id: s for s in table_specs}
        parsed = parsed[parsed["series_id"].isin(wanted)]
        if parsed.empty:
            logger.warning("RBA table %s contained none of the configured "
                           "series; skipping.", table_key)
            continue
        for series_id, group in parsed.groupby("series_id"):
            spec = wanted[series_id]
            obs = group.rename(columns={"series_id": "source_series_id"})
            obs = assign_publication_dates(
                obs, spec.frequency, spec.publication_lag_days or 0
            )
            obs.insert(0, "series_name", spec.name)
            obs["source"] = spec.original_publisher
            obs["source_table"] = table_key
            obs["frequency"] = spec.frequency
            obs["unit"] = spec.unit
            frames.append(obs)
        logger.info("RBA table %s: %d usable observations",
                    table_key, len(parsed))

    # Cash-rate change history. The RBA announces at 2:30pm on the decision
    # date and the new target is EFFECTIVE the following weekday. The
    # observation is therefore dated at the ANNOUNCEMENT date (the day the
    # market knows the new rate; it is in force from that day's close), with
    # same-day publication — and the original effective date is kept for
    # provenance. This keeps every series consistent with the leakage rule
    # (publication_date >= observation_date).
    cash_cfg = settings.raw["ingestion"]["macro"]["rba"].get("cash_rate", {})
    if cash_cfg.get("enabled", True):
        content = fetch_with_retry(CASH_RATE_URL, max_retries, backoff_seconds,
                                   http_get)
        cash = parse_cash_rate_html(content)
        cash["observation_date"] = (
            cash["effective_date"] - pd.Timedelta(days=1)
        )
        cash = cash.rename(columns={"cash_rate_target": "value"})
        cash["source_series_id"] = "CASH_RATE_TARGET"
        cash = assign_publication_dates(
            cash, "daily", 0, method="same_day_publication"
        )
        cash.insert(0, "series_name", cash_cfg.get("series_name", "rba_cash_rate"))
        cash["source"] = "rba"
        cash["source_table"] = "CASH_RATE"
        cash["frequency"] = "daily"
        cash["unit"] = "per_cent"
        frames.append(cash)

    macro = pd.concat(frames, ignore_index=True)
    macro["source_url"] = ""  # per-row provenance kept at batch level in bronze
    logger.info("Macro observations built: %d rows, %d series",
                len(macro), macro["series_name"].nunique())
    return macro


def macro_history_start(settings) -> date:  # noqa: ANN001
    """Start date for macro ingestion: aligned with the price lookback."""
    from asx_quant.ingestion.yahoo import history_start

    return history_start(settings) - timedelta(days=90)  # buffer for lags
