"""Security universe construction and security master building.

Primary free constituent source: the S&P/ASX 200 constituent table on
Wikipedia (see docs/data_sources.md for terms and limitations). The curated
seed list in ``config/universe.yaml`` is the offline fallback.

**Known limitation (survivorship bias):** today's constituents are applied to
the whole historical period. Securities that left the index (or delisted) are
absent from history. This is recorded in the security master's
``inclusion_methodology`` and must be disclosed in every report.
"""

from __future__ import annotations

import io
import os
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

from asx_quant.exceptions import IngestionError
from asx_quant.observability.logging import get_logger
from asx_quant.storage.parquet import ParquetWriter

logger = get_logger("ingestion.universe")

WIKIPEDIA_ASX200_URL = "https://en.wikipedia.org/wiki/S%26P/ASX_200"
_DEFAULT_USER_AGENT = "asx-quant-research/0.1 (educational; local)"
_FETCH_TIMEOUT_SECONDS = 30

UniverseFetchFn = Callable[[], pd.DataFrame]


def fetch_asx200_constituents(user_agent: str | None = None) -> pd.DataFrame:
    """Fetch the S&P/ASX 200 constituent table from Wikipedia.

    Returns:
        Dataframe with columns ``code`` (3-letter ASX code, no suffix),
        ``company_name``, ``sector``, ``industry`` (both may be null).
    """
    headers = {"User-Agent": user_agent or _DEFAULT_USER_AGENT}
    try:
        response = requests.get(
            WIKIPEDIA_ASX200_URL, headers=headers, timeout=_FETCH_TIMEOUT_SECONDS
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise IngestionError(
            f"Could not fetch ASX 200 constituents from Wikipedia: {exc}"
        ) from exc

    try:
        tables = pd.read_html(io.StringIO(response.text))
    except ValueError as exc:
        raise IngestionError(
            f"Could not parse HTML tables on the Wikipedia ASX 200 page: {exc}"
        ) from exc

    for table in tables:
        if not isinstance(table, pd.DataFrame) or table.empty:
            continue
        columns = {str(c).strip().lower(): c for c in table.columns}
        code_col = next((c for k, c in columns.items() if k in ("code", "asx code", "symbol")), None)
        name_col = next((c for k, c in columns.items() if "company" in k or "name" in k), None)
        if code_col is None or name_col is None:
            continue
        sector_col = next(
            (c for k, c in columns.items() if "sector" in k and "sub" not in k),
            None,
        )
        industry_col = next(
            (c for k, c in columns.items() if "industry" in k or "sub" in k),
            None,
        )
        out = table[[code_col, name_col] + ([sector_col] if sector_col else [])
                    + ([industry_col] if industry_col else [])].copy()
        out.columns = ["code", "company_name"] + \
            (["sector"] if sector_col else []) + \
            (["industry"] if industry_col else [])
        for col in ("sector", "industry"):
            if col not in out.columns:
                out[col] = None
        out = _clean_codes(out)
        if len(out) >= 150:  # a genuine constituents table
            logger.info(
                "Fetched %d ASX 200 constituents from Wikipedia", len(out)
            )
            return out
    raise IngestionError(
        "No usable constituents table found on the Wikipedia ASX 200 page."
    )


def _clean_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a raw constituents table: canonical columns, valid codes only."""
    out = df.rename(columns=lambda c: str(c).strip().lower())
    out = out.rename(columns={"asx code": "code", "symbol": "code"})
    for base, pattern in (("sector", r"sector"), ("industry", r"industry")):
        if base not in out.columns:
            match = next((c for c in out.columns
                          if re.search(pattern, c) and "sub" not in c), None)
            if match:
                out = out.rename(columns={match: base})
    for col in ("sector", "industry"):
        if col not in out.columns:
            out[col] = None

    def _norm(code: str) -> str:
        code = re.sub(r"[^A-Za-z0-9]", "", str(code)).upper()
        return code if re.fullmatch(r"[A-Z0-9]{3}", code) else ""

    out["code"] = out["code"].map(_norm)
    out = out[(out["code"] != "") & (out["code"] != "NAN")]
    for col in ("sector", "industry"):
        out[col] = out[col].astype(str).str.strip().replace(
            {"nan": None, "None": None, "": None, "-": None}
        )
    return out.drop_duplicates(subset="code").reset_index(drop=True)


def build_universe(
    settings,  # noqa: ANN001
    fetch_fn: UniverseFetchFn | None = None,
) -> pd.DataFrame:
    """Build the target universe dataframe from configured sources.

    Tries the configured free constituent source first; on failure falls
    back to the seed list in ``config/universe.yaml`` when
    ``ingestion.universe.fallback_to_seed`` is true.

    Returns:
        Dataframe with ``ticker`` (``.AX`` suffixed), ``code``, and best
        available ``company_name`` / ``sector`` / ``industry`` plus a
        ``universe_source`` column recording where each row came from.
    """
    cfg = settings.raw["ingestion"]["universe"]
    if cfg["source"] == "wikipedia_asx200":
        agent = os.environ.get(cfg.get("user_agent_env", "")) or None
        try:
            wiki = (fetch_fn or fetch_asx200_constituents)(user_agent=agent)
            wiki["ticker"] = wiki["code"] + ".AX"
            wiki["universe_source"] = "wikipedia_asx200"
            return wiki
        except IngestionError as exc:
            if not cfg.get("fallback_to_seed", False):
                raise
            logger.warning(
                "Constituent fetch failed (%s); falling back to seed list.", exc
            )
    return _universe_from_seed(settings)


def _universe_from_seed(settings) -> pd.DataFrame:  # noqa: ANN001
    from asx_quant.config.loader import load_yaml

    seed = load_yaml("universe")["universe"]["seed_tickers"]
    rows = []
    for entry in seed:
        ticker = entry["ticker"].upper()
        rows.append({
            "code": ticker.split(".")[0],
            "ticker": ticker,
            "company_name": None,
            "sector": entry.get("sector"),
            "industry": None,
            "universe_source": "config_seed",
        })
    df = pd.DataFrame(rows)
    logger.info("Loaded %d seed tickers from universe.yaml", len(df))
    return df


def write_bronze_security_master(
    lake,  # noqa: ANN001
    registry,  # noqa: ANN001
    settings,  # noqa: ANN001
    universe: pd.DataFrame,
) -> Path:
    """Write an immutable Bronze batch of the security master."""
    writer = ParquetWriter(settings)
    # The run id doubles as the batch id: sortable (timestamp prefix) and
    # unique (uuid suffix), so rapid re-runs never overwrite each other.
    path = lake.bronze_batch_path("security_master", "asx200", registry.run_id)
    writer.write(universe, path, run_id=registry.run_id,
                 overwrite=True, dataset_name="security_master_bronze")
    return path


def add_master_metadata(universe: pd.DataFrame, settings) -> pd.DataFrame:  # noqa: ANN001
    """Extend the universe dataframe to the full DIM_SECURITY schema."""
    df = universe.copy()
    df["exchange"] = "ASX"
    df["currency"] = "AUD"
    df["instrument_type"] = "ordinary_share"
    df["active_flag"] = True
    df["source_identifier"] = df["ticker"]
    df["listing_date"] = pd.NaT
    df["delisting_date"] = pd.NaT
    df["inclusion_methodology"] = (
        f"{df['universe_source'].iloc[0]}_as_of_"
        f"{datetime.now(UTC):%Y-%m-%d}"
    )
    df["data_quality_status"] = "pending_coverage_check"
    df["security_key"] = df["ticker"]
    return df
