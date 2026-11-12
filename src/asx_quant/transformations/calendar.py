"""Market calendar and date dimension construction.

The ASX trading calendar is derived from the benchmark index (^AXJO) trading
dates — the index trades every ASX session. This is the ``trading-day``
basis for all feature windows and target horizons (never calendar days).
"""

from __future__ import annotations

import pandas as pd

from asx_quant.observability.logging import get_logger

logger = get_logger("transformations.calendar")

#: Benchmark source tickers (raw and sanitised forms) used as the ASX calendar.
#: ^AXJO is sanitised to IDX_AXJO in Silver ticker conventions.
BENCHMARK_TICKERS = ("^AXJO", "IDX_AXJO", "VAS.AX")


def build_market_calendar(prices: pd.DataFrame) -> pd.DataFrame:
    """Build the ASX market calendar and date dimension from price data.

    The calendar is the union of benchmark (^AXJO/VAS) sessions; when neither
    benchmark is present it falls back to the union of all observed dates.
    Output follows the DIM_DATE specification.
    """
    benchmark_dates = prices.loc[
        prices["ticker"].isin(BENCHMARK_TICKERS), "date"
    ]
    source = "benchmark_sessions"
    if benchmark_dates.empty:
        benchmark_dates = prices["date"]
        source = "union_of_sessions"
    calendar_dates = pd.DatetimeIndex(sorted(benchmark_dates.unique()))

    cal = pd.DataFrame({"calendar_date": calendar_dates})
    cal["date_key"] = cal["calendar_date"].dt.strftime("%Y%m%d").astype(int)
    cal["calendar_year"] = cal["calendar_date"].dt.year
    cal["calendar_month"] = cal["calendar_date"].dt.month
    cal["calendar_quarter"] = cal["calendar_date"].dt.quarter
    cal["day_of_week"] = cal["calendar_date"].dt.weekday + 1  # ISO: Mon=1…Sun=7
    cal["is_asx_trading_day"] = True
    cal["trading_day_sequence"] = range(len(cal))
    # Australian financial year: 1 July – 30 June.
    cal["financial_year"] = cal["calendar_year"].where(
        cal["calendar_month"] >= 7, cal["calendar_year"] - 1
    )
    cal["calendar_source"] = source
    logger.info(
        "Market calendar: %d ASX sessions (%s → %s), source=%s",
        len(cal), cal["calendar_date"].iloc[0].date(),
        cal["calendar_date"].iloc[-1].date(), source,
    )
    return cal


def compute_coverage(
    prices: pd.DataFrame,
    calendar: pd.DataFrame,
    security_master: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute per-security data-coverage metrics.

    Coverage is measured against the market calendar restricted to each
    security's own first→last observed window (securities listed mid-period
    are not penalised for missing earlier history).
    """
    sessions = pd.DatetimeIndex(calendar["calendar_date"])
    rows = []
    for ticker, group in prices.groupby("ticker"):
        dates = pd.DatetimeIndex(group["date"].unique())
        first, last = dates.min(), dates.max()
        expected = sessions[(sessions >= first) & (sessions <= last)]
        actual = set(dates)
        missing = sorted(set(expected) - actual)
        rows.append({
            "ticker": ticker,
            "first_observed_date": first,
            "last_observed_date": last,
            "n_sessions": len(dates),
            "expected_sessions": len(expected),
            "missing_sessions": len(missing),
            "missing_pct": round(len(missing) / max(len(expected), 1), 6),
            "coverage_status": (
                "ok" if len(missing) == 0
                else "minor_gaps" if len(missing) <= 2
                else "usable_with_gaps"
            ),
        })
    coverage = pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)

    if security_master is not None and not coverage.empty:
        coverage = coverage.merge(
            security_master[["security_key", "company_name", "sector",
                             "inclusion_methodology"]],
            left_on="ticker",
            right_on="security_key",
            how="left",
        ).drop(columns=["security_key"])
    return coverage
