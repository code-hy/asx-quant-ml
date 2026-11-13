"""Shared helpers for feature computation.

All feature functions are pure: the same input frame always produces the
same output, and every computation for session ``t`` uses only rows with
date ``<= t`` (rolling/backward windows, never forward).
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd


def per_ticker(
    prices: pd.DataFrame,
    fn: Callable[[pd.DataFrame], pd.DataFrame],
) -> pd.DataFrame:
    """Apply ``fn`` to each ticker's sub-frame, preserving row order.

    ``fn`` receives the ticker's rows (already sorted by date) and returns a
    frame with the same index plus new feature columns. Grouping keeps
    leakage boundaries intact: a ticker's features never see another
    ticker's rows except through explicitly date-aligned cross-sectional
    helpers.
    """
    parts = [
        fn(group)
        for _, group in prices.groupby("ticker", sort=False)
    ]
    if not parts:
        return prices.copy()
    return pd.concat(parts).sort_index()


def cross_section(
    prices: pd.DataFrame,
    fn: Callable[[pd.DataFrame], pd.DataFrame],
) -> pd.DataFrame:
    """Apply ``fn`` to each session's cross-section, preserving row order.

    ``fn`` receives all universe rows for ONE date and returns a frame with
    the same index plus new columns (percentile ranks, breadth inputs...).
    Same-date-only information — never future or per-ticker history.
    """
    parts = [
        fn(group)
        for _, group in prices.groupby("date", sort=False)
    ]
    if not parts:
        return prices.copy()
    return pd.concat(parts).sort_index()


def ensure_sorted(prices: pd.DataFrame) -> pd.DataFrame:
    """Sort by (ticker, date) and normalise the date column."""
    out = prices.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)
