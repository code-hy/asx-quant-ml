"""Point-in-time short-interest features (ASIC aggregated reports).

A short position reported for trade date ``T`` becomes usable from its
``publication_date`` (the earliest ASIC report date >= ``T``). The join is
as-of backward on publication_date per ticker, then forward-filled per
ticker onto the feature grid; changes and percentiles are computed on the
filled series with shift (never comparing values across a publication gap
as if both were known at the same time — the fill itself is the
disclosure-correct carrier).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_short_pit_features(
    positions: pd.DataFrame,
    universe: pd.DataFrame,
    change_window: int = 5,
) -> pd.DataFrame:
    """Short-interest features aligned to the universe (ticker, date) grid.

    Args:
        positions: Silver ``short_positions`` (ticker, trade_date,
            publication_date, reported_short_positions,
            pct_of_product_in_issue).
        universe: Feature grid (ticker, date) rows, sorted.
        change_window: Sessions between consecutive short-pct values for the
            change feature.

    Returns:
        Frame with the universe's index and columns ``short_pct``,
        ``short_positions``, ``short_pct_change_{w}d``,
        ``short_xsec_pctile``.
    """
    out = pd.DataFrame(index=universe.index)
    if positions.empty:
        out["short_pct"] = float("nan")
        out["short_positions"] = float("nan")
        return out

    positions = positions.copy()
    positions["publication_date"] = pd.to_datetime(
        positions["publication_date"]
    ).dt.normalize()
    positions["trade_date"] = pd.to_datetime(
        positions["trade_date"]
    ).dt.normalize()
    positions = positions.sort_values(
        ["publication_date", "trade_date", "ticker"]
    )
    # At a given publication the LATEST trade date's value is the freshest
    # known position; earlier trade dates in the same publication are stale.
    positions = positions.drop_duplicates(
        subset=["ticker", "publication_date"], keep="last"
    )
    positions = positions.rename(columns={"publication_date": "_asof_date"})

    grid = universe[["ticker", "date"]].copy()
    grid["_row"] = np.arange(len(grid))
    # merge_asof requires BOTH frames sorted on the join key ('date').
    grid = grid.sort_values("date", kind="stable")
    merged = pd.merge_asof(
        grid,
        positions[[
            "_asof_date", "ticker", "pct_of_product_in_issue",
            "reported_short_positions",
        ]],
        left_on="date",
        right_on="_asof_date",
        by="ticker",
        direction="backward",
        suffixes=("", "_pos"),
    )
    # merge_asof returns a fresh RangeIndex in date-major order — restore
    # the universe's row order explicitly via the carried row position.
    merged = merged.sort_values("_row")

    out = pd.DataFrame(index=universe.index)
    out["short_pct"] = pd.Series(
        merged["pct_of_product_in_issue"].to_numpy(),
        index=merged["_row"].to_numpy(),
    ).reindex(universe.index).to_numpy()
    out["short_positions"] = pd.Series(
        merged["reported_short_positions"].to_numpy(),
        index=merged["_row"].to_numpy(),
    ).reindex(universe.index).to_numpy()

    # Per-ticker session shifts on the FILLED grid (backward-looking).
    filled = out["short_pct"].groupby(universe["ticker"], sort=False)
    out[f"short_pct_change_{change_window}d"] = filled.transform(
        lambda s: s - s.shift(change_window)
    )
    out["short_xsec_pctile"] = out.groupby(
        universe["date"], sort=False
    )["short_pct"].rank(pct=True)
    return out
