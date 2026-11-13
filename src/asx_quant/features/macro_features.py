"""Point-in-time macro features.

THE leakage rule of the platform: a macro value may only appear in a
feature row whose session date is >= the value's ``publication_date``.
Joins are as-of backward on publication_date over the trading calendar —
NEVER on the observation period. Monthly/quarterly values stay constant
between publications; every such row inherits the source's
``potentially_revised`` flag from the Silver data.
"""

from __future__ import annotations

import pandas as pd


def build_macro_pit_matrix(
    macro: pd.DataFrame,
    calendar_dates: pd.DatetimeIndex,
    series_ids: list[str],
) -> pd.DataFrame:
    """As-of-join every series onto the calendar by publication_date.

    Args:
        macro: Silver ``macro_observations`` (series_name, observation_date,
            publication_date, value).
        calendar_dates: Trading sessions to index the output by.
        series_ids: Canonical series names to include.

    Returns:
        Frame indexed by calendar date, one column per series, values as
        known at that date (NaN until first publication).
    """
    if macro.empty:
        empty = pd.DataFrame(index=pd.DatetimeIndex(calendar_dates))
        for series in series_ids:
            empty[f"macro_{series}"] = float("nan")
        return empty

    macro = macro.copy()
    macro["publication_date"] = pd.to_datetime(macro["publication_date"])
    macro["value"] = pd.to_numeric(macro["value"], errors="coerce")
    macro = macro.sort_values(["publication_date", "observation_date"])

    out = pd.DataFrame({"date": pd.DatetimeIndex(calendar_dates)})
    for series in series_ids:
        one = macro[macro["series_name"] == series]
        if one.empty:
            out[f"macro_{series}"] = float("nan")
            continue
        one = one.dropna(subset=["publication_date", "value"])
        # Newest observation wins if several share a publication date.
        one = one.drop_duplicates(subset=["publication_date"], keep="last")
        merged = pd.merge_asof(
            out,
            one[["publication_date", "value"]],
            left_on="date",
            right_on="publication_date",
            direction="backward",
        )
        out[f"macro_{series}"] = merged["value"].to_numpy()
    return out.set_index("date")


def add_derived_macro_features(matrix: pd.DataFrame) -> pd.DataFrame:
    """Derived macro features computable from the joined matrix only."""
    out = matrix.copy()
    y10, y2 = "macro_au_gov_yield_10y", "macro_au_gov_yield_2y"
    if y10 in out.columns and y2 in out.columns:
        out["macro_curve_slope_10y_2y"] = out[y10] - out[y2]
    return out
