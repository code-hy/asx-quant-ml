"""Target labels (forward returns) with purge/embargo metadata.

Targets are the ONLY gold columns allowed to reference future sessions.
Each label carries ``label_available_date`` — the session whose close
completes the label — so downstream train/test splits can purge overlapping
samples and embargo by the configured session count. Price basis is the
ADJUSTED close (dividends accrue to holders), per docs/backtesting notes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_target_labels(
    universe: pd.DataFrame,
    horizons: tuple[int, ...] = (3, 10),
    direction_threshold: float = 0.0,
) -> pd.DataFrame:
    """Forward returns and direction labels for each horizon.

    Args:
        universe: Feature grid (ticker, date, adjusted_close) sorted by
            (ticker, date).
        horizons: Label horizons in TRADING sessions.
        direction_threshold: Class 1 when future return exceeds threshold.

    Returns:
        Frame with the universe's index and columns per horizon:
        ``future_return_{h}d``, ``direction_{h}d``,
        ``label_available_date_{h}d``.
    """
    out = pd.DataFrame(index=universe.index)
    grouped = universe.groupby("ticker", sort=False)["adjusted_close"]
    dates = universe["date"]
    for horizon in horizons:
        future_price = grouped.shift(-horizon)
        future_return = future_price / universe["adjusted_close"] - 1.0
        out[f"future_return_{horizon}d"] = future_return
        out[f"direction_{horizon}d"] = (
            (future_return > direction_threshold).astype(float).where(
                future_return.notna(), np.nan
            )
        )
        # The session whose close completes the label (for purge/embargo).
        out[f"label_available_date_{horizon}d"] = dates.groupby(
            universe["ticker"], sort=False
        ).shift(-horizon)
    return out
