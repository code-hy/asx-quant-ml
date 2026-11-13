"""Market, benchmark, breadth and cross-sectional features.

The benchmark (^AXJO / VAS.AX) provides market-level context broadcast to
every ticker by session date. Breadth and cross-sectional percentiles use
ONLY same-session universe information — never future sessions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from asx_quant.transformations.calendar import BENCHMARK_TICKERS


def pick_benchmark_ticker(prices: pd.DataFrame) -> str | None:
    """First available benchmark ticker in priority order."""
    present = set(prices["ticker"].unique())
    for candidate in BENCHMARK_TICKERS:
        if candidate in present:
            return candidate
    return None


def compute_benchmark_features(
    benchmark: pd.DataFrame,
    regime_windows: list[int],
) -> pd.DataFrame:
    """Benchmark-level features, one row per session.

    ``benchmark`` must contain EXACTLY ONE benchmark series (call
    ``pick_benchmark_ticker`` first). Mixing two index series of different
    levels would fabricate enormous fake returns at every switch.

    Returns a frame indexed by date with columns: ``bench_return_1d``,
    ``bench_return_{w}d`` and ``bench_close_to_sma_{w}`` per regime window,
    plus ``bench_volatility_{w}d`` for the first window.
    """
    tickers = benchmark["ticker"].unique()
    if len(tickers) != 1:
        raise ValueError(
            f"compute_benchmark_features expects a single benchmark series, "
            f"got {sorted(str(t) for t in tickers)}. Select one with "
            "pick_benchmark_ticker() before calling."
        )
    bench = (
        benchmark.sort_values("date")
        .drop_duplicates("date")
        # Reset so later Series assignments align by POSITION, not by the
        # original (non-contiguous) row labels — a label misalignment here
        # silently NaNs most of the series.
        .reset_index(drop=True)
    )
    out = pd.DataFrame({"date": bench["date"].to_numpy()})
    out["bench_return_1d"] = bench["adjusted_close"].pct_change()
    first = regime_windows[0] if regime_windows else 20
    out[f"bench_volatility_{first}d"] = (
        out["bench_return_1d"].rolling(first).std() * np.sqrt(252)
    )
    for window in regime_windows:
        out[f"bench_return_{window}d"] = (
            bench["adjusted_close"].pct_change(window)
        )
        sma = bench["adjusted_close"].rolling(window).mean()
        out[f"bench_close_to_sma_{window}"] = bench["adjusted_close"] / sma
    return out


def compute_breadth_features(
    universe: pd.DataFrame,
    ma_windows: list[int],
) -> pd.DataFrame:
    """Fraction of the universe above its own SMA per session (index=date).

    Uses the price_to_sma_* columns (computed per ticker); a session's
    breadth depends only on that session's values.
    """
    out = pd.DataFrame({"date": sorted(universe["date"].unique())})
    for window in ma_windows:
        col = f"price_to_sma_{window}"
        if col not in universe.columns:
            continue
        above = (
            universe[["date", col]]
            .assign(above=universe[col] > 1.0)
            .groupby("date")["above"]
            .mean()
            .rename(f"breadth_above_sma_{window}")
        )
        out = out.merge(above.reset_index(), on="date", how="left")
    return out


def compute_cross_sectional_features(
    universe: pd.DataFrame,
    momentum_window: int = 20,
) -> pd.DataFrame:
    """Same-session cross-sectional percentile ranks (0..1).

    * ``momentum_xsec_pctile_{w}d`` — rank of the w-session return among the
      universe that session;
    * ``turnover_xsec_pctile`` — rank of 20-session average dollar turnover.
    """
    out = pd.DataFrame(index=universe.index)
    ret_col = f"return_{momentum_window}d"
    if ret_col in universe.columns:
        out[f"momentum_xsec_pctile_{momentum_window}d"] = universe.groupby(
            "date", sort=False
        )[ret_col].rank(pct=True)
    if "dollar_turnover_20d" in universe.columns:
        out["turnover_xsec_pctile"] = universe.groupby(
            "date", sort=False
        )["dollar_turnover_20d"].rank(pct=True)
    return out


def compute_beta_feature(
    universe: pd.DataFrame,
    benchmark_returns: pd.Series,
    beta_window: int = 252,
) -> pd.Series:
    """Rolling beta of each ticker's daily return vs the benchmark.

    ``benchmark_returns`` must be a date-indexed series aligned to the
    universe frame's dates. Computed per ticker over its own trailing
    window (backward-looking).
    """

    def _beta(group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()
        bench = g["date"].map(benchmark_returns)
        g["_bench_ret"] = bench
        cov = g["daily_return"].rolling(beta_window).cov(g["_bench_ret"])
        var = g["_bench_ret"].rolling(beta_window).var()
        g[f"beta_{beta_window}d"] = cov / var.replace(0.0, np.nan)
        return g.drop(columns=["_bench_ret"])

    from asx_quant.features.base import per_ticker

    result = per_ticker(universe, _beta)
    return result[f"beta_{beta_window}d"]
