"""Ticker-level price, momentum, MA, volatility, technical and volume features.

Every computation is backward-looking per ticker: rolling windows, shifting
and EWM only. Warm-up periods yield NaN (null policy ``null_as_missing``) —
never zeros, never filled from the future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from asx_quant.features.base import per_ticker


def _pct_change_per_ticker(series: pd.Series, tickers: pd.Series,
                           periods: int) -> pd.Series:
    """pct_change over ``periods`` sessions within each ticker."""
    grouped = series.groupby(tickers, sort=False)
    return grouped.pct_change(periods)


def compute_return_features(
    df: pd.DataFrame,
    windows: list[int],
) -> pd.DataFrame:
    """Trailing returns over configured windows on ADJUSTED close."""
    out = pd.DataFrame(index=df.index)
    adj = df["adjusted_close"]
    tickers = df["ticker"]
    for window in windows:
        out[f"return_{window}d"] = _pct_change_per_ticker(adj, tickers, window)
    return out


def compute_moving_average_features(
    df: pd.DataFrame,
    windows: list[int],
    slope_windows: list[int],
    crossovers: tuple[tuple[int, int], ...] = ((5, 20), (20, 50)),
) -> pd.DataFrame:
    """Price-to-SMA ratios, SMA crossover signals and SMA slopes."""
    out = pd.DataFrame(index=df.index)
    grouped = df.groupby("ticker", sort=False)["adjusted_close"]
    smas: dict[int, pd.Series] = {}
    for window in windows:
        smas[window] = grouped.transform(lambda s, w=window: s.rolling(w).mean())
        out[f"price_to_sma_{window}"] = df["adjusted_close"] / smas[window]
    for fast, slow in crossovers:
        if fast in smas and slow in smas:
            diff = smas[fast] - smas[slow]
            out[f"sma_cross_{fast}_{slow}"] = np.sign(diff).where(
                diff.notna(), np.nan
            )
    # Slope features are computed on the SMA(20) when available (the most
    # interpretable trend measure), else the longest configured window.
    base = 20 if 20 in smas else max(smas) if smas else None
    sma = smas.get(base) if base else None
    if sma is not None:
        for slope in slope_windows:
            out[f"sma_{base}_slope_{slope}d"] = sma.groupby(
                df["ticker"], sort=False
            ).pct_change(slope, fill_method=None)
    return out


def compute_volatility_features(
    df: pd.DataFrame,
    windows: list[int],
    atr_window: int,
    drawdown_window: int,
) -> pd.DataFrame:
    """Annualised rolling volatility, ATR, drawdown (beta added in market.py)."""
    out = pd.DataFrame(index=df.index)
    ret = df["daily_return"]
    grouped = ret.groupby(df["ticker"], sort=False)
    for window in windows:
        out[f"volatility_{window}d"] = grouped.transform(
            lambda s, w=window: s.rolling(w).std() * np.sqrt(252)
        )
    prev_close = df.groupby("ticker", sort=False)["close_price"].shift(1)
    tr = pd.concat([
        df["high_price"] - df["low_price"],
        (df["high_price"] - prev_close).abs(),
        (df["low_price"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    out[f"atr_{atr_window}"] = tr.groupby(df["ticker"], sort=False).transform(
        lambda s: s.rolling(atr_window).mean()
    )
    rolling_max = df.groupby("ticker", sort=False)["adjusted_close"].transform(
        lambda s: s.rolling(drawdown_window).max()
    )
    out[f"drawdown_{drawdown_window}d"] = (
        df["adjusted_close"] / rolling_max - 1.0
    )
    return out


def _rsi(close: pd.Series, window: int) -> pd.Series:
    """Wilder's RSI with EWM smoothing (backward-looking by construction).

    Edge cases: an unbroken up-trend (avg_loss == 0, avg_gain > 0) gives 100;
    an unbroken down-trend gives 0; the warm-up window and perfectly flat
    series give NaN.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False,
                        min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False,
                        min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    pure_gains = (avg_loss == 0) & (avg_gain > 0)
    rsi = rsi.where(~pure_gains, 100.0)
    pure_losses = (avg_gain == 0) & (avg_loss > 0)
    rsi = rsi.where(~pure_losses, 0.0)
    rsi[avg_gain.isna()] = np.nan  # warm-up
    return rsi


def _adx(high: pd.Series, low: pd.Series, close: pd.Series,
         window: int) -> pd.Series:
    """ADX (Wilder). Uses only past data; NaN during warm-up."""
    up = high.diff()
    down = -low.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / window, adjust=False,
                                  min_periods=window).mean() / atr
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / window, adjust=False,
                                    min_periods=window).mean() / atr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(
        0.0, np.nan)
    adx = dx.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    return adx


def compute_technical_features(
    df: pd.DataFrame,
    rsi_window: int,
    macd: dict,
    bollinger: dict,
    stochastic: dict,
    adx_window: int,
    recent_high_low_window: int,
) -> pd.DataFrame:
    """RSI, MACD, Bollinger %b/bandwidth, Stochastic %K/%D, ADX, 63d range."""

    def _tech(group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()
        close = g["adjusted_close"]
        high, low = g["high_price"], g["low_price"]

        g[f"rsi_{rsi_window}"] = _rsi(close, rsi_window)

        ema_fast = close.ewm(span=macd["fast"], adjust=False).mean()
        ema_slow = close.ewm(span=macd["slow"], adjust=False).mean()
        g["macd_line"] = ema_fast - ema_slow
        g["macd_signal"] = g["macd_line"].ewm(
            span=macd["signal"], adjust=False).mean()
        g["macd_hist"] = g["macd_line"] - g["macd_signal"]

        mid = close.rolling(bollinger["window"]).mean()
        std = close.rolling(bollinger["window"]).std()
        upper = mid + bollinger["n_std"] * std
        lower = mid - bollinger["n_std"] * std
        g["bollinger_pctb"] = (close - lower) / (upper - lower).replace(
            0.0, np.nan)
        g["bollinger_bandwidth"] = (upper - lower) / mid.replace(0.0, np.nan)

        low_n = low.rolling(stochastic["window"]).min()
        high_n = high.rolling(stochastic["window"]).max()
        g[f"stoch_k_{stochastic['window']}"] = (
            100.0 * (close - low_n) / (high_n - low_n).replace(0.0, np.nan)
        )
        g[f"stoch_d_{stochastic['window']}"] = g[
            f"stoch_k_{stochastic['window']}"
        ].rolling(stochastic["smooth"]).mean()

        g[f"adx_{adx_window}"] = _adx(high, low, close, adx_window)

        g[f"pct_from_{recent_high_low_window}d_high"] = (
            close / high.rolling(recent_high_low_window).max() - 1.0
        )
        g[f"pct_from_{recent_high_low_window}d_low"] = (
            close / low.rolling(recent_high_low_window).min() - 1.0
        )
        return g

    result = per_ticker(df, _tech)
    feature_cols = [
        f"rsi_{rsi_window}",
        "macd_line", "macd_signal", "macd_hist",
        "bollinger_pctb", "bollinger_bandwidth",
        f"stoch_k_{stochastic['window']}", f"stoch_d_{stochastic['window']}",
        f"adx_{adx_window}",
        f"pct_from_{recent_high_low_window}d_high",
        f"pct_from_{recent_high_low_window}d_low",
    ]
    return result[feature_cols]


def compute_volume_features(
    df: pd.DataFrame,
    windows: list[int],
    min_avg_dollar_turnover_aud: float,
) -> pd.DataFrame:
    """Volume ratios, dollar turnover and the liquidity eligibility flag."""
    out = pd.DataFrame(index=df.index)
    grouped = df.groupby("ticker", sort=False)["volume"]
    for window in windows:
        avg = grouped.transform(lambda s, w=window: s.rolling(w).mean())
        out[f"volume_ratio_{window}d"] = df["volume"] / avg.replace(0.0, np.nan)
    turnover = df["close_price"] * df["volume"]
    avg_turnover = turnover.groupby(df["ticker"], sort=False).transform(
        lambda s: s.rolling(max(windows)).mean()
    )
    out["dollar_turnover_20d"] = avg_turnover
    out["liquid_flag"] = (avg_turnover >= min_avg_dollar_turnover_aud).astype(
        float
    ).where(avg_turnover.notna(), np.nan)
    return out
