# Feature Catalogue

**Feature set version:** `v0.1` (Phase 4)
**Code:** `src/asx_quant/features/` — enabled families and windows come from
`config/features.yaml`; this document is the authoritative human-readable
description of every column in `gold/features.parquet`.

---

## Conventions

- **Grain:** one row per (universe ticker, ASX session). Benchmark index rows
  (`IDX_*`) are used to compute market context but receive no feature row.
- **Session basis:** every window is counted in TRADING sessions, never
  calendar days.
- **Price basis for returns/momentum:** ADJUSTED close (split+dividend
  adjusted), same convention as targets.
- **Price basis for OHLC-derived features (ATR, Bollinger, stochastic,
  ranges):** raw close/high/low, as traded.
- **Null policy:** `null_as_missing` — rolling warm-ups and publication gaps
  are NaN. Values are never filled with 0 and never forward-filled from the
  future (macro/short point-in-time carriers are forward-filled BY
  PUBLICATION, see below — that fill carries only information already
  public).
- **Leakage controls (enforced by tests):**
  1. Ticker features at session `t` use only rows with date `<= t`
     (regression-tested: truncated recomputation must be identical).
  2. Macro/short values appear in a row only when
     `publication_date <= row date` (as-of backward join).
  3. Targets live in a SEPARATE dataset and carry `label_available_date`
     for purge/embargo.

## Returns (on adjusted close)

| Column | Definition |
|---|---|
| `return_{w}d` | `adj_close[t] / adj_close[t-w] - 1`, w ∈ {1, 3, 5, 10, 20, 60, 120, 252} |

## Momentum

| Column | Definition |
|---|---|
| `momentum_rel_bench_20d` | `return_20d(stock) - return_20d(benchmark)` |
| `momentum_rel_bench_63d` | `return_60d(stock) - return_20d(benchmark)` |
| `momentum_ts_pctile_63d` | percentile of today's `return_20d` within the stock's own trailing 63 sessions |
| `momentum_xsec_pctile_20d` | cross-sectional rank (0–1) of `return_20d` among the universe on the same session |

**Limitation:** `relative_to: sector` is declared in config but NOT
implemented — no free, point-in-time sector mapping is currently ingested.
Sector momentum requires a sector source (GICS via ASX is paid; a static
Wikipedia mapping would introduce survivorship/mapping bias). Documented
gap, not silently ignored.

## Moving averages (on adjusted close)

| Column | Definition |
|---|---|
| `price_to_sma_{w}` | `adj_close[t] / SMA_w(t)`, w ∈ {5, 10, 20, 50, 100, 200} |
| `sma_cross_{fast}_{slow}` | sign(SMA_fast − SMA_slow): +1/0/−1 for (5,20) and (20,50) |
| `sma_20_slope_{k}d` | `SMA_20(t) / SMA_20(t-k) - 1`, k ∈ {5, 20} |

## Volatility

| Column | Definition |
|---|---|
| `volatility_{w}d` | rolling std of daily returns × √252, w ∈ {5, 10, 20, 60} |
| `atr_14` | 14-session simple average of true range (Wilder's EWM variant used for ADX only; documented simplification) |
| `drawdown_252d` | `adj_close[t] / rolling_max(adj_close, 252) - 1` (≤ 0) |
| `beta_252d` | rolling covariance(stock, benchmark) / variance(benchmark) over 252 sessions |

## Technical (raw OHLC)

| Column | Definition |
|---|---|
| `rsi_14` | Wilder RSI, EWM smoothing (pure up-trend → 100, pure down-trend → 0) |
| `macd_line`, `macd_signal`, `macd_hist` | EMA12−EMA26; EMA9 of MACD; their difference |
| `bollinger_pctb` | `%b` = (close − lower) / (upper − lower), SMA20 ± 2σ |
| `bollinger_bandwidth` | (upper − lower) / SMA20 |
| `stoch_k_14`, `stoch_d_14` | raw and 3-session-smoothed %K |
| `adx_14` | Wilder ADX (EWM smoothing) |
| `pct_from_63d_high` / `pct_from_63d_low` | `close / rolling_high(63) - 1` (≤ 0) / `close / rolling_low(63) - 1` (≥ 0) |

## Volume / liquidity

| Column | Definition |
|---|---|
| `volume_ratio_{w}d` | `volume[t] / SMA_w(volume)`, w ∈ {5, 20} |
| `dollar_turnover_20d` | 20-session average of `close × volume` (AUD) |
| `turnover_xsec_pctile` | cross-sectional rank (0–1) of dollar turnover, same session |
| `liquid_flag` | 1 if `dollar_turnover_20d ≥ 250,000 AUD` (config), else 0; NaN during warm-up |

## Market context (broadcast to every ticker by session)

| Column | Definition |
|---|---|
| `bench_return_1d` / `bench_return_{w}d` | benchmark (^AXJO) session return / trailing w-session return, w ∈ {20, 100} |
| `bench_volatility_20d` | benchmark 20-session annualised volatility |
| `bench_close_to_sma_{w}` | benchmark close / its SMA, w ∈ {20, 100} (regime) |
| `breadth_above_sma_{w}` | fraction of the universe with `price_to_sma_{w} > 1`, w ∈ {20, 50} |

## Macro (point-in-time on publication_date)

| Column | Definition |
|---|---|
| `macro_{series}` | value of the series AS KNOWN at the session (as-of backward join on `publication_date` over the trading calendar; constant between publications). Series: `rba_cash_rate`, `au_gov_yield_2y/3y/10y`, `aud_usd`, `abs_cpi`, `abs_unemployment_rate`, `rba_commodity_index` |
| `macro_curve_slope_10y_2y` | `macro_au_gov_yield_10y − macro_au_gov_yield_2y` |

Join rule: a value dated for observation period `P` with publication date
`pub` enters features from the first session `>= pub`. Monthly/quarterly
series carry conservative estimated lags (15/35 days) and are flagged
`potentially_revised` in Silver — the join may make them available a few
days LATER than in reality, never earlier (safe direction for leakage).

## Short interest (point-in-time on publication_date)

| Column | Definition |
|---|---|
| `short_pct` | ASIC `% of product in issue reported short`, as known at the session (as-of backward join on `publication_date`, forward-filled per ticker between publications) |
| `short_positions` | ASIC aggregated net short position (units), same carrier |
| `short_pct_change_5d` | `short_pct[t] − short_pct[t-5]` on the filled grid |
| `short_xsec_pctile` | cross-sectional rank (0–1) of `short_pct`, same session |

A position reported for trade date `T` becomes usable only from its
publication (normally the next ASIC report); on the trade date itself the
feature is NaN — asserted by tests.

## Target labels (`gold/target_labels.parquet` — SEPARATE dataset)

| Column | Definition |
|---|---|
| `future_return_{h}d` | `adj_close[t+h] / adj_close[t] - 1`, h ∈ {3 (primary), 10 (secondary)} |
| `direction_{h}d` | 1 if `future_return_{h}d > 0` else 0 (NaN where the label is incomplete) |
| `label_available_date_{h}d` | the session whose close completes the label — the purge/embargo key for train/test splits |

**Embargo:** `targets.embargo_sessions = 3`. When splitting, samples whose
label window overlaps the evaluation boundary must be purged; the
availability date is the machine-checkable key for that (Phase 5+).
