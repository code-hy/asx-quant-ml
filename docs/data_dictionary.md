# Data Dictionary

**Version:** 0.1 (Phase 1 — core entities; feature columns are catalogued
separately in `docs/feature_catalogue.md` from Phase 4).

Conventions: all timestamps are **UTC ISO-8601**; all dates are timezone-free
`DATE`; prices are **AUD**; identifiers are lowercase snake_case column names.

## DIM_SECURITY

| Column | Type | Nullable | Description |
|---|---|---|---|
| `security_key` | string | no | Surrogate key (stable across ticker changes). |
| `ticker` | string | no | ASX code with `.AX` suffix (Yahoo convention). |
| `company_name` | string | yes | Registered company name where available. |
| `exchange` | string | no | Always `ASX`. |
| `sector` | string | yes | Business sector label (source-permitting). |
| `industry` | string | yes | Industry label (source-permitting). |
| `listing_date` | date | yes | First listing date if available free. |
| `delisting_date` | date | yes | Null for active securities. |
| `currency` | string | no | `AUD`. |
| `instrument_type` | string | no | `ordinary_share`. |
| `active_flag` | bool | no | True while tradable. |
| `source_identifier` | string | no | Source-native ticker (yfinance). |
| `inclusion_methodology` | string | no | How the security entered the universe, e.g. `current_constituent_proxy`. |
| `data_quality_status` | string | no | `ok | flagged | unusable`. |
| `first_observed_date` / `last_observed_date` | date | yes | Coverage bookkeeping from ingestion. |

## DIM_DATE

| Column | Type | Description |
|---|---|---|
| `date_key` | int | `YYYYMMDD`. |
| `calendar_date` | date | Calendar date. |
| `calendar_year` / `calendar_month` / `calendar_quarter` | int | Calendar decomposition. |
| `day_of_week` | int | 1=Monday … 7=Sunday (ISO). |
| `is_asx_trading_day` | bool | True for ASX trading sessions. |
| `trading_day_sequence` | int | Dense index over ASX sessions (feature/target offset basis). |
| `financial_year` | int | Australian financial year (starts 1 July). |

## FACT_EQUITY_PRICE

Grain: one security × one ASX trading session.

| Column | Type | Nullable | Description |
|---|---|---|---|
| `security_key` | string | no | FK → `DIM_SECURITY`. |
| `date_key` | int | no | FK → `DIM_DATE`. |
| `open_price`, `high_price`, `low_price`, `close_price` | decimal | yes | Raw (unadjusted) prices. `high ≥ low` enforced by quality rules. |
| `adjusted_close` | decimal | yes | Split + dividend adjusted close (Yahoo `Adj Close`). |
| `volume` | bigint | yes | Shares traded; ≥ 0. |
| `dividend_amount` | decimal | no (default 0) | Cash dividend per share with ex-date on this session. |
| `split_factor` | decimal | no (default 1.0) | Split ratio applied on this session. |
| `daily_return` | decimal | yes | Close-to-close return on **adjusted** close; null on first session. |
| `source_identifier` | string | no | e.g. `yfinance`. |
| `ingestion_run_id` | string | no | FK → run registry. |

## FACT_MACROECONOMIC

Grain: one series × one observation × one publication.

| Column | Type | Description |
|---|---|---|
| `macro_series_key` | string | e.g. `rba_cash_rate`, `abs_cpi_qtr`. |
| `observation_date` | date | Reference period the value describes. |
| `publication_date` | date | **Point-in-time join key** — earliest date the value was public. |
| `value` | decimal | Observed value. |
| `unit` | string | `%`, `index`, `AUD`, … |
| `frequency` | string | `daily | monthly | quarterly`. |
| `source` | string | `rba | abs`. |
| `retrieved_at` | timestamp | UTC ingestion time. |
| `vintage_or_revision_flag` | string | `final` or `potentially_revised` (look-ahead risk disclosure). |

## FACT_SHORT_POSITION

Grain: one security × one ASIC reporting period.

| Column | Type | Description |
|---|---|---|
| `security_key` | string | FK → `DIM_SECURITY`. |
| `reporting_date` | date | Position reference date (short-sold as at). |
| `publication_date` | date | ASIC publication date — **the only usable join key**. |
| `reported_short_position` | decimal | Net short position in units. |
| `percentage_of_issued_capital` | decimal | % of issued capital. |
| `source` | string | `ASIC`. |

## FACT_ANNOUNCEMENT (optional extension)

| Column | Type | Description |
|---|---|---|
| `announcement_key` | string | Surrogate. |
| `security_key` | string | FK. |
| `publication_timestamp` | timestamp | ASX release time (UTC). |
| `announcement_category` | string | Earnings, trading update, capital raising, … |
| `headline` | string | Title only (no redistribution of protected text). |
| `source_url` | string | Link to the public source. |
| `text_reference` | string | Local reference to stored text, if permission permits. |
| `sentiment_score` | float | Nullable; optional NLP extension. |
| `source_permission_status` | string | Recorded terms status for the source. |

## FACT_FEATURES

Grain: one security × one prediction date × feature-set version. One **column
per feature** (families grouped for readability here; see
`docs/feature_catalogue.md`). All features use information available at or
before the prediction timestamp.

| Family (representative columns) | Type | Notes |
|---|---|---|
| `return_1d … return_252d` | float | Session-offset returns on adjusted close. |
| `sma_5 … sma_200`, `price_to_sma_*`, `sma_slope_*` | float | Moving-average family. |
| `vol_5d … vol_60d`, `atr_14`, `downside_vol_20d`, `beta_252d` | float | Risk family. |
| `rsi_14`, `macd`, `macd_signal`, `boll_pos_20_2`, `stoch_14_3`, `adx_14`, `obv` | float | Technical family. |
| `avg_volume_5d`, `avg_dollar_turnover_20d`, `volume_ratio`, `liquidity_flag` | float/bool | Liquidity family. |
| `benchmark_return_1d`, `sector_return_20d`, `pct_above_ma20`, `regime_*` | float | Market family. |
| `cash_rate_latest`, `yield_3y_10y_spread`, `cpi_yoy_latest` | float | Macro family (publication-date aligned). |
| `short_pct_issued_capital`, `short_change_1p`, `days_since_publication` | float | Short-interest family (publication-date aligned). |
| `ingestion_run_id`, `feature_set_version` | string | Lineage. |

## FACT_PREDICTIONS

| Column | Type | Description |
|---|---|---|
| `security_key`, `prediction_date`, `model_version` | — | Composite key. |
| `predicted_return_3d` / `predicted_return_10d` | float | Model return forecasts (primary signal = 3d). |
| `predicted_probability_positive` | float | Classifier output where applicable. |
| `predicted_rank` | int | Cross-sectional rank on the prediction date (1 = best). |
| `prediction_timestamp` | timestamp | When computed (must precede any use in portfolio construction). |

## FACT_PORTFOLIO_POSITION

| Column | Type | Description |
|---|---|---|
| `rebalance_date` | date | Rebalance decision date (session t). |
| `security_key` | string | Selected security. |
| `predicted_rank` | int | Rank used for selection. |
| `target_weight` | decimal | Pre-constraint model weight. |
| `actual_weight` | decimal | Post-constraint executed weight. |
| `execution_price` | decimal | Price used at session t+1 execution. |
| `shares_or_units` | bigint | Whole-share lots (fractional off by default). |
| `transaction_cost` | decimal | Brokerage + slippage for this position change. |
| `portfolio_id` | string | FK → `DIM_PORTFOLIO`. |
| `rejection_reason` | string | Why an excluded security was not held (nullable). |

## FACT_BACKTEST_DAILY

| Column | Type | Description |
|---|---|---|
| `portfolio_id` | string | FK. |
| `date` | date | ASX session. |
| `portfolio_value` | decimal | End-of-session value (AUD). |
| `daily_return` | decimal | Session return incl. costs. |
| `cumulative_return` | decimal | Growth of 1. |
| `benchmark_return` | decimal | Same-session benchmark return. |
| `cash_weight` | decimal | Uninvested fraction. |
| `drawdown` | decimal | Drawdown from running peak. |
| `transaction_cost` | decimal | Costs booked this session. |

## Shared audit columns

Every Parquet dataset written by `ParquetWriter`:

| Column | Type | Description |
|---|---|---|
| `ingestion_run_id` | string | Pipeline run that produced/loaded the data. |
| `retrieved_at` | timestamp | UTC retrieval/writing time. |
