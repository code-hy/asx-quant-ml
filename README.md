# ASX Quantitative Intelligence & Machine Learning Portfolio Platform

A research-only, end-to-end quantitative platform for Australian equities: free data
ingestion → lakehouse storage (Parquet + DuckDB) → feature engineering →
time-series-aware machine learning → long-only portfolio construction → backtesting
→ Streamlit research dashboard.

> **Not financial advice.** This is an educational project. It does not execute live
> trades, and its outputs are historical research signals, not trade instructions.

## Status — Phase 1 (Project Foundation) ✅

| Phase | Scope | Status |
|---|---|---|
| 1 | Project foundation: config, storage, logging, health check, docs | **complete** |
| 2 | Universe + historical market data ingestion | **complete** |
| 3 | RBA / ABS / ASIC macro & alternative data | **complete** |
| 4 | Feature engineering + target labels | **complete** |
| 5 | Model training (purged walk-forward CV) + OOS predictions | **complete** |
| 6 | Portfolio construction engine | planned |
| 7 | Backtesting framework | planned |
| 8 | Sensitivity analysis + robustness | planned |
| 9 | Streamlit dashboard + automation | planned |
| 10 | Final report + presentation | planned |

## Quick start

Requires Python **3.13** and [uv](https://docs.astral.sh/uv/).

```bash
# 1. Create the virtual environment and install pinned dependencies
uv sync --extra dev

# 2. Run the foundation health check
uv run asx_quant health

# 3. Initialise the data lake + DuckDB warehouse + run registry
uv run asx_quant initialise

# 4. Run the test suite
uv run pytest
```

All heavy lifting (data ingestion, training, backtesting) runs locally —
no API keys, no paid services.

## Configuration

Everything is configuration-driven via five YAML files in `config/`:

| File | Controls |
|---|---|
| `settings.yaml` | paths, storage conventions, ingestion behaviour, quality thresholds |
| `universe.yaml` | security universe methodology + benchmark definitions |
| `portfolio.yaml` | holdings, weighting method, constraints, transaction-cost defaults |
| `features.yaml` | enabled feature families, target engineering, leakage controls |
| `backtest.yaml` | execution assumptions, cost scenarios, sensitivity grid |

## Storage model

Medallion architecture (details in [`docs/architecture.md`](docs/architecture.md)):

```
data/
├── bronze/     # immutable raw source batches (prices, rba, abs, asic, ...)
├── silver/     # cleaned, conformed, deduplicated datasets
├── gold/       # analytical + ML datasets (features, labels, predictions)
├── models/     # serialised model artefacts + metadata
└── reports/    # data-quality reports, run logs, backtest results
```

Parquet is the durable system of record; DuckDB (`data/warehouse.duckdb`) is a
rebuildable query layer exposing registered views over the lake.

## CLI

```bash
uv run asx_quant health      # foundation self-check
uv run asx_quant initialise  # create dirs + warehouse + run registry
uv run asx_quant ingest --mode historical  # universe + 10y prices → Bronze
uv run asx_quant validate                  # data-quality checks + report
uv run asx_quant transform                 # Bronze → Silver + coverage report
# Phase 3+: features, train, predict, portfolio, backtest, report, update
```

## Data coverage (Phase 2-4 build)

- Universe: S&P/ASX 200 constituents fetched from Wikipedia (free source;
  survivorship bias documented in `docs/data_sources.md`)
- ~460,000 unique security-session records in Bronze/Silver
- Macro: RBA tables F2/F11/F11.1/G1/H5/I2 + cash-rate history ingested with
  point-in-time publication dates (silver `macro_observations`)
- Short interest: ASIC aggregated reports 2015→present, daily per stock
  (silver `short_positions`, publication-date aligned)
- Gold feature set (`gold/features.parquet`, ~50 features per
  ticker-session) and target labels (`gold/target_labels.parquet`,
  future_return_3d/10d with purge-ready availability dates) — see
  `docs/feature_catalogue.md`
- Walk-forward trained models (logistic + HistGBT classifier/regressor,
  purged/embargoed, held-out window 2024-07-01+) with OOS predictions in
  `gold/predictions.parquet` — see `docs/training_methodology.md`
- Per-ticker coverage, missing sessions and quality status in
  `data/reports/data_coverage.json`, `data/reports/data_quality.json`,
  `data/reports/data_quality_features.json` and
  `data/reports/model_quality.json`

## Repository layout

```
asx-quant-ml/
├── config/            # YAML configuration (all behaviour is config-driven)
├── data/              # bronze / silver / gold lake (gitignored)
├── docs/              # architecture, data model, dictionary, methodologies
├── notebooks/         # EDA + experiment notebooks (Phase 5+)
├── src/asx_quant/     # installable package (src layout)
├── dashboard/         # Streamlit multi-page research dashboard (Phase 9+)
├── tests/             # pytest suite
└── scripts/           # convenience entrypoints
```

## Research framing

The platform investigates (not assumes) eight hypotheses — momentum, mean
reversion, volume, volatility, market regime, sector behaviour, ASIC short
positions and portfolio construction — using chronological, leakage-controlled
validation. See `docs/model_methodology.md` (Phase 5+) for the full protocol.
