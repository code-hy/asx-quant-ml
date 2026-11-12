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
| 2 | Universe + historical market data ingestion | planned |
| 3 | RBA / ABS / ASIC macro & alternative data | planned |
| 4 | Gold model-ready features + targets | planned |
| 5 | EDA + baseline models | planned |
| 6 | Advanced ML (Random Forest, XGBoost, LightGBM) | planned |
| 7 | Portfolio construction engine | planned |
| 8 | Backtesting framework | planned |
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
# Phase 2+: ingest, validate, transform, features, train, predict,
#           portfolio, backtest, report, update
```

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
