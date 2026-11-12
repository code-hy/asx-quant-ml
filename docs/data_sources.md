# Data Sources — Inventory, Terms and Limitations

**Version:** 0.1 (Phase 2)
**Budget constraint:** AUD $0 for data and software. Every source below is free
to access for research use. No source here is used for redistribution.

## 1. Source inventory

| # | Source | What it provides | Access method | Used for | License/terms notes |
|---|---|---|---|---|---|
| 1 | Yahoo Finance | Daily OHLC, Adjusted Close, volume, dividends, stock splits for ASX listings (`.AX` tickers); benchmark index `^AXJO` and ETF `VAS.AX` | `yfinance` library (public endpoints) | Market data (Bronze `prices`) | Yahoo ToS prohibit redistribution and commercial resale of the data. Use here is local, research-only analysis. **Do not republish raw data.** |
| 2 | Wikipedia — S&P/ASX 200 page | Current constituent list: ASX code, company name, GICS sector/industry | HTTPS GET of the public page, parsed with pandas (`lxml`) | Security universe / security master | Content under CC BY-SA 4.0 / GFDL; attribution required if republished. We only use it internally. |
| 3 | RBA statistical tables (planned — Phase 3) | Cash rate, bond yields, FX, commodity prices, financial aggregates | Public XLS/CSV endpoints | Macro features | RBA data is generally free to use with attribution; check specific table terms at ingest time. |
| 4 | ABS time series (planned — Phase 3) | CPI, labour force, WPI, retail trade | Public download endpoints | Macro features | ABS statistics are licensed CC BY 4.0 (attribution required). |
| 5 | ASIC short position reports (planned — Phase 3) | Daily net short position reports per stock | Public reports/downloads | Short-interest features | ASIC reports are public; reuse restrictions on aggregated republication apply — research use only. |
| 6 | ASX company announcements (optional — Phase 9+) | Headlines/categories of company announcements | To be assessed against ASX terms | Optional NLP extension | **Only if** access terms permit; no scraping that bypasses restrictions. Off by default. |

## 2. Method: universe construction (Phase 2)

`ingestion.universe.fetch_asx200_constituents()` retrieves the S&P/ASX 200
constituent table from Wikipedia and normalises it (valid 3-character ASX
codes, deduplicated). The curated seed list in `config/universe.yaml` is the
offline fallback if the fetch fails, and the fallback is recorded in the
ingestion report and security master (`universe_source` column).

**Disclosed limitation — survivorship bias.** Today's constituents are applied
across the whole historical period. Companies that left the index or delisted
(e.g. takeovers, collapses) are missing from history, which typically
*overstates* backtest performance versus a true point-in-time universe. This
is recorded per security in `inclusion_methodology` and must be restated in
every performance report. Historical constituent membership cannot be obtained
free-of-charge reliably; if a legitimate free source appears, it can replace
this methodology without code changes (config-driven).

## 3. Method: price ingestion (Phase 2)

`YahooDownloader` fetches daily bars in configurable batches (default 50)
with retries and pauses, then normalises each ticker's frame:

- Keeps raw `Close` **and** `Adj Close` (split + dividend adjusted).
- Records dividends per ex-date; split factor defaults to 1.0 on
  non-split sessions (canonical semantics — not silent null filling).
- Rows with all OHLCV fields missing (provider padding) are dropped and
  counted, never treated as data.
- Failed tickers are retried individually; remaining failures are listed in
  the ingestion summary and marked `no_price_data` in the security master.

Each ticker writes an **immutable Bronze batch** under
`data/bronze/prices/<ticker>/<run_id>.parquet` stamped with source,
`retrieved_at`, `ingestion_run_id` and `batch_id`; metadata is appended to
`data/bronze/prices/ingestion_log.jsonl`.

## 4. Data-budget status

| Requirement | Status |
|---|---|
| No paid market data | ✅ Yahoo Finance free endpoints |
| No paid constituent data | ✅ Wikipedia (with disclosed limitations) |
| No cloud subscriptions | ✅ Local Parquet + DuckDB |
| Reproducible without network | ✅ Seed-list fallback + offline test suite |
