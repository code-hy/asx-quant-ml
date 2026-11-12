# Data Sources — Inventory, Terms and Limitations

**Version:** 0.1 (Phase 2)
**Budget constraint:** AUD $0 for data and software. Every source below is free
to access for research use. No source here is used for redistribution.

## 1. Source inventory

| # | Source | What it provides | Access method | Used for | License/terms notes |
|---|---|---|---|---|---|
| 1 | Yahoo Finance | Daily OHLC, Adjusted Close, volume, dividends, stock splits for ASX listings (`.AX` tickers); benchmark index `^AXJO` and ETF `VAS.AX` | `yfinance` library (public endpoints) | Market data (Bronze `prices`) | Yahoo ToS prohibit redistribution and commercial resale of the data. Use here is local, research-only analysis. **Do not republish raw data.** |
| 2 | Wikipedia — S&P/ASX 200 page | Current constituent list: ASX code, company name, GICS sector/industry | HTTPS GET of the public page, parsed with pandas (`lxml`) | Security universe / security master | Content under CC BY-SA 4.0 / GFDL; attribution required if republished. We only use it internally. |
| 3 | RBA statistical tables (F2, F11, F11.1, G1, H5, I2) | Cash-rate changes, government bond yields (daily), FX (daily from 2023 / monthly from 2010), CPI (quarterly, ABS data), labour force (monthly, ABS data), commodity prices (monthly) | Full-history CSV endpoints: `https://www.rba.gov.au/statistics/tables/csv/<table>-data.csv`; cash-rate history HTML at `/statistics/cash-rate/` | Macro features | RBA copyright: "You may use RBA data for your own purposes, including reproducing it, provided any republication attributes the Reserve Bank" — attribute `Source: RBA` on any derived publication. ABS-specified series (G1/H5) carry ABS CC BY 4.0 (attribution: `Source: ABS`, via RBA tables). |
| 4 | ABS time series (via RBA tables — see #3) | CPI (G1), labour force (H5) | Ingested through the RBA's free CSV endpoints (no ABS API key needed; documented workaround) | Macro features | ABS statistics are licensed CC BY 4.0 (attribution required). If a direct ABS download is preferred later, it is config-swappable. |
| 5 | ASIC aggregated short position reports | Daily aggregated net short positions per stock (% of product in issue), 2010→present | Report index `https://download.asic.gov.au/short-selling/short-selling-data.json` (4,150 report dates); year-to-date CSVs `RR{YYYYMMDD}-{v}-SSDailyYTD.csv` (one per year, latest version) | Short-interest features | ASIC data is public; research-only use, no aggregated republication. Disclaimer from ASIC: "No responsibility is accepted for any inaccuracies contained in the matter published." Daily reports are normally published the next business day. |
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

## 4. Method: macro ingestion (Phase 3)

`ingestion/rba.py` ingests the configured RBA full-history CSV tables
(configured in `config/settings.yaml` → `ingestion.macro.rba.tables`):
F2 (daily yields), F11/F11.1 (monthly/daily FX), G1 (CPI), H5 (labour force),
I2 (commodity index), plus the cash-rate change history from
`/statistics/cash-rate/`.

**Point-in-time publication rules (AC07)** — the highest-priority control in
the platform:

- The CSV `Publication date` metadata row is the CURRENT vintage only; it is
  never used as a per-row publication date.
- Daily series: `publication_date = observation_date` (`same_day_publication`).
- Monthly/quarterly series: `publication_date = observation_date + lag`
  (conservative, configurable: monthly 15 days, quarterly 35 days) and
  `potentially_revised = true` — joining must use `publication_date`.
- Cash rate: observations dated at the ANNOUNCEMENT date (the day before the
  new target becomes effective; announced 2:30pm AEST), `same_day_publication`,
  with the original effective date kept in bronze for provenance.
- Quality rule `publication_not_before_observation` is ERROR severity: any
  observation whose publication precedes its observation date fails the stage.

**Known limitation:** exact historical release dates for monthly/quarterly
series are not freely available in machine-readable form. The configured lags
are deliberately conservative — the join may make a value available a few
days LATER than it truly was, which is the safe direction (data is never
joined too early), and every such value is flagged
`estimated_publication_lag` + `potentially_revised`.

## 5. Method: ASIC short positions (Phase 3, AC08)

`ingestion/asic.py` fetches the report index JSON (4,150 report dates,
2010→present), picks the latest YTD report per year from the configured
`start_year`, parses each YTD pivot CSV, and computes each trade date's
`publication_date` as the earliest index report date ≥ that trade date
(`estimated_from_report_calendar`). YTD files are downloaded with retries and
a pause; per-year failures are logged and never fabricated.

**Two file vintages** (both handled by `parse_ytd_pivot`; detected by BOM and
delimiter, never guessed):

- Current (2022→): UTF-8, comma-separated; row 0 holds `dd/mm/yyyy` trade
  dates repeated per column pair; the `Product`/`Product Code` label row is
  row 1; data from row 2.
- Pre-2022 (→2021): UTF-16LE (BOM `FF FE`), tab-separated; ISO dates
  (`2018-11-30`) at pair-start columns with empty gap columns; a 4-row header
  block (title with `Trade Date`, dates, sub-headers, `Product`/`Product
  Code` label row); data from row 4.

Verified against a real pre-2022 file (`RR20181130-001-SSDailyYTD.csv`):
115,891 records, 1,063 product codes, dates 2018-01-02→2018-12-11. Product
names may be quoted and CONTAIN the delimiter (both vintages), so the parser
uses a proper CSV reader — naive splitting shifts columns on such rows.

**Known source anomaly:** a small number of rows (mostly exchange-traded
products — ETFs, bond trusts) report MORE than 100% of the product in issue
as short. This is genuine published ASIC data (the `product in issue`
denominator lags creations/redemptions for ETPs). Rows are kept verbatim;
the `pct_within_bounds` quality rule is a WARN with the count and samples
disclosed in the saved quality report — never silently dropped or corrected.
Negative percentages (structurally invalid) remain an ERROR. Some very
early report dates in the index may have no YTD file (404); such years are
logged as failed attempts and disclosed as a coverage gap.

## 6. Data-budget status

| Requirement | Status |
|---|---|
| No paid market data | ✅ Yahoo Finance free endpoints |
| No paid constituent data | ✅ Wikipedia (with disclosed limitations) |
| No cloud subscriptions | ✅ Local Parquet + DuckDB |
| Reproducible without network | ✅ Seed-list fallback + offline test suite |
