# Final Technical Report (Phase 10)

**Platform:** ASX Quantitative Intelligence & Machine Learning Portfolio
Platform — research-only, long-only, local-first.
**Author:** Henry. **Status:** Phases 1–10 complete. **Not financial
advice; no live trading; no real orders were ever placed.**

---

## 1. Executive summary

The platform investigates one primary research question end to end:

> Can a leakage-controlled machine-learning signal built from free data
> support a profitable, cost-aware, long-only portfolio of 20 ASX stocks
> rebalanced weekly on a 3-session horizon?

**Answer found: the pipeline works; the edge does not clear its costs.**

| | Strategy | Benchmark (VAS.AX) |
|---|---|---|
| Window | 2020-11-10 → 2026-11-12 (1,521 sessions) | same |
| Final NAV / return | AUD 159,395.37 (**+65.27%**) | +79.02% |
| CAGR | 8.68% | 10.13% |
| Volatility (annualised) | 24.4% | — |
| Sharpe ratio | 0.289 | — |
| Max drawdown | −41.18% (2021-11-18 → 2022-09-28, recovered 2025-06-04) | — |
| Trades / total costs | 7,571 / AUD 94,474 | ~0 |
| Held-out 2024-07-01+ (diagnostics only) | +39.86% (Sharpe 0.501) | +27.49% |

The mechanism of underperformance is identified, not guessed: **fixed
brokerage of AUD 9.50 per trade against 56.5% average weekly turnover
produced AUD 94,474 of costs** (~94% of initial capital), overwhelming a
small (evaluation Spearman IC 0.030) but genuine ranking signal. The
Phase 8 sensitivity grid confirms this mechanistically: identical
portfolios traded **monthly** average Sharpe 0.90 vs **weekly** −0.10 —
frequency changes costs, not skill.

All ten phases are delivered with test-enforced integrity controls
(209 tests passing): point-in-time eligibility, machine-enforced
execution lag, no-leverage cash accounting, sealed held-out window, and
a reproducing sensitivity core (`reference_check_vs_primary.matches =
true`).

---

## 2. Scope and fixed design decisions

| Decision | Value | Rationale |
|---|---|---|
| Market / universe | ASX, 200 companies | requested scope |
| Long-only, max holdings | 20 names | requested scope |
| Rebalance | weekly (Mondays, session-shifted) | requested scope |
| Horizon | 3 sessions primary / 10d secondary | requested scope |
| Data budget | AUD 0 (free sources only) | requested scope |
| Storage / stack | Parquet + DuckDB, Python 3.13, uv, Streamlit | local-first |
| Live trading | out of scope | research-only |

Constraints enforced everywhere: **no fabrication** (unrunnable analyses
are recorded as skipped with reasons, e.g. the 10d horizon was never
trained), **no look-ahead** (point-in-time eligibility +
`trained_through` guards), **no leverage** (buys exceeding cash are
skipped and counted), and **no silent behaviour** (every skip is a
diagnostic counter surfaced in quality rules).

---

## 3. Data (Phases 1–3)

* **Universe:** ASX 200 constituents from Wikipedia; 200 securities.
* **Prices:** Yahoo Finance daily OHLC + adjusted close + dividend +
  split records; silver lake holds 460,031 security-session rows
  2016-11-14 → 2026-11-12; open prices NaN share 0.0; benchmarks
  VAS.AX and IDX_AXJO ingested (IDX_AXJO is the fallback proxy).
* **Corporate actions:** 3,073 dividend rows; real splits confirmed in
  the lake (e.g. GMD.AX 2022-01-06 consolidation, factor 0.1).
* **Macro (RBA):** cash rate, AUD/USD, yields; noted limitation:
  `macro_aud_usd` is all-NaN before 2023 (source coverage), handled by
  imputation dropping the column — documented, not fabricated.
* **Short positions (ASIC):** aggregated net short positions,
  2015-present.
* **Quality gates:** bronze validation `pass` (3 warns), silver coverage
  report, per-security missing-session tracking. Known limitation:
  survivorship/point-in-time coverage of the Wikipedia constituent list
  is imperfect; documented in `docs/data_sources.md`.

## 4. Features and labels (Phase 4)

The gold feature set (`v0.1`) covers the research-plan hypotheses:
momentum (multi-window), mean reversion, volume, volatility (incl.
`volatility_20d` used in weighting), market regime, and ASIC short
interest. Target labels are forward 3-session direction (`direction_3d`)
and return (`future_return_3d`) plus a 10d variant, with an embargo.
Catalogue: `docs/feature_catalogue.md`; dictionary:
`docs/data_dictionary.md`.

## 5. Model training (Phase 5)

* **Protocol:** purged + embargoed walk-forward CV, **25 folds**
  (11 evaluation folds for model selection; 14 folds cover the sealed
  held-out window starting 2024-07-01). Selection used evaluation folds
  ONLY. Protocol: `docs/training_methodology.md`.
* **Model zoo v0.1:** `hist_gbt` / `logistic` (classifiers,
  direction_3d); `hist_gbt_reg` / `ridge` (regressors,
  future_return_3d). 283,957 OOS prediction rows per model,
  2020-11-06 → 2026-11-09 (the last 3 sessions lack complete labels).

| model | kind | evaluation metric | value |
|---|---|---|---|
| hist_gbt_reg | regressor | **Spearman IC** | **0.030** |
| ridge | regressor | Spearman IC | 0.007 |
| hist_gbt | classifier | ROC-AUC | 0.522 |
| logistic | classifier | ROC-AUC | 0.516 |

Held-out window (never used for selection): hist_gbt_reg Spearman
0.011, ridge 0.030 — consistent with a small, noisy edge.

* **Selection:** `hist_gbt_reg` (best evaluation rank correlation).
* **Known limitation:** only the 3d horizon was trained
  (`train.yaml target.horizon: 3`); the 10d axis of the sensitivity
  grid is recorded as skipped with an explicit reason (no
  fabricated substitute).

## 6. Portfolio construction (Phase 6)

Per Monday-rebalance (shifted to the next actual session when the ASX is
closed; schedule never precedes the first OOS prediction date):

1. point-in-time eligibility: prediction with `trained_through` strictly
   before the rebalance date, `liquid_flag == 1`, ≥250 sessions history;
2. rank by signal, take top **20**;
3. equal weighting with a **10%** per-name cap
   (`prefer_cash_over_violation: true` — underfilled slots stay cash);
4. turnover vs previous targets computed per rebalance (avg 56.5%).

314 rebalances were constructed. **Deliberate non-implementation:**
sector-weight constraints were skipped because no free point-in-time
sector mapping exists — documented as a warn rule rather than approximated
with current-day labels.

## 7. Backtest engine (Phase 7)

Mechanics (each test-enforced):

* execution ONLY at the next session after the signal close (default
  `next_session_open`; `next_session_close` supported), lag machine-
  checked (`execution_after_signal_date`);
* whole-share sizing off previous-close NAV;
* slippage buys pay up / sells receive down; costs from the active
  scenario (base: AUD 9.50 fixed + 0.05% + 0.10% slippage);
* sells before buys; buys exceeding cash SKIPPED (no leverage; cash
  never negative — enforced rule);
* dust rule (AUD 500 min trade value) applies to adjustments; **full
  exits always execute** (`forced_dust_exits` counter; Phase 8 fix —
  stranded dust positions broke the holdings cap on monthly 30-name
  scenarios);
* dividends credited to start-of-session holders (149 events); splits
  multiply quantities with cash-in-lieu (6 events, 2 cash-in-lieu);
* suspended tickers carried at last close (1 suspended valuation day);
* diagnostics: 314/314 rebalances executed, 2,051 dust skips,
  108 cash-floor skips, 0 no-price skips — all counted, surfaced as warn
  rules;
* benchmark = VAS.AX adjusted_close total-return proxy (IDX_AXJO
  fallback); held-out window (2024-07-01+) metrics computed and reported
  separately, never used for selection.

*Curve-start nuance (stated for full transparency):* the equity curve
begins on the first effective session (2020-11-10) with the portfolio
already deployed at that day's prices — first recorded NAV 96,447 —
while the benchmark series starts at exactly 100,000 on the same close.
The strategy's +65.27% therefore includes the first-day entry effect,
and the benchmark's +79.02% does not; comparing final values on the
strategy's own base (96,447 × 1.790155 ≈ 172,655) does not change the
qualitative conclusion (strategy below benchmark).

Engine quality: `pass` (3 warns). Cost drag: total costs AUD 94,474.44
= 94.5% of initial capital; avg turnover 56.5% per rebalance; hit rate
53.1%; best/worst day +13.1% / −7.6%.

## 8. Sensitivity & robustness (Phase 8)

Grid: portfolio sizes {10, 20, 30} × frequencies {weekly, fortnightly,
monthly} × caps {0.05, 0.10, 0.15} × cost scenarios {optimistic, base,
conservative} = **81 points**, all `status=ok`, run in memory over the
same pure construction core; the base grid point **reproduces the
primary backtest exactly** (NAV 159,395.37 both sides).

Findings (read as diagnostics, NOT a leaderboard — the report file says
so explicitly):

| dimension | mean Sharpe | reading |
|---|---|---|
| monthly | 0.903 | less cost churn, same signal |
| fortnightly | 0.497 | — |
| weekly | −0.096 | fixed-cost drag dominates |
| 10 names | 0.613 | concentrated size helps this history |
| 20 names | 0.395 | staged configuration |
| 30 names | 0.296 | dilution + more trades |
| cost scenarios | optimistic > base > conservative | sanity ranking holds |

Skipped axes (documented, never fabricated): horizon **10d** (no model
trained — flip `train.yaml target.horizon` and rerun `train` to
evaluate), **sector weights** (no point-in-time mapping). Full range:
total return −89.99%…+476.58%; Sharpe −2.07…1.19; maxDD −91.65%…−23.94%.

## 9. Dashboard & automation (Phase 9)

* `report` command consolidates artifact presence, quality statuses,
  staleness and headline metrics into `data/reports/platform_summary.json`
  (read-only).
* `update` command chains ingest → validate → transform → features →
  train → portfolio → backtest (optionally sensitivity), aborting at the
  first failure and refreshing the snapshot last; restartable by rerun.
* Streamlit dashboard (5 pages, AppTest-verified on empty, populated and
  real lakes): Home, Backtest, Portfolio, Sensitivity, Data Quality —
  read-only, disclaimer on every page.
* `scripts/update.ps1` + scheduled-task recipe in `docs/automation.md`.

## 10. Honest limitations

1. **The strategy loses to its benchmark** over the full window
   (+65.3% vs +79.0%), chiefly through cost drag; the 0.030 IC does not
   pay for weekly turnover at base costs.
2. **One market history.** All results come from 2020–2026; the
   sensitivity grid varies construction, not regimes. No claim of
   out-of-sample generality beyond the sealed window.
3. **Free-data limitations:** Yahoo Finance (not exchange-official),
   Wikipedia constituents (survivorship + imperfect point-in-time
   membership), RBA macro gaps pre-2023, no point-in-time sector map
   (axis skipped).
4. **Single horizon trained** (3d); 10d unevaluated — skipped, not
   substituted.
5. **Selection-lock discipline:** the held-out window has now been
   reported; anyone continuing the research must re-seal a new window
   rather than re-tune against the reported one.
6. **Equal-weight + cap** is the only weighting implemented; the
   volatility-adjusted method exists in config but was not selected.
7. **Slippage is modelled**, not measured (no free order-book data).

## 11. Conclusions and recommended next steps

The durable deliverable is the **discipline**: a reproducible,
leakage-controlled, test-enforced research pipeline on free data. The
empirical deliverable is the honest negative: a small real edge, at
weekly frequency and base costs, does not clear its friction.

If this research continued, the evidence points to:

1. retrain with `target.horizon: 10` and evaluate the weekly schedule
   against lower-frequency variants (the skip is one config flip away);
2. test turnover-aware construction (e.g. hold-until-rank-drops)
   against the cost model — the monthly/weekly Sharpe gap suggests most
   of the "edge" consumed itself in fees;
3. replace fixed-brokerage sensitivity with a percent-only scenario to
   bound the small-account case;
4. source a point-in-time sector mapping before implementing the sector
   axis (or keep it skipped);
5. re-seal a fresh held-out window before any further model iteration.

---

## 12. Deliverables index

| Artifact | Location |
|---|---|
| Pipeline | `src/asx_quant/` (ingest, validate, transform, features, train, portfolio, backtest, sensitivity, report, update) |
| CLI | `uv run asx_quant <command>` |
| Dashboard | `uv run streamlit run dashboard/Home.py` |
| Tests | `tests/` — 209 passing |
| Quality reports | `data/reports/*.json` |
| Backtest artifacts | `data/gold/backtest_equity_curve.parquet`, `backtest_trades.parquet` |
| Sensitivity table | `data/reports/sensitivity_results.parquet` + `sensitivity_report.json` |
| Platform snapshot | `data/reports/platform_summary.json` |
| Run history | `data/reports/runs/run_log.jsonl` |
| Docs | `docs/` (architecture, data model, data sources, dictionary, feature catalogue, training methodology, automation, **explainer**, final report, presentation) |
