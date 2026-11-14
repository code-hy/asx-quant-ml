# Phase 10 Presentation — slide-by-slide deck

*Render with any Markdown-to-slides tool (e.g. Marp), or read as a
summary. Speaker notes are in block quotes.*

---

## Slide 1 — Title

**ASX Quantitative Intelligence & ML Portfolio Platform**
A research-only, end-to-end study of ML-driven stock selection on the
Australian market — free data, local-first, fully reproducible.

> Note up front: this is not financial advice. No real trades were ever
> placed. Everything is simulated on paper with honest frictions.

---

## Slide 2 — The question

> Can a machine-learning signal, built only from free data, support a
> profitable long-only portfolio of 20 ASX stocks rebalanced weekly?

Fixed frame: ASX 200 universe, 3-session prediction horizon, weekly
rebalance, AUD 0 data budget, Parquet + DuckDB lake, Python, Streamlit.

---

## Slide 3 — The pipeline (10 phases)

```
ingest → validate → transform → features → train → portfolio
       → backtest → sensitivity → report → update + dashboard
```

Every stage: quality rules, run-registry logging, 209 automated tests.

> The engineering contract: point-in-time correctness, no leverage, and
> "counted, never silent" for every skipped action.

---

## Slide 4 — Data: free and public only

- Wikipedia → 200-company universe
- Yahoo Finance → 460k daily price rows (2016–2026), dividends, splits
- RBA → interest rates, AUD/USD, yields
- ASIC → aggregate short positions

> Honest gaps documented: Yahoo is not exchange-official; no
> point-in-time sector map (that axis was skipped, not faked).

---

## Slide 5 — The ML task

Predict each stock's **return over the next 3 sessions** from
price/volume/volatility/momentum/macro/short-interest features.

Four models trained side-by-side with **purged, embargoed,
walk-forward** validation (25 folds). A **sealed window from
2024-07-01** was never used for any selection decision.

> The anti-cheating rule: every prediction is stamped
> `trained_through`, and the portfolio builder refuses stamps that are
> not strictly in the past.

---

## Slide 6 — Model results (evaluation folds only)

| model | metric | value |
|---|---|---|
| hist_gbt_reg | Spearman IC | **0.030** |
| ridge | Spearman IC | 0.007 |
| hist_gbt | ROC-AUC | 0.522 |
| logistic | ROC-AUC | 0.516 |

> 0.03 is a small edge — realistic for 3-day stock returns, where noise
> dominates. Selected: hist_gbt_reg.

---

## Slide 7 — The simulated portfolio (2020-11 → 2026-11)

| | Strategy | Market (VAS.AX) |
|---|---|---|
| Final NAV | AUD 159,395 | +79.0% |
| Total return | +65.3% | |
| CAGR | 8.7% | 10.1% |
| Max drawdown | −41.2% | — |
| Sharpe | 0.29 | — |
| Trades / costs | 7,571 / AUD 94,474 | ~0 |

> The strategy made money but lost to just holding the market. The
> mechanism is visible: AUD 94.5k of fees on 56% weekly turnover.

---

## Slide 8 — Why it lost: the cost drag

- Fixed AUD 9.50 brokerage × 7,571 trades = **AUD 94,474** (≈94% of
  starting capital)
- Average weekly turnover: 56.5% of NAV

> A 0.03 IC cannot pay weekly rent. This is exactly what honest
> backtesting exists to reveal.

---

## Slide 9 — Sensitivity: NOT a leaderboard (81 grid points)

Mean Sharpe by trading frequency: **monthly 0.90** / fortnightly 0.50 /
weekly −0.10. By size: 10 names 0.61 / 20 names 0.40 / 30 names 0.30.

- Trading less often doesn't create skill — it wastes less money.
- Fewer names helped on this one history (caveat: single sample).
- The base grid point reproduces the primary backtest to the cent
  (verified by the runner).

> Reporting all 81 points — never cherry-picking — is the point.

---

## Slide 10 — Integrity controls (the real deliverable)

- **Look-ahead:** machine-enforced execution lag; `trained_through`
  guards; purged/embargoed walk-forward; sealed held-out window.
- **No leverage:** cash never negative; unaffordable buys skipped and
  counted.
- **No fabrication:** untrained horizons and impossible axes recorded
  as skipped with reasons.
- **Reproducibility:** run registry, deterministic rebuilds, 209 tests.

---

## Slide 11 — What we would do next

1. Train the 10-day horizon (one config flip + rerun) — lower turnover
   may let the edge keep its fees.
2. Turnover-aware construction (hold until rank drops).
3. Percent-only cost scenario to bound small accounts.
4. Point-in-time sector mapping before any sector analysis.
5. Re-seal a fresh holdout before further tuning.

---

## Slide 12 — Closing

**Built:** a complete, honest, reproducible quant research platform.
**Found:** a small real edge that weekly trading costs consume.
**Lesson:** discipline beats optimism — the pipeline tells you the
truth about your idea, and that is the product.

```
uv run asx_quant report
uv run streamlit run dashboard/Home.py
```

> `docs/explainer.md` explains all of this from zero; `docs/final_report.md`
> carries the full technical detail.
