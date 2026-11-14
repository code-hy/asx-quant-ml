# The Ten Phases, Explained From Zero

*This document walks through every implementation phase of the platform:
what each phase was trying to do, in plain words how it does it, how it
depends on the phase before it, and what it leaves behind for the phase
after it. It assumes **no prior knowledge of stock-market trading,
machine learning or statistical analysis** — every specialist term is
explained where it first appears. (A companion document,
[`explainer.md`](explainer.md), explains the finished whole; this one
explains the construction journey, phase by phase.)*

> **Not financial advice.** This is a research software project. Nothing
> described here places real trades or recommends any investment.

---

## The story in one paragraph

We wanted to answer a question — *can a computer program learn, from
free public data, which of 200 Australian stocks will rise over the next
few days well enough to beat simply holding the market?* — **honestly**.
"Honestly" turned out to be the hard part: it is frighteningly easy to
accidentally cheat (let the software peek at the future, trade at
yesterday's prices, ignore fees) and get a beautiful fake result. So the
project was built as ten careful layers: first a trustworthy storage
skeleton (Phase 1), then trustworthy data (2–3), then honest questions
to ask of that data (4), then honest training and predictions (5), then
honest shopping-list building (6), then honest play-through of history
with real-world frictions (7), then stress-testing the conclusions (8),
then windows into the results (9), and finally a written verdict (10).
Each layer only trusts the layers beneath it.

```
Phase 1   foundation        "build an empty, trustworthy house"
Phase 2   market data       "move the raw facts in"
Phase 3   macro/alt data    "move the context in"
Phase 4   features/labels   "turn raw facts into answerable questions"
Phase 5   ML training       "answer the questions without peeking"
Phase 6   portfolio         "turn answers into a shopping list"
Phase 7   backtest          "play the shopping lists through history"
Phase 8   sensitivity       "dare the conclusions to survive stress"
Phase 9   dashboard/ops     "let a human watch it work"
Phase 10  report            "write down the truth, even when unflattering"
```

---

# Phase 1 — Project Foundation

**Commit:** `169e80a` · **Modules:** `config/`, `storage/`, `observability/`, `pipeline/health.py`, `cli.py`

### Objective
Before touching any market data, build the **empty but trustworthy
skeleton**: a place to store data, a place to record what the software
did, a way to configure behaviour, a command-line entry point, and a
self-check that proves the plumbing works. Analogy: before renovating a
kitchen, pour the foundations and wire the circuits — nothing visible,
everything load-bearing.

### Concepts introduced
* **Data lake / medallion layers** — data is stored in three stages of
  trust: **bronze** (raw, exactly as downloaded, warts included),
  **silver** (cleaned, standardised), **gold** (analysis-ready). Like
  raw ingredients → washed and chopped → cooked dishes.
* **Parquet** — a file format that stores tables on disk compactly and
  reads them fast. Think "a very efficient Excel file that computers
  love".
* **DuckDB** — a small database engine that can query those Parquet
  files with SQL (the standard database-question language) for
  record-keeping.
* **Run registry** — an append-only diary (`run_log.jsonl`) where every
  stage records: who ran, when, on what data, with what result. Like a
  lab notebook that can never be edited, only appended.
* **CLI (command-line interface)** — the typed commands
  (`uv run asx_quant ingest`) that drive the platform.
* **Config-driven behaviour** — all important numbers (universe size,
  rebalance weekday, cost assumptions) live in YAML text files under
  `config/`, not buried in code. Change behaviour by editing text, not
  rewriting software.

### How it achieves it
Six small YAML files define the platform's contract (settings paths,
universe, features, training, portfolio, backtest). A `DataLake` object
knows every storage path; a Parquet writer and a DuckDB wrapper handle
reading/writing; the run registry appends a diary line per stage; the
CLI wires subcommands; a **health check** (`uv run asx_quant health`)
runs five self-probes (directories exist, warehouse opens, config loads,
logging works, registry appends) and fails loudly if anything is wrong.

### How it builds on nothing — and prepares Phase 2
This is the first layer, but it deliberately anticipates the rest: the
bronze/silver/gold directory skeleton *is* where Phase 2 will drop
market data; the run registry *is* what every later phase will record
itself in; the quality-check framework (a rule = name + severity +
pass/fail + count of failures) is the template Phases 2–8 will reuse;
and the CLI has placeholder commands for future phases that **refuse to
run** (exit code 2, "planned for a later phase — do not assume this
command has run") rather than pretending. Preparing for Phase 2 meant
making "write a batch of data, then check it" a one-line operation.

---

# Phase 2 — Market Data Ingestion (Universe + Prices)

**Commit:** `6d303cd` · **Modules:** `ingestion/universe.py`, `ingestion/yahoo.py`, `quality/`, `transformations/`

### Objective
Answer two factual questions and store the answers: *which 200
Australian companies do we study?* and *what did their shares do every
day since 2016?* — and make sure the answers are **clean enough to
trust**.

### Concepts introduced
* **Universe / security master** — the official list of companies
  (ticker, name, listing dates) that defines the study population.
* **Ticker** — a stock's short code (`BHP.AX`). The `.AX` suffix tells
  Yahoo Finance "this is the Australian exchange".
* **OHLC prices** — for each stock each day: **O**pening price (first
  trade), **H**ighest, **L**owest, **C**losing price (last trade), plus
  volume (how many shares traded).
* **Adjusted close** — the closing price *rewritten as if dividends and
  splits never removed value*, so that a price series is comparable
  across years. Essential for measuring returns.
* **Corporate actions** — events that mechanically change prices:
  **dividends** (cash paid to shareholders) and **splits/consolidations**
  (one share becomes N shares, or N become one; the price changes by
  the same factor but your wealth doesn't).
* **Data quality rules** — automated yes/no checks: no negative prices,
  no duplicated (ticker, date) rows, no prices beyond today, opening
  price inside the day's high-low range, and so on. Each rule has a
  **severity**: *error* (stop the pipeline) or *warn* (record and
  continue).
* **Coverage** — how much of the intended data actually arrived (how
  many securities, how many days, which tickers have holes).

### How it achieves it
1. **Universe:** read the Wikipedia table of ASX-200 constituents and
   normalise it into a bronze `security_master` table.
2. **Prices:** for each of the 200 tickers (plus market benchmarks like
   `VAS.AX`), download daily history from Yahoo Finance (a free public
   service) into bronze Parquet batches.
3. **Validate:** run the quality rules over bronze; anything
   error-severity stops the pipeline.
4. **Transform to silver:** standardise currencies/dates, build the
   **trading calendar** (the authoritative list of ASX sessions), record
   dividend amounts and split factors per row, and write a coverage
   report.

Result: 460,031 security-day price rows spanning 2016-11-14 →
2026-11-12, with 3,073 dividend records and real split events, and a
coverage report that names every security with missing sessions.

### How it builds on Phase 1 — and prepares Phase 3
It writes into the Phase 1 bronze/silver skeleton using the Phase 1
writer, registry and quality framework — no new plumbing needed. Two
honest limitations are documented rather than hidden: the Wikipedia list
reflects the **current** ASX-200 (some companies that existed in 2016
but were later demoted are absent — a so-called **survivorship** gap),
and Yahoo prices are good but not exchange-official. It prepares Phase 3
by proving the "ingest → validate → transform" pattern that the macro
and short-position sources will reuse, and by leaving the trading
calendar that every later date computation depends on.

---

# Phase 3 — Macro & Alternative Data (RBA / ABS / ASIC)

**Commit:** `9c29143` · **Modules:** `ingestion/rba.py`, `ingestion/asic.py`, `transformations/macro.py`

### Objective
Add the **context** around the stock prices: what the Australian economy
and the market's sentiment were doing — interest rates, inflation,
unemployment, currency, and how heavily investors were betting *against*
each stock.

### Concepts introduced
* **Macro data** — economy-wide statistics published by official bodies:
  the **RBA** (Reserve Bank of Australia, the central bank) publishes
  the cash rate (its key interest rate) and government bond **yields**
  (the interest the government pays to borrow for 2, 3, 10 years); the
  **ABS** (Australian Bureau of Statistics) publishes **CPI**
  (inflation: how much a basket of goods costs) and the unemployment
  rate. Currency markets give the **AUD/USD** exchange rate.
* **Alternative data** — non-price information that might hint at future
  moves. This project uses **ASIC short-position reports**: ASIC (the
  corporate regulator) publishes daily how many shares of each company
  investors have *sold short* — i.e. borrowed and sold hoping to buy
  back cheaper. Heavy shorting signals skepticism about a company.
* **Point-in-time publication dates** — the most important statistical
  idea in this phase. A statistic is only *knowable* when it was
  **published**, not when it *describes*. April's CPI might be published
  in late May. If software pretended to "know" April's CPI on April 15,
  it would be time travel — cheating. So each macro/short row stores
  both the **reference date** (what it describes) and the **publication
  date** (when the world could first see it).

### How it achieves it
The same ingest → validate → transform pattern from Phase 2, applied to
three new sources: RBA tables (scraped from the RBA's public files),
ABS series, and ASIC's aggregated short-position reports, each landing
in bronze with quality checks, then cleaned into silver with
publication-date columns attached.

### How it builds on Phase 2 — and prepares Phase 4
It reuses Phase 2's ingestion machinery and quality rules verbatim (that
was the point of building them), and extends the trading-calendar work
by aligning monthly/quarterly statistics onto daily rows. It documents
honest gaps (some AUD/USD history starts only in 2023). It prepares
Phase 4 by stocking the pantry: Phase 4 will convert each of these
series into per-stock numeric features, and the publication-date
discipline here is what makes Phase 5's no-time-travel guarantee
possible.

---

# Phase 4 — Feature Engineering & Target Labels

**Commit:** `350cf90` · **Modules:** `features/` (price, market, macro, short features, targets)

### Objective
Convert the raw silver tables into **one big analysis-ready table** where
each row answers: *for one stock on one day, everything knowable as of
that day's close* — and attach the **question we want the machine to
learn**: what did that stock actually do over the next few sessions?

### Concepts introduced
* **Feature** — a computed number summarising some aspect of a stock on
  a day. Features are the "hints" the model learns from. This platform
  computes **69 of them**, grouped by research idea (hypotheses):
  * **Momentum** — "has it been rising?" e.g. return over the last 20
    or 63 days, or relative to the whole market.
  * **Mean reversion** — "has it overshot and may bounce back?" e.g.
    distance from its 63-day high/low, position inside its **Bollinger
    bands** (a band two standard deviations — a statistical measure of
    typical wiggle — around the average price).
  * **Volatility** — how much it wobbles (`volatility_20d`: the standard
    deviation of the last 20 daily returns).
  * **Trend indicators** — classic trading-chart quantities: **MACD**
    (a fast average minus a slow average — rising means recent prices
    beat older ones), **ADX** (trend strength), **ATR** (average daily
    range), price relative to its **SMA** (simple moving average =
    average of the last N closes).
  * **Market regime** — what the *whole market* is doing: benchmark
    returns, how far the benchmark is from its own average, **breadth**
    (what fraction of all 200 stocks sit above their own average — a
    crowd mood gauge), the stock's **beta** (how much it moves when the
    market moves).
  * **Macro features** — the RBA/ABS/CPI/currency values *as of their
    publication dates* (Phase 3's discipline pays off here).
  * **Short-interest features** — ASIC short positions relative to the
    stock.
* **Target label** — the answer the model must learn to predict. Here:
  `future_return_3d` (the stock's return over the **next 3 trading
  sessions**) and `direction_3d` (did it rise? yes/no). A 10-session
  variant exists too.
* **Embargo** — a deliberate gap inserted between "information used" and
  "outcome measured" so the two can never overlap. The label for Monday
  uses Tuesday–Thursday prices, so any feature computed with Friday's
  data would leak the answer; the embargo rule makes overlaps
  impossible by construction.
* **Liquidity flag** — a per-day yes/no: is this stock realistically
  tradable today (enough shares changing hands)?

### How it achieves it
For every (stock, session) pair since 2016, compute the 69 features
using **only data up to that session's close** (including honoring
publication dates for macro/ASIC data), attach the liquid flag, then
compute the forward-looking labels in *separate columns* clearly
marked as the future. Stamp every row with the **`feature_set_version`**
(a version label, `v0.1`) so a model can always prove which recipe made
its training data. Quality rules re-run here too (no NaN spikes where
they're forbidden, labels align to the calendar, etc.).

### How it builds on Phases 2–3 — and prepares Phase 5
It consumes exactly the Phase 2/3 silver tables (prices + macro + short
positions) and the Phase 1 quality framework. Its output — the gold
`features.parquet` — **is** Phase 5's input. It also fixes the ML
question in stone: "3-session horizon". Crucially, it embeds the
anti-cheating structure (embargo, publication dates, version stamps)
that Phase 5's training will lean on.

---

# Phase 5 — Model Training (Walk-Forward, No Peek)

**Commit:** `8af4517` · **Modules:** `models/` (dataset, splits, cv, fit, checks, build)

### Objective
Train four candidate machine-learning models to predict the 3-session
question, and produce **honest, never-peeking predictions** for every
stock-day — plus a defensible choice of the single best model.

### Concepts introduced
* **Model** — a program that learns a mapping from features (69 hints)
  to a label (the answer). "Training" = showing it historical examples
  until its internal knobs fit the patterns.
* **Classifier vs regressor** — a **classifier** answers yes/no ("will
  it rise?"); a **regressor** outputs a number ("it will return about
  +0.4%"). This project trains two of each.
* **Tree ensembles** — the strongest models here (`hist_gbt`,
  `hist_gbt_reg`) are **gradient-boosted decision trees**: think
  thousands of simple yes/no flowcharts voting together. **Logistic
  regression** and **ridge regression** are older, simpler linear
  models — useful benchmarks: if fancy trees can't beat them, the hints
  were weak.
* **Cross-validation (CV)** — testing a model fairly. You can't test on
  data the model saw during training (that's memorisation, not
  learning). CV = split history into chunks, train on some, test on the
  rest, rotate.
* **Walk-forward CV** — the time-series-correct CV. Ordinary CV shuffles
  chunks randomly, which would let a model trained on 2024 predict 2021
  — backwards time travel. Walk-forward instead always trains on the
  **past** and tests on the **next** chunk, then slides forward: exactly
  how the model would have been used in real life. This platform runs
  **25 such folds** (chunks).
* **Purging and embargoing** — removing training rows whose label
  windows overlap the test chunk, plus a buffer, so no answer leaks
  across the boundary.
* **Evaluation window vs held-out window** — the first ~11 folds are the
  **evaluation** set: the only place where comparing models is allowed.
  The final 14 folds (everything from **2024-07-01**) are **held out and
  sealed**: reported once, never used to choose anything.
* **IC (information coefficient)** — for stock-picking, we care less
  about exact predicted returns and more about **ranking**: did the
  model's top-ranked stocks actually do better than its bottom-ranked?
  The **Spearman rank correlation** measures exactly that: 1.0 = perfect
  ordering, 0 = no better than random. **0.03 is small but positive.**
* **OOS (out-of-sample) predictions** — predictions made by a model on
  data it never trained on; the only kind worth recording.

### How it achieves it
For each of 25 folds: slice the feature table at the fold boundary,
**purge/embargo** the danger zone, train all four models on the past,
predict the next chunk, record predictions stamped with `fold_id` and
`trained_through` (the date up to which the model knew the world).
Aggregate evaluation metrics across the evaluation folds → pick the
winner. Then run the same protocol across the held-out folds and report
those metrics once, separately, in a file marked "diagnostics only".

Results: 283,957 OOS predictions per model (2020-11-06 → 2026-11-09).
On evaluation folds, `hist_gbt_reg` ranked stocks best (Spearman
**0.030**); ridge 0.007; the classifiers' ROC-AUC ~0.52 (barely better
than coin-flip — typical for short-horizon direction). Held-out
re-check: same small-positive story. **Model selected: `hist_gbt_reg`.**

### How it builds on Phase 4 — and prepares Phase 6
It reads Phase 4's gold features/labels, obeys the version stamps, and
produces `gold/predictions.parquet` — which **is** Phase 6's input. Two
hand-offs are locked in: every prediction carries `trained_through`
(so Phase 6 can refuse stale ones), and the selection decision (which
model, which signal) is **finished here** — Phase 6 will not re-choose
based on later results, keeping the sealed window meaningful.

---

# Phase 6 — Portfolio Construction

**Commit:** `2a66582` · **Modules:** `portfolio/` (schedule, selection, weights, build, checks)

### Objective
Turn the model's daily predictions into a concrete, rule-following
**shopping list**: every Monday, exactly which 20 stocks to own, with
exactly what fraction of money in each — decided using only information
available at that Monday's close.

### Concepts introduced
* **Rebalance** — the periodic act of re-aligning the portfolio to the
  latest target list (buy the new entrants, sell the dropouts). This
  project: **weekly**, on Mondays; when the ASX is closed on a Monday,
  the decision shifts to the next actual trading session.
* **Eligibility screens** — safety filters before ranking: the
  prediction must be **stale-free** (`trained_through` strictly before
  the decision date — the look-ahead guard in action), the stock must
  be **liquid** (Phase 4's flag), and must have **≥250 sessions of
  history** (~1 year) so the features behind the prediction are real.
* **Rank and select** — order eligible stocks by predicted return, keep
  the **top 20**.
* **Weighting** — how much money per stock. Here: **equal weight**
  (1/20 = 5% each), with a hard cap of **10% per name** (defensive
  evenness), and — the platform's signature rule — **prefer cash over
  violation**: if fewer than 20 names pass the screens, hold the
  leftover as cash rather than bend any rule.
* **Turnover** — how much of the portfolio changed at a rebalance
  (0 = identical list, 1 = everything swapped). Matters because every
  change costs money (Phase 7's theme).

### How it achieves it
A schedule generator produces the 314 weekly decision dates
(2020-11-09 → 2026-11-09, Mondays shifted to real sessions). For each
date: apply screens, rank, select 20, weight them, compute cash
residual and turnover vs the previous list. Output:
`gold/target_portfolios.parquet` — 6,280 rows of
(date, ticker, weight, rank, signal, `trained_through`), plus a
per-rebalance summary (avg holdings exactly 20.0, avg turnover 56.6%).
Quality rules verify the constraints (no stock over 10%, never more
than 20 holdings, every listed name was eligible that day).

### How it builds on Phase 5 — and prepares Phase 7
It consumes Phase 5's predictions with their `trained_through` stamps
and the Phase 4 liquidity flag; the Phase 1 quality framework now
guards *portfolio* rules. It deliberately **does not** simulate money —
no prices, no fees, no trading — that separation is the design: Phase 7
can then be honest about execution without construction logic and
simulation logic entangled. It prepares Phase 7 by defining precisely
*what should be held* on every effective date, so the engine's only job
is *how reality would have treated those orders*.

---

# Phase 7 — Backtest Engine

**Commit:** `8ef37f5` · **Modules:** `backtest/` (engine, checks, metrics, build)

### Objective
Replay history **as if** the Phase 6 shopping lists had really been
traded — with every real-world friction included — and produce the
money answer: equity curve, trades, returns, risk, costs, plus a
benchmark comparison.

### Concepts introduced
* **Backtest** — historical simulation of a strategy. The honesty bar
  is high: a bad backtest quietly assumes perfect conditions and
  produces fantasy profits.
* **Execution lag** — the decision is final at Monday's close; the trade
  happens at **Tuesday's open** (next-session). You cannot both know
  Monday's close and trade at it. Enforced by a machine rule, not
  goodwill.
* **Slippage** — the price you actually get is slightly worse than the
  quoted price (buy a touch higher, sell a touch lower), because
  demanding instant execution has a price. Modelled here as 0.10%.
* **Brokerage** — the broker's fee: AUD 9.50 fixed **per trade** plus a
  small percentage. Fixed fees punish small trades — this detail will
  turn out to decide the whole story.
* **Whole shares** — you buy 493 shares, never 493.6.
* **No leverage** — never borrow to buy. If cash runs out mid-list,
  skip the buy and count it. Cash can never go negative (enforced).
* **NAV (net asset value)** — the portfolio's total value: cash + value
  of every share at its last known price. Tracked daily; the curve of
  NAV over time is the strategy's report card.
* **Drawdown** — how far NAV has fallen from its highest point so far.
  A −41% drawdown means an investor once watched their wealth halve-ish.
* **Benchmark comparison** — the same-period result of just holding
  `VAS.AX` (the whole-market ETF), so "did the clever strategy win?" has
  a fair baseline.
* **Sharpe ratio** — average return per unit of wobble. ~1.0 is decent;
  0.29 is weak.
* **Corporate-action handling in simulation** — dividends credited to
  whoever held at the ex-date; splits multiply share counts (fractional
  remainders paid as cash-in-lieu); suspended stocks valued at last
  known close, never invented prices.

### How it achieves it
A session-by-session loop over 1,521 trading days (2020-11-10 →
2026-11-12): on each effective date, execute the Phase 6 targets at the
open with slippage and fees (sells first to free cash, then buys,
skipping any buy cash can't cover), then update closes, credit
dividends, apply splits, mark NAV. Every skip (2,051 dust-sized trades,
108 unaffordable buys, 1 suspended valuation) is a counted diagnostic
surfaced in quality rules — "counted, never silent". Metrics module
computes returns, CAGR, volatility, Sharpe, Sortino, max drawdown with
its peak/trough/recovery dates, tracking error, information ratio,
beta, alpha, hit rate — plus a **held-out-window** block (from
2024-07-01) reported separately.

The verdict (full period): NAV AUD 96,447 → **159,395 (+65.27%)**;
benchmark **+79.02%**; Sharpe **0.289**; max drawdown **−41.18%**
(Nov 2021 → Sep 2022, recovered Jun 2025); **7,571 trades costing
AUD 94,474** — 94% of starting capital in fees. The strategy made
money; it lost to doing nothing clever. Held-out window: strategy
+39.86% vs benchmark +27.49% (diagnostics only).

### How it builds on Phase 6 — and prepares Phase 8
It consumes exactly Phase 6's target lists plus Phase 2's silver prices
— construction and simulation stay cleanly separated (Phase 6 says
*what to hold*; Phase 7 says *what reality would have done to it*).
The Phase 1 quality framework now guards simulation integrity (cash
never negative, execution strictly after signal, holdings cap
respected). It prepares Phase 8 by leaving behind **one reproducible,
in-memory simulation function** — the thing the sensitivity grid will
call 81 times.

---

# Phase 8 — Sensitivity & Robustness

**Commit:** `348acff` · **Modules:** `sensitivity/` (grid, runner)

### Objective
Ask the sobering question: *was the Phase 7 result a fragile accident of
our arbitrary choices?* Re-run everything under 81 combinations of the
choices we could have made differently, and report **all** of them.

### Concepts introduced
* **Sensitivity analysis** — systematically varying assumptions (number
  of stocks, rebalance frequency, weight cap, cost scenario) and
  observing whether conclusions survive. A conclusion that only holds
  at one setting is an accident; one that holds across the grid is a
  finding.
* **Grid** — the full set of combinations (a cartesian product): 3
  sizes {10, 20, 30} × 3 frequencies {weekly, fortnightly, monthly} ×
  3 caps {5%, 10%, 15%} × 3 cost scenarios {optimistic, base,
  conservative} = **81 points**.
* **Cost scenarios** — three graded guesses about trading friction, to
  see how much conclusions depend on cost optimism.
* **"Not a leaderboard"** — the critical statistical discipline. With
  81 results, one will look best by luck. Choosing *after* seeing
  results is selection on noise (and quietly re-uses the same history
  twice). So the output is labelled — in the file itself and in the
  dashboard — as diagnostics, never a menu to pick from.
* **Skipped axes with reasons** — two planned axes could not run:
  the **10-session horizon** (only the 3-session model was ever
  trained — one config line would train it, but no fabricated
  substitute is allowed) and **sector weights** (no free point-in-time
  sector map exists). Both are recorded as explicit skips with
  explanations, not silently omitted.
* **Reference check** — the base configuration inside the grid must
  reproduce the Phase 7 primary run **to the cent**; the runner
  compares NAVs and refuses to continue on any mismatch. This proves
  the grid reuses the same engine rather than a lookalike.

### How it achieves it
Inputs are loaded once into memory; the grid is expanded; each point
re-runs portfolio construction (via the exact same pure function Phase 6
uses) and the Phase 7 engine, runs all quality checks per point, and
appends one result row. Outputs: a results table
(`sensitivity_results.parquet`, 81 rows) and a JSON report with the
skip list, the reference check (`matches: true`) and the
not-a-leaderboard note.

Findings worth trusting: **monthly rebalancing averaged Sharpe 0.90 vs
weekly −0.10** — trading less often doesn't add skill, it just wastes
less on fixed fees; **10 names beat 20 beat 30** on this single history
(0.61 / 0.40 / 0.30 — one sample, so a hint not a law); cost-scenario
ranking is sane (cheaper costs → better results). Full spread: returns
−90%…+477%.

### How it builds on Phase 7 — and prepares Phase 9
It is literally Phase 7's engine called 81 times, with Phase 6's pure
construction function reused — no new simulation logic. (It even found
and fixed a real engine bug: dust-sized *exit* trades were being
skipped, stranding positions and breaking the 30-name holdings cap on
monthly scenarios; full exits now always execute, counted as
`forced_dust_exits`.) It prepares Phase 9 by being exactly the kind of
table a dashboard should display — and by being the last pipeline
stage, it defines what "everything finished" means for automation.

---

# Phase 9 — Dashboard & Automation

**Commit:** `0ffbe4b` · **Modules:** `reporting/artifacts.py`, `pipeline/stages.py` (report, update), `dashboard/`

### Objective
Make the results **watchable** and the platform **refreshable** — a
local web dashboard a human can browse, plus one command that re-runs
the entire pipeline end-to-end (the "automation").

### Concepts introduced
* **Dashboard** — a local web page rendering the stored results
  (Streamlit, a Python framework for quick data apps). Read-only: it
  can show you anything, change nothing, trade nothing.
* **Artifact** — any file a stage leaves behind (a JSON quality
  report, a Parquet equity curve). The dashboard only ever displays
  artifacts; a missing artifact shows "run stage X first", never a
  fabricated number.
* **`report` command** — consolidates the platform's state (which
  stages ran, their quality statuses, artifact freshness, headline
  metrics) into one snapshot file, `platform_summary.json`.
* **`update` command** — the automation: run *ingest → validate →
  transform → features → train → portfolio → backtest* (optionally
  sensitivity) in order, abort at the first failure, and refresh the
  snapshot last. Because every stage is deterministic and logged, a
  failed update is fixed by repairing the cause and rerunning — the
  **restartability** contract.
* **Scheduling** — the OS can run `scripts/update.ps1` weekly (a
  Task Scheduler recipe is in `docs/automation.md`), matching the
  strategy's weekly cadence.

### How it achieves it
A read-only `reporting` module loads every artifact safely (env-var
overridable lake root, `None` for anything absent). Five Streamlit
pages: **Home** (headline NAV/benchmark/quality, stage-status table,
latest run), **Backtest** (equity curve vs benchmark, drawdown chart,
all metrics incl. the held-out block marked diagnostics-only,
diagnostics counters, trades), **Portfolio** (holdings/cash/turnover
per rebalance, latest 20 target weights), **Sensitivity** (heat map by
size × cap, the not-a-leaderboard banner, skipped axes), **Data
Quality** (every stage's rules with detail, the run log). The research-
only disclaimer is on every page.

### How it builds on Phase 8 — and prepares Phase 10
It adds **no new analysis** — every number shown was already produced
by Phases 1–8; the module's discipline is *never fabricate, only
display*. `update` chains the existing stage functions in order. It
prepares Phase 10 by making the platform self-describing: the snapshot,
the run log and the dashboard give the report writer (and any reader)
a complete, current picture of what the platform believes and why.

---

# Phase 10 — Final Report, Presentation & Explainer

**Commit:** `b75a635` · **Docs:** `final_report.md`, `presentation.md`, `explainer.md`, this document

### Objective
Write down the verdict — for every audience: a full technical report, a
slide deck, and plain-language documents (including this phase-by-phase
walkthrough) for readers new to markets and ML.

### Concepts introduced
* **Research write-up discipline** — state the question, the method,
  the numbers, the limitations, and the honest verdict, with every
  figure traceable to an artifact file.
* **Audience-splitting** — the same truth rendered three ways: precise
  (report), scannable (slides), from-zero (explainers).

### How it achieves it
`final_report.md` carries the complete verdict: +65.27% vs benchmark
+79.02%, cost drag (AUD 94,474 of fees on 56.5% weekly turnover)
identified as the mechanism, small-but-real signal (IC 0.030), the
sealed-window check, ten explicit limitations, and five
evidence-directed next steps (first: train the 10-session horizon, so
lower turnover can let the edge keep its fees). `presentation.md`
compresses it to 12 slides. `explainer.md` and this document explain
everything from zero.

### How it builds on Phase 9 — and what it leaves behind
Every number was already computed and snapshotted; Phase 10 only
interprets, never invents. It leaves the project **complete**: a
reproducible pipeline (10 CLI commands), a 210-test safety net, a
dashboard, automation, and an honest written verdict — the durable
product being the *discipline*, and the honest negative being its
first, most valuable finding.

---

## How the phases chain together (the dependency spine)

| Phase | Consumes (from) | Produces (for the next phase) |
|---|---|---|
| 1 foundation | — | storage paths, quality framework, run registry, CLI |
| 2 market data | 1's storage + quality | silver prices + calendar + coverage |
| 3 macro/alt | 2's ingest pattern | macro/short silver with publication dates |
| 4 features/labels | 2 + 3 silver tables | gold features (69) + labels + version stamps |
| 5 training | 4's features/labels | OOS predictions + selected model (`hist_gbt_reg`) |
| 6 portfolio | 5's predictions | 314 weekly target lists (weights, ranks) |
| 7 backtest | 6's lists + 2's prices | equity curve, trades, metrics, costs |
| 8 sensitivity | 6 + 7 as functions | 81-point grid + skips + reference check |
| 9 dashboard/ops | all artifacts | snapshot, update command, dashboard |
| 10 report | everything | the written verdict |

Two rules run through all ten phases and are worth memorising:

1. **Never peek** — every phase that touches time must prove (via tests,
   not intentions) that it used only information the world had at that
   moment.
2. **Never stay silent** — every skipped action, data gap and warning is
   counted in a file a human can read.

---

## Mini-glossary

| Term | Plain meaning |
|---|---|
| ASX | Australia's stock exchange |
| Ticker | A stock's short code (`BHP.AX`) |
| OHLC | A day's Open / High / Low / Close prices |
| Adjusted close | Price rewritten so dividends/splits don't fake a fall |
| Dividend | Cash a company pays its shareholders |
| Split / consolidation | One share becomes N (or N become one); wealth unchanged |
| Feature | A computed hint about a stock-day (69 of them here) |
| Label | The answer to learn: next-3-session return |
| Model | A program that maps features → label after learning from examples |
| Training | The learning process |
| Fold | One train-on-past / test-on-next slice in walk-forward CV |
| Overfitting | Memorising history instead of learning patterns |
| OOS | Out-of-sample: predictions on unseen data |
| IC / Spearman | Does the predicted ranking match reality? (0 = random) |
| ROC-AUC | Yes/no prediction quality (0.5 = coin flip) |
| Rebalance | The periodic portfolio refresh |
| Weight | Fraction of money in one stock |
| Turnover | Fraction of the portfolio traded per rebalance |
| NAV | Total portfolio value (cash + shares) |
| Drawdown | Fall from the highest NAV so far |
| Sharpe | Return per unit of wobble |
| Backtest | Honest historical simulation |
| Slippage | Executing slightly worse than the quoted price |
| Brokerage | The broker's fee (fixed AUD + %) |
| Leverage | Borrowing to invest (never allowed here) |
| Survivorship bias | Studying only companies that still exist today |
| Point-in-time | Only what was knowable *then* |
| Bronze/Silver/Gold | Raw / cleaned / analysis-ready data layers |
| Parquet | Fast columnar table file format |
| DuckDB | Local SQL engine over Parquet |
| Streamlit | Python framework for quick local data web-apps |
