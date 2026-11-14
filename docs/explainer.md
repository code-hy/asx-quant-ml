# A Plain-Language Guide to This Project

*No prior knowledge of stock markets or machine learning is assumed. Every
term is explained the first time it appears. This document describes what
the platform does, why each part exists, and what the results mean — and,
just as importantly, what they do **not** mean.*

> **The one-line summary:** we built a computer program that studies
> history from the Australian share market, tries to learn which stocks
> tend to rise over the next few days, and then plays out — on paper only
> — what would have happened if we had invested AUD 100,000 that way.
> It is a research exercise. It never places real trades and is not
> financial advice.

---

## 1. What is a stock, and what is the ASX?

A **share** (or **stock**) is a tiny slice of ownership in a company. If a
company is worth $1 billion and you own one share worth $10, you own one
billionth of it. Two things can happen to that $10:

* the share **price** can go up or down as the company's prospects and
  the market's mood change, and
* the company may pay a **dividend** — a cash payment to shareholders,
  usually twice a year in Australia, out of its profits.

The **ASX** (Australian Securities Exchange) is the marketplace in
Australia where shares are bought and sold. It is open on business days
(called **trading sessions**). Roughly 2,000 companies are listed; the
**ASX 200** is an index tracking the 200 largest ones. This project uses
**200 large Australian companies** as its study population.

A **ticker** is a short code identifying a company on the exchange — for
example, `BHP` for the mining company BHP. In our data files, ASX tickers
carry a `.AX` suffix (e.g. `VAS.AX`).

**Buy-and-hold benchmark.** To know whether a strategy is any good, you
compare it with the simplest possible alternative: buy the whole market
and hold it. We use **VAS.AX** (an exchange-traded fund holding roughly
every large ASX company) as that yardstick. Over our study window it
returned **+79%**.

---

## 2. What is this project actually doing?

Imagine you had $100,000 in early November 2020 (when the experiment's
prediction data begins) and made this promise to yourself:

> "Every Monday I will look at how each of 200 stocks has behaved over
> the past weeks and months. I will pick the 20 that my prediction model
> likes best, put about 5% of my money into each one, and sell whatever
> fell out of the top 20. I will repeat this every week until today."

The platform's job is to **simulate that promise honestly** — without
cheating by peeking at the future — and report exactly how it would have
turned out, including all the small real-world frictions (trading fees,
price gaps between deciding and buying, whole-share limits).

This "play it out on paper" exercise is called a **backtest**.

---

## 3. Where does the data come from?

All data is **free and public**; nothing was purchased, and every source
is documented in `docs/data_sources.md`:

| Data | Source | What it gives us |
|---|---|---|
| Company list | Wikipedia (ASX 200) | which 200 companies to study |
| Share prices | Yahoo Finance | daily open/high/low/close prices, dividends, stock splits |
| Interest rates & macro | Reserve Bank of Australia (RBA) | cash rate, AUD/USD, yields |
| Short-position reports | ASIC | how much of each stock investors have bet *against* |

The data is stored locally in **Parquet files** (an efficient columnar
format) arranged in three layers: raw (**bronze**), cleaned (**silver**)
and analysis-ready (**gold**). The data-quality rules that check each
layer are enforced by automated tests, so a silent data problem is far
less likely.

One honest gap: point-in-time **sector** labels (e.g. "BHP is a mining
stock") are not freely available in a historical form that respects our
no-peeking rule, so the sector-related ideas in the research plan were
recorded as *explicitly skipped* rather than approximated with today's
sectors (using today's labels for past dates would be a subtle form of
cheating).

---

## 4. What is machine learning, and what is the model doing?

**Machine learning (ML)** is a way of writing software where, instead of
hand-coding rules, we show the computer many historical examples and let
it find patterns itself.

Our ML task is deliberately simple to state:

> "Looking at everything known about a stock **today**, predict its
> return over the **next 3 trading sessions**."

Two kinds of models were trained:

* **Classifiers** predict a yes/no answer: "will the price rise?"
* **Regressors** predict a number: "by how much will it rise?"

The inputs (called **features**) are computed from the price history and
macro data — things like:

* **momentum**: how much has the stock moved in the last week/month?
* **mean reversion**: has it overshot recently and tends to bounce back?
* **volatility**: how much does it usually swing?
* **volume**: how many shares trade hands (a proxy for attention)?
* **market regime**: what is the whole market doing?

## The cardinal rule: never peek at the future

The single biggest way ML-in-finance projects fool themselves is
**look-ahead bias** ("leakage"): accidentally letting the model learn
from information that didn't exist yet at decision time. Example: if you
train on January data and test on December data, you must be certain that
no December information (like that month's average) sneaked into the
January rows.

This project is obsessed with preventing that, in test-enforced ways:

* **Walk-forward validation**: the models are trained the way they would
  have been used in real life — train on the past, predict the next
  stretch, move the window forward, repeat. We do this **25 times**
  (folds), giving an honest "what would the model have predicted at that
  moment, knowing only what was knowable then" record.
* **Purging and embargo**: any example whose label window overlaps the
  test period is removed, plus a small safety buffer after it.
* **A sealed final window**: everything from **1 July 2024 onward** was
  locked away. Model choices were made *without looking at it*, and it is
  reported separately as a one-shot sanity check. (Tempting as it is to
  pick the strategy with the best held-out number, doing so would turn
  the seal into decoration — the report keeps it diagnostic.)
* **Every prediction is stamped** with the date up to which the model had
  been trained (`trained_through`), and the portfolio builder refuses any
  prediction whose stamp is not strictly before the decision date.

## The four models and the one we use

Four models were trained side by side: two classifiers (a gradient-boosted
tree model `hist_gbt` and a classic `logistic` regression) and two
regressors (the gradient-boosted `hist_gbt_reg` and a linear `ridge`).
On the honest walk-forward evaluation, the gradient-boosted regressor
`hist_gbt_reg` ranked the stocks best (a rank-correlation of **0.030** —
small, but positive and stable; see §7 for why "small" is normal). It is
the signal the portfolio uses.

---

## 5. From predictions to a portfolio

Every Monday (shifted to the next trading day when the ASX is closed) the
platform:

1. collects that day's predictions for all 200 stocks,
2. throws out stocks that fail safety screens (not liquid enough, less
   than ~1 year of history, or the prediction is stale),
3. ranks the rest by predicted return and keeps the **top 20**,
4. puts an equal slice of money into each (about 5%), **capped at 10%**
   per stock, and
5. keeps whatever doesn't fit as **cash** rather than bending a rule.

That last bullet is a design philosophy worth naming: **when rules
conflict, hold cash; never invent leverage.** The engine would rather
leave money idle than buy on credit.

---

## 6. The backtest: playing history honestly

The simulator takes each weekly portfolio and replays it session by
session:

* **Execution timing**: the decision is made on Monday's close, but the
  trade happens at the **next session's open** — you cannot trade on
  information and act in the same instant.
* **Costs**: every trade pays a fixed fee (AUD 9.50), a percentage fee,
  and **slippage** (buying slightly above the quoted price, selling
  slightly below — the real-world cost of demanding immediacy).
* **Whole shares only**: you cannot buy 437.6 shares.
* **Cash discipline**: sells happen first (they free up money); a buy
  that would overdraw cash is skipped, not borrowed for.
* **Corporate actions**: dividends are credited to whoever held the
  shares when the ex-dividend date arrived; stock **splits** multiply
  share counts (a 1-for-10 consolidation divides them) with fractional
  remainders paid as cash.
* **Suspensions**: a stock that stops trading that day is valued at its
  last known price, never at a made-up one.

Everything skipped (dust-sized trades, unaffordable buys) is **counted in
a public diagnostics table** — the project's motto is "counted, never
silent."

### What happened? (the headline numbers)

| | Strategy | Market (VAS.AX) |
|---|---|---|
| Period simulated | 10 Nov 2020 → 12 Nov 2026 (1,521 sessions) | same |
| Starting capital | AUD 100,000 (first full NAV 96,447) | AUD 100,000 |
| Final value | **AUD 159,395** | AUD 172,655 on the same NAV base |
| Total return | **+65.3%** | +79.0% |
| Growth per year (CAGR) | **8.7%** | 10.1% |
| Worst peak-to-trough fall | **−41.2%** | similar-period market fall |
| Sharpe ratio (see §7) | **0.29** | — |
| Trades executed | 7,571 | 1 |
| Total trading costs | **AUD 94,474** | negligible |
| Dividend events credited | 149 | included in benchmark |

**Honest reading:** the strategy made money (+65%) but **lost to simply
holding the market** (+79%), and its worst losing stretch (−41%, from
November 2021 to September 2022, taking until June 2025 to recover)
would test most investors' patience. The single biggest reason is **cost
drag**: weekly trading of 20 names at ~$9.50 fixed brokerage per trade
burned AUD 94,474 — nearly the entire starting capital — in fees. The
signal's edge was not strong enough to pay for its own churn. That is a
*finding*, not a failure of the exercise: this is precisely what honest
simulation is for.

The sealed window (July 2024 → November 2026) tells a friendlier
but still cautious story: the strategy did **+39.9%** vs the market's
**+27.5%** there — but that window is small, one market phase, and
reporting it is a sanity check, not a victory lap.

---

## 7. How to read the performance words

* **Return / CAGR**: how much money was made in total / per year
  (compounded).
* **Volatility**: how much the value wobbles; ours was 24% per year
  (wobbly).
* **Max drawdown**: the deepest fall from a peak to a trough. Ours:
  −41%. A number to meditate on before investing real money.
* **Sharpe ratio**: return earned *per unit of wobble*. Above ~1 is
  usually considered good; **0.29 is weak**. For context, 0 means you
  were paid only for taking risk, like a coin-flip with extra steps.
* **Hit rate**: share of days the portfolio's value went up (53%).
* **Beta**: how much the strategy moves when the market moves. Ours 1.27
  — it amplifies market swings rather than hedging them.
* **Spearman rank correlation (IC) of 0.030**: when the model ranked
  stocks, the ordering matched what actually happened 3 days later only
  slightly better than random. In markets, where noise drowns signal,
  a small consistent positive correlation is the honest realistic goal —
  but 0.03 is tiny and fragile, and the backtest shows the cost of
  acting on it weekly can exceed its value.

---

## 8. The sensitivity study: "was that result a fluke of our choices?"

Any backtest depends on dozens of arbitrary choices: 20 stocks or 10?
Weekly or monthly trading? How expensive to assume trading is? The
**sensitivity grid** re-runs everything under 81 combinations and reports
them all — explicitly labelled **"NOT a leaderboard"**, because picking
the best-looking combination after seeing the results is another way of
cheating (it "learns" from the same history twice).

Two robust findings emerged:

1. **Costs dominate frequency.** Mean Sharpe across all configurations:
   monthly 0.90, fortnightly 0.50, weekly −0.10. Trading less often does
   not create skill; it just wastes less money on fixed fees.
2. **Fewer names did better** (mean Sharpe 10 names 0.61, 20 names 0.40,
   30 names 0.30) — with a caveat that this is over one history, not a
   guarantee.

The full table is in `data/reports/sensitivity_results.parquet`, visible
in the dashboard's Sensitivity page.

---

## 9. What is the dashboard?

`dashboard/` is a **Streamlit** app — a local web page you run with:

```
uv run streamlit run dashboard/Home.py
```

It reads the results files and shows them: the equity curve, drawdown,
per-stage health checks, the portfolio's current target holdings, and the
sensitivity table. It can only *display* — it cannot trade, and it never
modifies data.

---

## 10. What this project does NOT claim

This may be the most important section.

* **It is not financial advice.** Nothing here recommends buying or
  selling anything.
* **No real trades happened.** Every number is a simulation on paper.
* **Past patterns may not repeat.** A strategy that worked 2020–2026 can
  fail tomorrow; the next six years are not obligated to resemble the
  last six.
* **The model's edge is small.** A 0.03 rank correlation is a whisper,
  not a shout; transaction costs can easily shout over it.
* **Free data has limits.** Yahoo Finance prices are good but not
  exchange-official; survivorship and point-in-time coverage of the
  universe list are imperfect; sector data was skipped rather than
  faked.
* **The sealed window stays sealed.** Using it to re-tune the system
  would erase its meaning. If the system changes, honest practice is to
  re-declare a *new* untouched window going forward.

### What the project *does* deliver

A complete, reproducible, leakage-controlled research pipeline — data
ingestion with quality gates, feature engineering, walk-forward ML,
rule-constrained portfolio construction, a cost-aware backtest engine
with no-lookahead and no-leverage guarantees enforced by tests, a
robustness grid, and a dashboard — all on free data, all local, all
inspectable. The engineering discipline (point-in-time correctness,
counted-not-silent skips, sealed holdout) is the durable deliverable;
the specific strategy numbers are just what the discipline found.

---

## 11. Try it yourself (30 seconds)

```
uv run asx_quant report                      # snapshot of platform state
uv run streamlit run dashboard/Home.py       # explore results in a browser
uv run pytest                                # run all 209 automated tests
```

Further reading (each builds on the last):

* `docs/architecture.md` — how the software is organised
* `docs/training_methodology.md` — the ML protocol in detail
* `docs/final_report.md` — the full technical report
* `docs/presentation.md` — a slide-by-slide summary deck
