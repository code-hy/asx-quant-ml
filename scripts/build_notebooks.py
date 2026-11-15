"""Build and execute the project notebooks (EDA + end-to-end workflow).

The committed ``.ipynb`` files are the deliverable; this script is how they
are (re)generated so the source-of-truth stays reviewable in plain Python:

    uv run --extra dev python scripts/build_notebooks.py

Each notebook is assembled with ``nbformat`` and executed in place with
``nbclient`` (an ipykernel Python session), so the committed files contain
real outputs produced against the current data lake. The EDA notebook is
read-only; the workflow notebook actually re-runs the cheap pipeline stages
(ingest-incremental -> validate -> transform -> features -> portfolio ->
backtest -> report), which are deterministic and reproduce the lake exactly.
"""

from __future__ import annotations

import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def md(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(source)


def code(source: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(source)


def execute_and_write(cells: list[nbformat.NotebookNode], path: Path, title: str) -> None:
    """Assemble cells into a notebook, execute it, and write it in place."""
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.13"},
        },
    )
    started = time.perf_counter()
    NotebookClient(notebook, timeout=3600, kernel_name="python3").execute()
    nbformat.write(notebook, path)
    print(f"{title}: executed and written to {path} in {time.perf_counter() - started:.1f}s")


# --------------------------------------------------------------------------
# Notebook 1 - Exploratory Data Analysis
# --------------------------------------------------------------------------
EDA_CELLS = [
    md("""\
# Exploratory Data Analysis — features, targets and predictions

This notebook is the project's **EDA deliverable**. It inspects the
analysis-ready dataset (`data/gold/features.parquet` joined with
`target_labels.parquet`) and the model's out-of-sample predictions, and
answers four questions:

1. **What are we predicting?** — the distribution of the prediction target
   (`TO_PREDICT = future_return_3d`, the return over the *next 3 trading
   sessions*) and its up/down balance.
2. **Which features relate to it?** — Spearman and Pearson correlation of
   all engineered features against the target, plus a year-by-year
   stability check.
3. **How do predictions and precision differ by ticker?** — per-ticker
   information coefficients and the precision of the names the strategy
   actually selects each week.
4. **Where does it go wrong?** — an error analysis of the worst misses,
   with improvement ideas.

Data provenance: free public sources only (Yahoo Finance, RBA, ABS, ASIC,
Wikipedia), assembled by the pipeline demonstrated in
[`02_end_to_end_workflow.ipynb`](02_end_to_end_workflow.ipynb).

> **Research only — not financial advice.**"""),
    code("""\
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 160)
plt.rcParams["figure.dpi"] = 110

GOLD = Path("data/gold")
TO_PREDICT = "future_return_3d"  # the label every model is trained to predict
META_COLS = ["ticker", "date", "liquid_flag", "feature_set_version", "ingestion_run_id",
             "retrieved_at"]
LABEL_COLS = [
    "future_return_3d", "direction_3d", "label_available_date_3d",
    "future_return_10d", "direction_10d", "label_available_date_10d",
]

features = pd.read_parquet(GOLD / "features.parquet")
labels = pd.read_parquet(GOLD / "target_labels.parquet")
# features = every numeric column that is neither metadata nor a label
FEATURE_COLS = [
    c for c in features.columns
    if c not in META_COLS and c not in LABEL_COLS
    and pd.api.types.is_numeric_dtype(features[c])
]
print(f"features: {len(features):,} rows x {len(FEATURE_COLS)} feature columns "
      f"({features['date'].min()} -> {features['date'].max()})")

# One combined data frame: features (known at each close) + labels (the future).
df = features.merge(
    labels[["ticker", "date", "future_return_3d", "direction_3d", "future_return_10d"]],
    on=["ticker", "date"],
    how="inner",
)
print(f"joined features + labels: {len(df):,} rows")"""),
    md("""\
## 1. The prediction target

`TO_PREDICT = future_return_3d` is the stock's return from its close on day
*t* to its close three trading sessions later (labels are computed on
adjusted closes and live in a separate gold table, `target_labels.parquet`,
so the feature columns can never accidentally contain them)."""),
    code("""\
target = df[TO_PREDICT].dropna()
print(target.describe().to_string())
print(f"\\nshare of stock-days with positive next-3d return "
      f"(direction_3d == 1): {(df['direction_3d'] == 1).mean():.3f}")

fig, ax = plt.subplots(figsize=(8, 3.4))
ax.hist(target, bins=80, range=(-0.25, 0.25), color="#3572b0")
ax.set_title(f"Distribution of {TO_PREDICT}  (n={len(target):,})")
ax.set_xlabel("next-3-session return")
ax.set_ylabel("stock-days")
fig.tight_layout()
plt.show()"""),
    md("""\
Typical of short-horizon equity returns: a near-zero mean, fat tails, and a
roughly balanced up/down split. Any real, learnable edge is small relative
to this noise — which is exactly why the models are evaluated on *ranking*
quality (Spearman IC) rather than raw return prediction error."""),
    md("""\
## 2. Correlation of every feature with `TO_PREDICT`

Spearman (rank) correlation is the more robust lens for skewed returns;
Pearson is shown for reference. Both are pooled over the full 2016–2026
sample."""),
    code("""\
spearman = df[FEATURE_COLS].corrwith(df[TO_PREDICT], method="spearman")
pearson = df[FEATURE_COLS].corrwith(df[TO_PREDICT], method="pearson")
corr = pd.DataFrame({"spearman": spearman, "pearson": pearson}).sort_values("spearman")

print("weakest 8 (negative side):")
print(corr.head(8).round(4).to_string())
print("\\nstrongest 8 (positive side):")
print(corr.tail(8).round(4).to_string())"""),
    code("""\
top = pd.concat([corr.head(10), corr.tail(10)]).sort_values("spearman")
colors = np.where(top["spearman"] > 0, "#2e8b57", "#b0413e")
fig, ax = plt.subplots(figsize=(7, 6.5))
ax.barh(top.index, top["spearman"], color=colors)
ax.axvline(0, color="black", lw=0.8)
ax.set_title(f"Spearman correlation with {TO_PREDICT}\\n"
             f"(extreme 10 on each side of {len(corr)} features)")
ax.set_xlabel("Spearman correlation (pooled 2016-2026)")
fig.tight_layout()
plt.show()"""),
    md("""\
Every single-feature correlation is small — of the order of a few percent
at best. That is the honest, expected picture for 3-day stock returns: no
single indicator carries the signal; the value (if any) comes from combining
many weak hints, which is what the gradient-boosting model in Phase 5 does."""),
    md("""\
### Are the relationships stable over time?

A correlation that flips sign by year is regime-dependent, not a law. The
plot below re-computes the Spearman IC of the six strongest features inside
each calendar year."""),
    code("""\
focus = list(pd.concat([corr.head(3), corr.tail(3)]).index)
df["year"] = pd.to_datetime(df["date"]).dt.year
stability = pd.DataFrame({
    year: group[focus].corrwith(group[TO_PREDICT], method="spearman")
    for year, group in df.groupby("year")
}).T.sort_index()

fig, ax = plt.subplots(figsize=(8, 3.6))
for name in focus:
    ax.plot(stability.index, stability[name], marker="o", ms=3, label=name)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Year-by-year Spearman IC of the six strongest features")
ax.set_xlabel("calendar year")
ax.legend(fontsize=7)
fig.tight_layout()
plt.show()"""),
    md("""\
Read this before trusting any backtest: the per-year lines wobble around
zero and some change sign. Relationships on a 3-day horizon are weak and
regime-dependent — the reason the project trains with walk-forward
cross-validation (always train on the past, test on the next chunk) instead
of one global fit, and why a small pooled IC of ~0.03 is treated as a
*whisper*, not a law."""),
    md("""\
## 3. Per-ticker prediction quality

Out-of-sample predictions of the **selected model** (`hist_gbt_reg`) are
joined with realised labels, and one Spearman IC is computed per ticker
(tickers with fewer than 250 predictions are skipped). The spread across
tickers matters for the portfolio: averaging a mix of good and
anti-predictable names dilutes the edge."""),
    code("""\
preds = pd.read_parquet(GOLD / "predictions.parquet")
preds = preds[preds["model_name"] == "hist_gbt_reg"]  # the selected model
pp = preds.merge(
    labels[["ticker", "date", "future_return_3d"]],
    on=["ticker", "date"],
    how="inner",
)
print(f"selected-model predictions with a realised label: {len(pp):,}")


def ticker_ic(group: pd.DataFrame) -> float:
    if len(group) < 250:
        return np.nan
    return group[["prediction", "future_return_3d"]].corr(method="spearman").iloc[0, 1]


tic = pp.groupby("ticker").apply(ticker_ic, include_groups=False).dropna().sort_values()
print(f"tickers with >= 250 predictions: {len(tic)}; mean per-ticker IC: {tic.mean():+.4f}")
print("\\nmost predictable 8:")
print(tic.tail(8).round(3).to_string())
print("\\nleast predictable 8:")
print(tic.head(8).round(3).to_string())"""),
    code("""\
fig, ax = plt.subplots(figsize=(7, 3.2))
ax.hist(tic, bins=40, color="#3572b0")
ax.axvline(0, color="black", lw=0.8)
ax.set_title("Per-ticker Spearman IC, prediction vs realised 3-session return")
ax.set_xlabel("per-ticker IC")
fig.tight_layout()
plt.show()"""),
    md("""\
The dispersion around zero is wide: a handful of tickers are reliably
predictable and a similar handful are *anti*-predictable (high predicted
return followed by a fall). Ideas this suggests: down-weight or exclude
persistent negative-IC names, or size positions by per-ticker confidence.
Any such rule must itself be validated walk-forward before it earns a place
in the pipeline."""),
    md("""\
## 4. Precision of the names actually selected

The strategy holds the top-20 ranked stocks each Monday. "Precision" here:
the share of selected names whose realised next-3-session return was
positive — compared against the same-day universe baseline (the share of
*all* stocks that rose over the next 3 sessions)."""),
    code("""\
port = pd.read_parquet(GOLD / "target_portfolios.parquet")
sel = port.merge(
    labels[["ticker", "date", "future_return_3d", "direction_3d"]],
    left_on=["ticker", "rebalance_date"],
    right_on=["ticker", "date"],
    how="left",
)
precision = (sel["direction_3d"] == 1).mean()
baseline = (df[df["date"].isin(port["rebalance_date"])]["direction_3d"] == 1).mean()
print(f"selected names positive over next 3 sessions: {precision:.3f}")
print(f"universe baseline on the same dates:          {baseline:.3f}")

sel["year"] = pd.to_datetime(sel["rebalance_date"]).dt.year
by_year = sel.groupby("year").agg(
    precision=("direction_3d", lambda s: (s == 1).mean()),
    n=("direction_3d", "size"),
)
print(by_year.round(3).to_string())"""),
    code("""\
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.bar(by_year.index.astype(str), by_year["precision"], color="#2e8b57", label="selected names")
ax.axhline(baseline, color="black", ls="--", label=f"universe baseline ({baseline:.2f})")
ax.set_title("Share of picks with a positive next-3-session return, by year")
ax.set_ylabel("precision")
ax.legend()
fig.tight_layout()
plt.show()"""),
    md("""\
Selection adds only a slim hit-rate edge over the same-day universe
baseline (see the output: pooled and per-year figures). Individual years
sit on both sides of the baseline — the edge is a whisper, not a law. The
backtest (Phase 7) shows why even this thin edge did not survive trading
costs: weekly turnover of ~56% spent AUD ~94k in fees, more than the edge
was worth."""),
    md("""\
## 5. Error analysis — the worst misses

Among all selected (ticker, date) picks, which were the worst realised
losses, and did they share a feature profile?"""),
    code("""\
sel_named = sel[sel["future_return_3d"].notna()].copy()
worst = sel_named.nsmallest(10, "future_return_3d")
print("ten worst picks:")
print(worst[["rebalance_date", "ticker", "rank", "signal", "future_return_3d"]]
      .to_string(index=False))"""),
    code("""\
PROF_COLS = [
    "volatility_20d", "beta_252d", "atr_14", "pct_from_63d_high",
    "momentum_ts_pctile_63d", "dollar_turnover_20d",
]
# bring the feature values of each pick alongside its realised outcome
prof_df = sel_named.merge(
    features[["ticker", "date"] + PROF_COLS],
    left_on=["ticker", "rebalance_date"],
    right_on=["ticker", "date"],
    how="left",
)
misses = prof_df[prof_df["future_return_3d"] <= prof_df["future_return_3d"].quantile(0.10)]
profile = prof_df[PROF_COLS].mean().rename("all picks").to_frame()
profile["worst 10% of picks"] = misses[PROF_COLS].mean()
print(profile.round(4).to_string())"""),
    md("""\
## Conclusions and improvement ideas

* **The target is close to noise**: near-zero mean, fat tails, balanced
  up/down; every single-feature correlation is small, and per-year ICs
  wobble and sometimes flip sign. Walk-forward validation was the right
  call; a pooled IC of ~0.03 is a whisper, not an edge.
* **Selection does rank better than chance** — selected names beat the
  same-day universe hit rate, consistently but narrowly.
* **Per-ticker dispersion is real**: both strongly-predictable and
  anti-predictable tickers exist. Candidate improvements: exclude or
  down-weight persistent negative-IC names; confidence-weighted sizing.
* **Costs, not signal, decide the outcome**: the thin edge paid for a very
  expensive trading pattern. Candidate improvements, in likely order of
  impact: a lower-turnover rebalance (the sensitivity grid already shows
  monthly rebalancing lifts Sharpe from ~-0.10 to ~0.90), a longer
  prediction horizon (the 10-session label exists but no model is trained
  on it yet), and only trading when the expected edge after costs is
  positive (a probability/return-threshold rule).
* Nothing here has been *proved* tradable: these are research insights
  intended to seed the next, equally honest, round of walk-forward
  experiments."""),
]

# --------------------------------------------------------------------------
# Notebook 2 - End-to-End Workflow
# --------------------------------------------------------------------------
WORKFLOW_CELLS = [
    md("""\
# End-to-End Workflow — calling the platform's own functions

This notebook demonstrates how to execute the **complete data workflow** by
calling the functions in `src/asx_quant` directly (the same functions the
`asx_quant` CLI drives). The full chain:

| step | CLI command | function | what it does |
|---|---|---|---|
| 1. download | `asx_quant ingest` | `stages.run_ingest` | Yahoo Finance + Wikipedia -> Bronze |
| 2. validate | `asx_quant validate` | `stages.run_validate` | data-quality rules on Bronze |
| 3. transform | `asx_quant transform` | `stages.run_transform` | Bronze -> Silver (clean + calendar) |
| 4. features | `asx_quant features` | `stages.run_features` | Silver -> Gold features + labels |
| 5. train | `asx_quant train` | `stages.run_train` | walk-forward training -> OOS predictions |
| 6. portfolio | `asx_quant portfolio` | `stages.run_portfolio` | predictions -> weekly top-20 lists |
| 7. simulate | `asx_quant backtest` | `stages.run_backtest` | exact cost-inclusive simulation |
| 8. report | `asx_quant report` | `stages.run_report_stage` | platform snapshot |

**Two regimes:**

* **Run from the local lake** (this notebook): the steps below execute
  in-process. `ingest --mode incremental` appends only new sessions, and
  every later stage rebuilds its dataset deterministically — a full refresh
  is never needed to stay current.
* **Full refresh from sources**: run
  `uv run asx_quant update --with-sensitivity`, or flip the guarded cell
  below to `RUN_FULL_REFRESH = True`. The only slow stage is *train*; it is
  guarded the same way (set `RUN_SLOW_TRAIN = True` to run it here).

> **Research only — not financial advice.**"""),
    code("""\
import time

import pandas as pd

from asx_quant.config.loader import load_settings
from asx_quant.pipeline import stages
from asx_quant.pipeline.health import run_health_check
from asx_quant.reporting import artifacts

pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 160)

settings = load_settings()
print("settings loaded (config/*.yaml) — lake root:", artifacts.lake_root())"""),
    md("""\
### Health check first

The foundation self-probe (storage, config, registry) uses a temporary
sandbox, so it never mutates real data."""),
    code("""\
health = run_health_check(settings)
print(health.summary())"""),
    md("""\
### Regime 2 — full refresh from sources (guarded)

One call chains every stage and aborts at the first failure. Off by
default: it re-downloads the full history of all 200 tickers and re-trains
every model."""),
    code("""\
RUN_FULL_REFRESH = False  # set True to re-download everything and retrain
if RUN_FULL_REFRESH:
    stages.run_update_stage(settings, mode="historical", with_sensitivity=True)
else:
    print("skipped — this notebook runs regime 1 (incremental) below instead")"""),
    md("""\
### Regime 1 — the live incremental run

Each stage is called with its real function and timed. `ingest` in
incremental mode appends only sessions newer than the lake (2s when
up to date); `validate` enforces the data-quality rules; `transform` and
`features` rebuild Silver and Gold deterministically."""),
    code("""\
def timed(name: str, fn, *args, **kwargs):
    started = time.perf_counter()
    summary = fn(*args, **kwargs)
    print(f"[{time.perf_counter() - started:6.1f}s] {name}")
    return summary


ing = timed("ingest (incremental)", stages.run_ingest, settings, "incremental")
val = timed("validate", stages.run_validate, settings)
print(f"   validate: {val.dataset} — status {'PASS' if val.passed else 'FAIL'}, "
      f"{len(val.results)} rules, {val.warn_count()} warnings")
tr = timed("transform", stages.run_transform, settings)
fe = timed("features", stages.run_features, settings)"""),
    md("""\
### Train (guarded — the only slow stage)

25 walk-forward folds x 4 models over ~455k feature rows. Left off in this
demonstration; the existing out-of-sample predictions are used below."""),
    code("""\
RUN_SLOW_TRAIN = False  # set True to retrain all models in this session
if RUN_SLOW_TRAIN:
    timed("train", stages.run_train, settings)
else:
    preds = pd.read_parquet("data/gold/predictions.parquet")
    print(f"skipped (slow) — using existing OOS predictions: {len(preds):,} rows, "
          f"{preds['model_name'].nunique()} models, "
          f"{preds['date'].min()} -> {preds['date'].max()}")"""),
    md("""\
### Portfolio and backtest

With fresh features and predictions in place, the portfolio stage turns
them into the weekly top-20 target lists and the backtest stage replays
them through history with slippage, brokerage, whole shares and dividends."""),
    code("""\
pf = timed("portfolio", stages.run_portfolio, settings)
bt = timed("backtest", stages.run_backtest, settings)"""),
    md("""\
### The latest trades

What the strategy would hold at the most recent rebalance, and the last
executed trades in the simulation."""),
    code("""\
port = artifacts.load_target_portfolios()
latest_date = port["rebalance_date"].max()
latest = port[port["rebalance_date"] == latest_date]
print(f"target portfolio for {latest_date}: {len(latest)} names")
latest[["ticker", "rank", "signal", "target_weight"]].reset_index(drop=True)"""),
    code("""\
trades = artifacts.load_trades()
print(f"last {len(trades.tail(8))} of {len(trades):,} simulated trades:")
trades.tail(8).reset_index(drop=True)"""),
    md("""\
### Headline results

Every figure below is read straight from the artifacts the stages just
wrote (`data/reports/backtest_metrics.json`).

Note: these figures reflect the **current lake** — the incremental ingest
above refreshed and reconciled the most recent sessions, so the final
sessions (and therefore the closing NAV) can differ slightly from the
build documented in `docs/final_report.md`. That is the reconciliation
window doing its job, not a reproducibility failure: with the same bronze
input, every stage is deterministic."""),
    code("""\
m = artifacts.load_json_report("backtest_metrics")
print(f"NAV            : {m['initial_nav']:,.0f} -> {m['final_nav']:,.0f}  "
      f"({m['total_return']:+.2%})")
print(f"benchmark      : {m['benchmark_total_return']:+.2%}")
print(f"CAGR           : {m['cagr']:.2%} vs benchmark {m['benchmark_cagr']:.2%}")
print(f"Sharpe         : {m['sharpe_ratio']:.3f}")
print(f"max drawdown   : {m['max_drawdown']:.2%} "
      f"({m['max_drawdown_peak_date']} -> {m['max_drawdown_trough_date']})")
print(f"trading costs  : AUD {m['turnover']['total_costs_aud']:,.0f} "
      f"({m['turnover']['n_trades']:,} trades, avg turnover "
      f"{m['turnover']['avg_turnover_per_rebalance']:.1%})")"""),
    md("""\
### Refresh the platform snapshot

The `report` stage consolidates stage statuses, artifact freshness and
headline metrics into `data/reports/platform_summary.json` — the same
snapshot the dashboard reads."""),
    code("""\
rep = timed("report", stages.run_report_stage, settings)
status = artifacts.platform_status()
stages_status = pd.DataFrame(
    [{"stage": k, **v} if isinstance(v, dict) else {"stage": k, "status": v}
     for k, v in status.get("stages", {}).items()]
)
stages_status"""),
    md("""\
### Keeping it current

Schedule the same chain without any notebook: `scripts/update.ps1` wraps
`uv run asx_quant update`, and `docs/automation.md` contains the Windows
Task Scheduler (`schtasks`) recipe for a weekly run matching the Monday
rebalance. For interactive monitoring, `uv run streamlit run
dashboard/Home.py` opens the read-only dashboard. Every stage is
deterministic and logged in `data/run_log.jsonl`, so a failed update is
repaired at the cause and simply re-run."""),
]


def main() -> None:
    NOTEBOOKS.mkdir(exist_ok=True)
    # workflow first: it refreshes the lake incrementally, so the EDA
    # notebook then reads the freshest gold tables
    execute_and_write(
        WORKFLOW_CELLS, NOTEBOOKS / "02_end_to_end_workflow.ipynb", "notebooks/02_end_to_end_workflow.ipynb"
    )
    execute_and_write(EDA_CELLS, NOTEBOOKS / "01_eda.ipynb", "notebooks/01_eda.ipynb")


if __name__ == "__main__":
    main()
