# Automation & operations (Phase 9)

The platform refreshes itself end to end with two commands. Everything is
local-first: no servers, no external services, no credentials.

## The `update` command

```
uv run asx_quant update --mode incremental            # standard refresh
uv run asx_quant update --with-sensitivity            # + slow Phase 8 grid
```

Stage chain, in order (each aborts the run on failure):

1. `ingest` — pull new sessions from the free sources into Bronze
   (incremental: appends since the last batch; universe + RBA macro +
   ASIC short positions re-checked).
2. `validate` — Bronze data-quality rules.
3. `transform` — Bronze → Silver (prices, macro, short positions).
4. `features` — Gold feature set + target labels (leakage-controlled).
5. `train` — purged/embargoed walk-forward training + OOS predictions.
6. `portfolio` — target portfolios from the selected signal.
7. `backtest` — cost-aware simulation + quality checks.
8. `sensitivity` (optional, `--with-sensitivity`) — the Phase 8 grid.
9. `report` — always last: refreshes
   `data/reports/platform_summary.json` from whatever state the lake is
   actually in (even after a mid-chain failure).

### Restartability

Each stage appends its own record to `data/reports/runs/run_log.jsonl`
(run id, stage, status, record count). An aborted update is fixed by
repairing the cause and rerunning `update` — completed stages simply run
again deterministically (the lake is rebuilt, not appended-to, except for
ingestion batches).

### Freshness

`data/reports/platform_summary.json` (written by `update`, or on demand
by `uv run asx_quant report`) contains per-artifact UTC mtimes
(`artifacts` block) and per-stage quality statuses. The dashboard Home
page renders the same snapshot.

## Scheduled weekly refresh (Windows)

Run in Task Scheduler or `schtasks`:

```
schtasks /Create /TN "ASXQuant Weekly Update" ^
  /TR "powershell -ExecutionPolicy Bypass -File C:\Users\henry\projects\asx-quant-ml\scripts\update.ps1" ^
  /SC WEEKLY /D MON /ST 07:00
```

Weekly cadence matches the portfolio's rebalancing frequency; the update
runs *before* the market open so the artifacts reflect the prior close.
Logs land in `data/reports/runs/run_log.jsonl`; a non-zero exit code
means a stage failed (check the log tail for the aborted stage).

## Notebooks

Two executed notebooks complement the CLI (regenerate with
`uv run --extra dev python scripts/build_notebooks.py`):

| Notebook | Purpose |
|---|---|
| `notebooks/01_eda.ipynb` | EDA: target description, feature-vs-target correlations, per-ticker ICs, selection precision, error analysis |
| `notebooks/02_end_to_end_workflow.ipynb` | Calls the stage functions directly and runs the incremental chain end-to-end; shows the latest trades; documents the full-refresh regime |

## Dashboard

```
uv run streamlit run dashboard/Home.py        # or scripts\dashboard.ps1
```

Pages:

| Page | Contents |
|---|---|
| Home | headline NAV/benchmark/quality, per-stage status table, latest run |
| Backtest | equity curve vs benchmark, drawdown, metrics, diagnostics, trades |
| Portfolio | holdings/cash/turnover per rebalance, latest target weights |
| Sensitivity | grid heat map + full table, skipped axes, NOT-a-leaderboard banner |
| Data quality | per-stage quality reports with rule detail, run log |

The dashboard is read-only over the lake: it never runs a pipeline stage
and renders "run stage X first" guidance for missing artifacts. Every
page carries the research-only disclaimer.

## Model/config changes

Selection-relevant decisions (signal model, `n_holdings`, weights method,
cost scenario) live in `config/*.yaml` and were fixed BEFORE the Phase 8
grid. The sensitivity table is diagnostics only — see
`data/reports/sensitivity_report.json` for the exact wording.
