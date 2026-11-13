"""Post-training verification of Phase 5 outputs."""
import json
from pathlib import Path

import pandas as pd

# 1. quality report
report = json.loads(Path("data/reports/model_quality.json").read_text())
print("model_quality status:", report["status"],
      "| warnings:", report["warn_failures"])
for r in report["rules"]:
    if not r["passed"]:
        print(f"  [{r['severity']}] {r['name']}: {r['failures']}")

# 2. metrics payload: folds, groups, purge discipline
m = json.loads(Path("data/reports/model_metrics.json").read_text())
print("\nfolds:", m["n_folds"], "| evaluation:", m["n_folds_evaluation"],
      "| holdout:", m["n_folds_holdout"])
boundary = "2024-07-01"
eval_folds = [f for f in m["per_fold"] if f["group"] == "evaluation"]
hold_folds = [f for f in m["per_fold"] if f["group"] == "holdout"]
print("evaluation fold test ends <= boundary:",
      all(f["test"][1] < boundary for f in eval_folds))
print("holdout fold test ends >= boundary:",
      all(f["test"][1] >= boundary for f in hold_folds))
purge_counts = [f["n_purged"] for f in m["per_fold"]]
print("purged rows per fold: min", min(purge_counts), "max", max(purge_counts))
print("\nevaluation aggregate:")
print(json.dumps(m["evaluation"], indent=2))
print("holdout diagnostics (not used for selection):")
print(json.dumps(m["holdout_diagnostics_only"], indent=2))

# base rates
rates = [f["base_rate"] for f in m["per_fold"]]
print("\nbase rate range:", round(min(rates), 4), "-", round(max(rates), 4))

# 3. predictions
p = pd.read_parquet("data/gold/predictions.parquet")
print("\npredictions:", len(p), "| cols:", list(p.columns))
print("models:", p["model_name"].unique().tolist())
print("date range:", p["date"].min().date(), "->", p["date"].max().date())
print("dups per (ticker,date,model):",
      int(p.duplicated(subset=["ticker", "date", "model_name"]).sum()))
print("predictions finite:", bool(p["prediction"].notna().all()))
clf = p[p["model_kind"] == "classifier"]
print("classifier probs in [0,1]:",
      bool(clf["prediction"].between(0, 1).all()))
print("per-model rows:")
print(p.groupby(["model_name", "model_kind"]).size().to_string())
print("\nno-prediction rows check: first prediction date >= first train end",
      p["date"].min())

# 4. artifacts
models_dir = Path("data/models/v0.1")
print("\nartifacts:", [f.name for f in models_dir.iterdir()])
meta = json.loads((models_dir / "hist_gbt.json").read_text())
print("hist_gbt trained_through:", meta["trained_through"],
      "| n_train_rows:", meta["n_train_rows"])

# 5. which feature is column 57 (the all-NaN imputer warning)?
feat_cols = m["feature_columns"]
print("\nfeature col 57:", feat_cols[57])
