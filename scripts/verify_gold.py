"""Post-fix verification of the gold feature build."""
import json
from pathlib import Path

import pandas as pd

f = pd.read_parquet(Path("data/gold/features.parquet"))
t = pd.read_parquet(Path("data/gold/target_labels.parquet"))
print("features:", len(f), "| cols:", len(f.columns))
print("targets :", len(t), "| labelled 3d:", int(t["future_return_3d"].notna().sum()),
      "| labelled 10d:", int(t["future_return_10d"].notna().sum()))

# 1. benchmark features sane
bench_nan = f["bench_return_1d"].isna().mean()
print(f"bench_return_1d NaN share: {bench_nan:.4f} (expect ~0.0004)")
print(f"max |bench_return_1d|: {f['bench_return_1d'].abs().max():.4f} (expect <= ~0.097)")
for c in ["bench_return_20d", "bench_return_100d", "bench_close_to_sma_20",
          "bench_close_to_sma_100", "bench_volatility_20d"]:
    s = f[c].dropna()
    print(f"  {c}: NaN share {f[c].isna().mean():.4f}, "
          f"min {s.min():.4f}, max {s.max():.4f}")

# 2. beta populated and O(1)
beta = f["beta_252d"].dropna()
print(f"beta_252d: NaN share {f['beta_252d'].isna().mean():.4f}, "
      f"count {len(beta)}, median {beta.median():.3f}, "
      f"p5 {beta.quantile(0.05):.3f}, p95 {beta.quantile(0.95):.3f}")

# 3. relative momentum sane
for c in ["momentum_rel_bench_20d", "momentum_rel_bench_63d"]:
    s = f[c].dropna()
    print(f"{c}: NaN {f[c].isna().mean():.4f}, min {s.min():.4f}, max {s.max():.4f}")

# 4. spot check CBA tail rows that were corrupt before
cba = f[f["ticker"] == "CBA.AX"].set_index("date")
print("\nCBA last 3 rows (previously corrupt):")
print(cba[["bench_return_1d", "bench_return_20d", "beta_252d",
           "momentum_rel_bench_20d"]].tail(3).to_string())

# 5. NaN>40% columns
high_nan = [c for c in f.columns
            if f[c].dtype != object and f[c].isna().mean() > 0.4]
print("\nNaN>40% columns:", high_nan)

# 6. quality report
rep = json.loads(Path("data/reports/data_quality_features.json").read_text())
print("\nquality status:", rep.get("status"))
for r in rep.get("results", []):
    if not r["passed"]:
        print(f"  [{r['severity']}] {r['name']}: {r.get('failures')} failures")
