"""Automated data-quality rules for Gold feature and label datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from asx_quant.quality.checks import QualityReport, RuleResult


def run_feature_quality_checks(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    dataset_name: str = "gold_features",
    config_version: str = "",
) -> QualityReport:
    """Quality rules for the feature set and target labels.

    Error-severity rules guard the platform's leakage invariants:

    * no ``future_``/``direction_`` columns may exist in the FEATURE table;
    * every direction label must agree with its future return's sign;
    * every label's availability date must be strictly after its session
      (the label genuinely required future data);
    * feature values must be finite (NaN allowed — null policy
      ``null_as_missing``; inf is a computation bug).
    """
    report = QualityReport(dataset=dataset_name)

    def _rule(name: str, severity: str, passed: bool, failures: int = 0,
              sample: list[str] | None = None, detail: str = "") -> None:
        report.results.append(RuleResult(
            name=name, severity=severity, passed=passed,
            failures=int(failures), sample=sample or [], detail=detail,
        ))

    _rule("feature_set_version_stamped", "error",
          bool(config_version), 0,
          detail=f"feature_set_version='{config_version}' recorded in the "
                 "dataset metadata column.")

    if features.empty:
        _rule("non_empty_features", "error", False)
        return report
    _rule("non_empty_features", "error", True)

    # Leakage guard 1: the feature table must not contain label columns.
    leak_cols = [c for c in features.columns
                 if c.startswith("future_") or c.startswith("direction_")]
    _rule("no_label_columns_in_features", "error", not leak_cols, len(leak_cols),
          leak_cols[:10],
          "Targets live in gold/target_labels; a forward-looking column in "
          "the feature table would leak directly into training.")

    # Structural: finite values (NaN allowed, inf is a bug).
    numeric = features.select_dtypes(include="number")
    values = numeric.to_numpy(dtype=float, na_value=np.nan)
    inf_total = int(np.isinf(values).sum())
    inf_cols = numeric.columns[np.isinf(values).any(axis=0)]
    _rule("feature_values_finite", "error", inf_total == 0, inf_total,
          [str(c) for c in inf_cols[:10]],
          "NaN is the documented null policy; +/-inf indicates a division "
          "bug or corrupted source value.")

    # Duplicates in the grid.
    dup = features.duplicated(subset=["ticker", "date"], keep=False)
    _rule("no_duplicate_ticker_date", "error", not dup.any(), dup.sum())

    if targets.empty:
        _rule("non_empty_targets", "error", False)
        return report
    _rule("non_empty_targets", "error", True)

    # Leakage guard 2: labels must agree with their future returns.
    for horizon in (3, 10):
        ret_col, dir_col = f"future_return_{horizon}d", f"direction_{horizon}d"
        if ret_col in targets.columns and dir_col in targets.columns:
            both = targets[[ret_col, dir_col]].dropna()
            mismatch = (
                ((both[ret_col] > 0) & (both[dir_col] != 1.0))
                | ((both[ret_col] <= 0) & (both[dir_col] != 0.0))
            )
            _rule(f"direction_matches_return_{horizon}d", "error",
                  not mismatch.any(), int(mismatch.sum()))

    # Leakage guard 3: availability date strictly after the session.
    avail_cols = [c for c in targets.columns if c.startswith(
        "label_available_date_")]
    if avail_cols and "date" in targets.columns:
        for col in avail_cols:
            avail = pd.to_datetime(targets[col])
            bad = avail <= targets["date"]
            _rule("label_available_after_session", "error", not bad.any(),
                  int(bad.sum()),
                  detail=f"{col} must be the session whose close completes "
                         "the label (strictly later than the feature "
                         "session).")

    # Coverage warnings (documented, never blocking).
    warmup = features.select_dtypes(include="number").isna().mean()
    high_nan = warmup[warmup > 0.4]
    _rule("warmup_nan_share", "warn", high_nan.empty, int(len(high_nan)),
          [f"{name}:{share:.0%}" for name, share in high_nan.items()][:10],
          "Columns with >40% NaN — mostly rolling-window warm-up and "
          "point-in-time publication gaps; never filled silently.")

    return report
