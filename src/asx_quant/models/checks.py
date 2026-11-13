"""Automated quality rules for model training runs.

Structural invariants (error severity — abort the stage):

* folds were generated and every fold's purge/embargo can be PROVEN
  (train label availability strictly before the test window);
* every enabled model produced predictions for every evaluated fold;
* predictions are finite; classifier scores are probabilities in [0, 1];
* the evaluation boundary (``evaluation_test_end``) was respected — no
  evaluation-group fold reaches into the held-out backtest window;
* the feature set version stamped in gold matches the training config
  (checked at load time; recorded here for the report).

Warnings document observed behaviour that is not a violation: skipped
single-class folds, class imbalance levels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from asx_quant.quality.checks import QualityReport, RuleResult


def run_training_quality_checks(
    folds: list[dict],
    predictions: pd.DataFrame,
    model_names: list[str],
    evaluation_test_end: str | None,
    feature_set_version: str,
    expected_feature_set_version: str,
    dataset_name: str = "model_training",
) -> QualityReport:
    """Quality rules over one training run's fold outcomes + predictions."""
    report = QualityReport(dataset=dataset_name)

    def _rule(name: str, severity: str, passed: bool, failures: int = 0,
              sample: list[str] | None = None, detail: str = "") -> None:
        report.results.append(RuleResult(
            name=name, severity=severity, passed=passed,
            failures=int(failures), sample=sample or [], detail=detail,
        ))

    _rule("feature_set_version_matches_config", "error",
          feature_set_version == expected_feature_set_version, 0,
          detail=f"gold stamped '{feature_set_version}', config expects "
                 f"'{expected_feature_set_version}'.")

    _rule("folds_generated", "error", len(folds) > 0, 0,
          detail="Walk-forward produced no folds — check train.yaml cv "
                 "windows against the data history.")

    # THE leakage rule: every fold must be purge-clean.
    violations = [
        f["fold_id"] for f in folds if not f.get("purge_verified", False)
    ]
    _rule("purge_verified_every_fold", "error", not violations,
          len(violations), [f"fold_{v}" for v in violations[:10]],
          "Every training row's label availability date must be strictly "
          "before the fold's test window start.")

    # Evaluation boundary discipline.
    boundary_reached = [
        f["fold_id"] for f in folds
        if f.get("group") == "evaluation"
        and evaluation_test_end is not None
        and f.get("test_end", "") > evaluation_test_end
    ]
    _rule("evaluation_boundary_respected", "error", not boundary_reached,
          len(boundary_reached),
          [f"fold_{v}" for v in boundary_reached[:10]],
          f"evaluation-group folds must end before {evaluation_test_end} "
          "(the final backtest window is never tuned against).")

    # Every enabled model present in every evaluated fold.
    missing = []
    for f in folds:
        for name in model_names:
            if name not in f.get("metrics", {}):
                missing.append(f"fold_{f['fold_id']}:{name}")
    _rule("all_models_scored_every_fold", "error", not missing,
          len(missing), missing[:10],
          "A model missing from a fold means it was silently skipped — "
          "usually a single-class training fold; investigate.")

    if predictions.empty:
        _rule("non_empty_predictions", "error", False, 0,
              detail="No out-of-sample predictions were produced.")
        return report
    _rule("non_empty_predictions", "error", True, 0)

    scores = pd.to_numeric(predictions["prediction"], errors="coerce")
    arr = scores.to_numpy(dtype=float, na_value=float("nan"))
    finite = bool(np.isfinite(arr).all())
    _rule("predictions_finite", "error", finite,
          int((~np.isfinite(arr)).sum()),
          detail="NaN/inf predictions indicate a broken pipeline stage.")

    classifiers = predictions[predictions["model_kind"] == "classifier"]
    if not classifiers.empty:
        out_of_bounds = classifiers[
            (classifiers["prediction"] < 0.0) | (classifiers["prediction"] > 1.0)
        ]
        _rule("classifier_scores_are_probabilities", "error",
              out_of_bounds.empty, len(out_of_bounds),
              detail="Classifier scores are P(up) and must lie in [0, 1].")

    # Warnings: class balance context for the fold base rates.
    extreme = [
        f["fold_id"] for f in folds
        if pd.notna(f.get("base_rate", float("nan")))
        and (f["base_rate"] < 0.05 or f["base_rate"] > 0.95)
    ]
    _rule("base_rate_extreme", "warn", not extreme, len(extreme),
          [f"fold_{v}" for v in extreme[:10]],
          "Fold base rates near 0/1 make accuracy meaningless; the "
          "class-weighting is active but interpret metrics carefully.")

    return report
