"""Walk-forward training and evaluation loop.

Per fold:

1. Build the purged/embargoed training mask and the test mask.
2. Fit every enabled model on the training rows (dropping rows with NaN
   labels), with preprocessing fit on training rows only.
3. Score the test rows; record per-model metrics and OOS predictions.

Aggregation is split into two groups by the config's ``evaluation_test_end``:

* ``evaluation`` — folds whose test window ends before the boundary. These
  numbers drive model selection.
* ``holdout``    — folds reaching into the final backtest window
  (``backtest.yaml periods.test_start``). Predictions are still produced
  (needed by Phase 6/7, trained strictly on earlier data), but the metrics
  are hold-out diagnostics and must never drive selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from asx_quant.models.dataset import feature_matrix
from asx_quant.models.fit import ModelSpec, build_model, predict_scores
from asx_quant.models.splits import Fold, assert_no_overlap, purged_train_mask, test_mask
from asx_quant.observability.logging import get_logger

logger = get_logger("models.cv")


@dataclass
class FoldOutcome:
    """Metrics and prediction counts for one fold."""

    fold_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_train: int
    n_test: int
    n_purged: int    # training rows dropped by purge + embargo combined
    base_rate: float
    group: str                     # evaluation | holdout
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)


@dataclass
class CVResult:
    """Everything the trainer needs to report and persist."""

    folds: list[FoldOutcome] = field(default_factory=list)
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    feature_columns: list[str] = field(default_factory=list)


def _classification_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    y_hat = (scores >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "log_loss": float(log_loss(y_true, scores, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, scores)),
        "accuracy": float(accuracy_score(y_true, y_hat)),
        "precision": float(precision_score(y_true, y_hat, zero_division=0)),
        "recall": float(recall_score(y_true, y_hat, zero_division=0)),
    }


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Rank correlation without scipy: Pearson on ranks."""
    if len(y_true) < 2:
        return float("nan")
    ra = pd.Series(y_true).rank().to_numpy()
    rb = pd.Series(y_pred).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": _spearman(y_true, y_pred),
    }


def run_walk_forward(
    frame: pd.DataFrame,
    folds: list[Fold],
    specs: list[ModelSpec],
    sessions: pd.DatetimeIndex,
    horizon: int,
    classification_label: str,
    regression_label: str,
    embargo_sessions: int = 0,
    evaluation_test_end: pd.Timestamp | None = None,
    feature_cols: list[str] | None = None,
) -> CVResult:
    """Train and evaluate every enabled model over every fold.

    Returns:
        :class:`CVResult` with per-fold metrics and the long-format OOS
        prediction frame (one row per ticker/date/model, later folds win
        for overlapping test windows — applied by the caller's dedup).
    """
    cols = feature_cols or []
    X_all = feature_matrix(frame, cols)
    y_class = frame[classification_label].to_numpy(dtype=float)
    y_reg = frame[regression_label].to_numpy(dtype=float)

    records: list[FoldOutcome] = []
    pred_parts: list[pd.DataFrame] = []

    for fold in folds:
        train_mask = purged_train_mask(frame, fold, sessions, horizon,
                                       embargo_sessions=embargo_sessions)
        test_mask_ = test_mask(frame, fold)
        assert_no_overlap(frame, fold, train_mask, horizon)

        n_train_window = int(((frame["date"] >= fold.train_start)
                              & (frame["date"] <= fold.train_end)).sum())
        purged_dropped = n_train_window - int(train_mask.sum())

        group = "evaluation"
        if evaluation_test_end is not None and fold.test_end >= evaluation_test_end:
            group = "holdout"

        outcome = FoldOutcome(
            fold_id=fold.fold_id,
            train_start=str(fold.train_start.date()),
            train_end=str(fold.train_end.date()),
            test_start=str(fold.test_start.date()),
            test_end=str(fold.test_end.date()),
            n_train=int(train_mask.sum()),
            n_test=int(test_mask_.sum()),
            n_purged=int(purged_dropped),
            base_rate=float(np.nanmean(y_class[test_mask_.to_numpy()])),
            group=group,
        )

        train_idx = train_mask.to_numpy()
        test_idx = test_mask_.to_numpy()
        test_rows = frame.loc[test_idx]

        for spec in specs:
            y = y_class if spec.kind == "classifier" else y_reg
            y_train = y[train_idx]
            valid = ~np.isnan(y_train)
            single_class = (
                spec.kind == "classifier"
                and len(np.unique(y_train[valid])) < 2
            )
            if valid.sum() < 2 or single_class:
                logger.warning(
                    "Fold %d: model '%s' skipped (insufficient or "
                    "single-class training labels).", fold.fold_id, spec.name)
                continue
            X_train = X_all.to_numpy()[train_idx][valid]
            model = build_model(
                spec,
                n_positive=int((y_train[valid] == 1).sum()),
                n_negative=int((y_train[valid] == 0).sum()),
            )
            model.fit(X_train, y_train[valid])

            y_test = y[test_idx]
            valid_test = ~np.isnan(y_test)
            if valid_test.sum() == 0:
                logger.warning(
                    "Fold %d: model '%s' has no labelled test rows.",
                    fold.fold_id, spec.name)
                continue
            scores = predict_scores(model, X_all.to_numpy()[test_idx][valid_test],
                                    spec.kind)

            if spec.kind == "classifier":
                metrics = _classification_metrics(y_test[valid_test], scores)
            else:
                metrics = _regression_metrics(y_test[valid_test], scores)
            outcome.metrics[spec.name] = metrics

            pred_parts.append(pd.DataFrame({
                "ticker": test_rows["ticker"].to_numpy()[valid_test],
                "date": test_rows["date"].to_numpy()[valid_test],
                "model_name": spec.name,
                "model_kind": spec.kind,
                "fold_id": fold.fold_id,
                "trained_through": fold.train_end,
                "prediction": scores,
            }))
            logger.info(
                "Fold %d [%s] %s: %s", fold.fold_id, group, spec.name,
                {k: round(v, 4) for k, v in metrics.items()})

        records.append(outcome)

    predictions = (
        pd.concat(pred_parts, ignore_index=True) if pred_parts
        else pd.DataFrame(columns=["ticker", "date", "model_name",
                                   "model_kind", "fold_id",
                                   "trained_through", "prediction"])
    )
    return CVResult(folds=records, predictions=predictions,
                    feature_columns=list(cols))


def aggregate_metrics(result: CVResult) -> dict[str, dict[str, float]]:
    """Mean per-model metrics over evaluation-group folds (selection-safe)."""
    agg: dict[str, dict[str, float]] = {}
    eval_folds = [f for f in result.folds if f.group == "evaluation"]
    if not eval_folds:
        return agg
    model_names = sorted({name for f in eval_folds for name in f.metrics})
    for name in model_names:
        series: dict[str, list[float]] = {}
        for fold in eval_folds:
            for metric, value in fold.metrics.get(name, {}).items():
                series.setdefault(metric, []).append(value)
        agg[name] = {
            metric: float(np.mean(values))
            for metric, values in series.items() if values
        }
    return agg


def holdout_metrics(result: CVResult) -> dict[str, dict[str, float]]:
    """Mean per-model metrics over holdout-group folds (diagnostics only)."""
    agg: dict[str, dict[str, float]] = {}
    hold_folds = [f for f in result.folds if f.group == "holdout"]
    if not hold_folds:
        return agg
    model_names = sorted({name for f in hold_folds for name in f.metrics})
    for name in model_names:
        series: dict[str, list[float]] = {}
        for fold in hold_folds:
            for metric, value in fold.metrics.get(name, {}).items():
                series.setdefault(metric, []).append(value)
        agg[name] = {
            metric: float(np.mean(values))
            for metric, values in series.items() if values
        }
    return agg


def latest_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Collapse overlapping test windows: the most recent fold wins."""
    if predictions.empty:
        return predictions
    ordered = predictions.sort_values(["ticker", "date", "model_name", "fold_id"])
    return ordered.drop_duplicates(
        subset=["ticker", "date", "model_name"], keep="last",
    ).reset_index(drop=True)


def metrics_to_payload(result: CVResult) -> dict[str, Any]:
    """JSON-ready evaluation summary for reports and the dashboard."""
    evaluation = aggregate_metrics(result)
    holdout = holdout_metrics(result)
    return {
        "n_folds": len(result.folds),
        "n_folds_evaluation": len([f for f in result.folds
                                   if f.group == "evaluation"]),
        "n_folds_holdout": len([f for f in result.folds
                                if f.group == "holdout"]),
        "evaluation": evaluation,
        "holdout_diagnostics_only": holdout,
        "per_fold": [
            {
                "fold_id": f.fold_id,
                "train": [f.train_start, f.train_end],
                "test": [f.test_start, f.test_end],
                "n_train": f.n_train,
                "n_test": f.n_test,
                "n_purged": f.n_purged,
                "base_rate": f.base_rate,
                "group": f.group,
                "metrics": f.metrics,
            }
            for f in result.folds
        ],
    }
