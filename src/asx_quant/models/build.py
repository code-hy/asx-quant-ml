"""Model training orchestrator: Gold -> walk-forward CV -> artifacts.

Pipeline:

1. Load gold ``features`` + ``target_labels`` (version-checked) and assemble
   the training frame and feature matrix.
2. Generate purged/embargoed walk-forward folds from ``config/train.yaml``.
3. Fit + evaluate every enabled model per fold; collect OOS predictions.
4. Run model-quality rules; save the report BEFORE asserting.
5. Write ``gold/predictions.parquet`` (latest-fold-wins per ticker/date/
   model), per-fold metrics report, and final model artifacts under
   ``data/models/<model_version>/`` (trained on all labelled history;
   evaluation uses ONLY walk-forward folds).

Leakage invariants enforced here (and by tests):

* train rows' label availability strictly before the fold's test window
  (purge) and outside the embargo gap;
* preprocessing (imputation/scaling) fit on training rows only;
* the final backtest window is excluded from evaluation aggregates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from asx_quant.config.loader import load_yaml
from asx_quant.exceptions import StorageError
from asx_quant.models import checks as model_checks
from asx_quant.models.cv import (
    CVResult,
    latest_predictions,
    metrics_to_payload,
    run_walk_forward,
)
from asx_quant.models.dataset import feature_columns, load_training_frame
from asx_quant.models.fit import ModelSpec
from asx_quant.models.splits import walk_forward_splits
from asx_quant.observability.logging import get_logger
from asx_quant.quality.checks import QualityReport
from asx_quant.storage.paths import DataLake

logger = get_logger("models.build")


@dataclass
class TrainResult:
    """Paths and outcomes of one training run."""

    predictions_path: Path
    metrics_path: Path
    report: QualityReport | None
    model_version: str
    n_folds: int
    n_models: int
    evaluation_metrics: dict
    holdout_metrics: dict


def train_models(
    lake: DataLake,
    settings,  # noqa: ANN001
    registry,  # noqa: ANN001
    with_quality: bool = True,
) -> TrainResult:
    """Run the full configured training pipeline into Gold + artifacts."""
    cfg = load_yaml("train").get("train", {})
    version = str(cfg.get("model_version", "v0"))
    target_cfg = cfg.get("target", {})
    cv_cfg = cfg.get("cv", {})
    columns_cfg = cfg.get("columns", {})
    horizon = int(target_cfg.get("horizon", 3))
    classification_label = str(
        target_cfg.get("classification_label", f"direction_{horizon}d"))
    regression_label = str(
        target_cfg.get("regression_label", f"future_return_{horizon}d"))

    frame = load_training_frame(
        lake,
        features_dataset=str(cfg.get("features_dataset", "features")),
        targets_dataset=str(cfg.get("targets_dataset", "target_labels")),
        feature_set_version=str(cfg.get("feature_set_version", "v0.1")),
    )

    sessions = pd.DatetimeIndex(sorted(frame["date"].unique()))
    folds = walk_forward_splits(
        sessions,
        train_window_sessions=cv_cfg.get("train_window_sessions"),
        test_window_sessions=int(cv_cfg.get("test_window_sessions", 252)),
        step_sessions=int(cv_cfg.get("step_sessions", 63)),
        min_train_sessions=int(cv_cfg.get("min_train_sessions", 1)),
        max_folds=cv_cfg.get("max_folds"),
    )
    if not folds:
        raise StorageError(
            "No walk-forward folds fit the configured windows; check "
            "config/train.yaml cv settings against the data history."
        )
    logger.info("Walk-forward: %d folds, %d sessions", len(folds), len(sessions))

    specs = [
        ModelSpec.from_config(name, mcfg)
        for name, mcfg in cfg.get("models", {}).items()
        if isinstance(mcfg, dict) and mcfg.get("enabled", False)
    ]
    if not specs:
        raise StorageError("No models enabled in config/train.yaml.")

    evaluation_end = cv_cfg.get("evaluation_test_end")
    evaluation_end_ts = (
        pd.Timestamp(evaluation_end) if evaluation_end else None
    )

    cols = feature_columns(frame, columns_cfg.get("exclude", []))
    logger.info("Feature matrix: %d columns", len(cols))

    result: CVResult = run_walk_forward(
        frame,
        folds=folds,
        specs=specs,
        sessions=sessions,
        horizon=horizon,
        classification_label=classification_label,
        regression_label=regression_label,
        embargo_sessions=int(cv_cfg.get("embargo_sessions", 0)),
        evaluation_test_end=evaluation_end_ts,
        feature_cols=cols,
    )

    # Fold records for the quality report (purge verified by construction +
    # asserted inside run_walk_forward via assert_no_overlap).
    fold_records = [
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
            "purge_verified": True,
        }
        for f in result.folds
    ]

    report = None
    if with_quality:
        report = model_checks.run_training_quality_checks(
            fold_records,
            result.predictions,
            model_names=[s.name for s in specs],
            evaluation_test_end=str(evaluation_end) if evaluation_end else None,
            feature_set_version=str(
                frame["feature_set_version"].iloc[0]),
            expected_feature_set_version=str(
                cfg.get("feature_set_version", "v0.1")),
        )
        report.save(lake.report_path("model_quality", ".json"))
        report.assert_passed()
        logger.info("Model quality: %s (%d warnings)",
                    report.to_dict()["status"], report.warn_count())

    # ---- persist: predictions (latest fold wins) + metrics + artifacts ----
    predictions = latest_predictions(result.predictions)
    predictions = predictions.assign(
        model_version=version,
        feature_set_version=str(cfg.get("feature_set_version", "v0.1")),
    )
    predictions_path = lake.gold_dataset_path(
        cfg.get("outputs", {}).get("predictions_dataset", "predictions"))
    predictions.to_parquet(predictions_path, index=False)
    logger.info("Predictions written: %d rows -> %s",
                len(predictions), predictions_path)

    payload = metrics_to_payload(result)
    payload.update({
        "run_id": registry.run_id,
        "model_version": version,
        "feature_set_version": str(cfg.get("feature_set_version", "v0.1")),
        "feature_columns": cols,
        "models": {s.name: {"kind": s.kind, "estimator": s.estimator}
                   for s in specs},
        "n_prediction_rows": len(predictions),
    })
    metrics_path = lake.report_path("model_metrics", ".json")
    metrics_path.write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8")

    if cfg.get("outputs", {}).get("save_final_artifacts", True):
        _save_final_artifacts(lake, settings, registry, cfg, version, frame,
                              cols, specs, horizon, classification_label,
                              regression_label, payload)

    evaluation_metrics = payload["evaluation"]
    holdout_metrics = payload["holdout_diagnostics_only"]
    return TrainResult(
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        report=report,
        model_version=version,
        n_folds=len(result.folds),
        n_models=len(specs),
        evaluation_metrics=evaluation_metrics,
        holdout_metrics=holdout_metrics,
    )


def _save_final_artifacts(
    lake: DataLake,
    settings,  # noqa: ANN001
    registry,  # noqa: ANN001
    cfg: dict,
    version: str,
    frame: pd.DataFrame,
    cols: list[str],
    specs: list[ModelSpec],
    horizon: int,
    classification_label: str,
    regression_label: str,
    payload: dict,
) -> None:
    """Refit each enabled model on ALL labelled history and persist it.

    The final artifacts exist for Phase 6+ inference. They are NEVER used
    for evaluation — every reported metric comes from walk-forward folds.
    """
    import joblib

    from asx_quant.models.fit import build_model

    out_dir = lake.models_dir / version
    out_dir.mkdir(parents=True, exist_ok=True)

    X = frame[cols].astype("float32")
    for spec in specs:
        label = (classification_label if spec.kind == "classifier"
                 else regression_label)
        y = frame[label].to_numpy(dtype=float)
        valid = ~pd.isna(y)
        model = build_model(
            spec,
            n_positive=int((y[valid] == 1).sum()),
            n_negative=int((y[valid] == 0).sum()),
        )
        model.fit(X.to_numpy()[valid], y[valid])
        path = out_dir / f"{spec.name}.joblib"
        joblib.dump(model, path)
        metadata = {
            "model_name": spec.name,
            "model_version": version,
            "kind": spec.kind,
            "estimator": spec.estimator,
            "feature_set_version": str(cfg.get("feature_set_version", "v0.1")),
            "feature_columns": cols,
            "label": label,
            "trained_through": str(frame.loc[valid, "date"].max().date()),
            "n_train_rows": int(valid.sum()),
            "run_id": registry.run_id,
            "walk_forward_evaluation": payload.get("evaluation", {}).get(
                spec.name),
            "note": "Evaluation metrics come exclusively from the walk-forward "
                    "folds; this artifact is for inference only.",
        }
        (out_dir / f"{spec.name}.json").write_text(
            json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        logger.info("Saved final artifact: %s", path)
