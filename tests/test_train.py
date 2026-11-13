"""Phase 5 tests: purged walk-forward splits, training CV, quality rules.

All offline. The synthetic frame embeds an informative feature whose value
at session t predicts the 3-session-forward return, plus pure-noise
features — models trained on the informative signal must beat chance
out-of-sample, noise-only models must not, and EVERY fold must provably
respect the purge/embargo rules.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from asx_quant.exceptions import AsxQuantError
from asx_quant.models import checks as model_checks
from asx_quant.models.cv import (
    aggregate_metrics,
    latest_predictions,
    metrics_to_payload,
    run_walk_forward,
)
from asx_quant.models.dataset import feature_columns
from asx_quant.models.fit import ModelSpec
from asx_quant.models.splits import (
    assert_no_overlap,
    purged_train_mask,
    walk_forward_splits,
)
from asx_quant.models.splits import (
    test_mask as fold_test_mask,  # aliased: pytest would collect 'test_mask'
)

# ------------------------------------------------------------ synthetic frame
N_SESSIONS = 600
SESSIONS = pd.bdate_range("2023-01-02", periods=N_SESSIONS)
TICKERS = ["CBA.AX", "BHP.AX", "WES.AX"]
SIGNAL_STRENGTH = 0.006
NOISE_STD = 0.01


def _synthetic_frame(seed: int = 7) -> pd.DataFrame:
    """Frame with one informative feature + two noise features + labels."""
    rng = np.random.default_rng(seed)
    dates = pd.Series(SESSIONS)
    # availability: the session whose close completes the 3d label
    avail = dates.shift(-3)          # NaT for the last 3 sessions
    parts = []
    for ticker in TICKERS:
        n = N_SESSIONS
        signal = rng.normal(0.0, 1.0, n)
        noise_a = rng.normal(0.0, 1.0, n)
        noise_b = rng.normal(0.0, 1.0, n)
        future_return = SIGNAL_STRENGTH * signal + rng.normal(
            0.0, NOISE_STD, n)
        parts.append(pd.DataFrame({
            "ticker": ticker,
            "date": dates,
            "alpha_signal": signal,
            "noise_a": noise_a,
            "noise_b": noise_b,
            "future_return_3d": future_return,
            "direction_3d": (future_return > 0).astype(float),
            "label_available_date_3d": avail,
        }))
    frame = pd.concat(parts, ignore_index=True)
    # last 3 sessions per ticker have incomplete labels
    for col in ("future_return_3d", "direction_3d",
                "label_available_date_3d"):
        frame.loc[frame.groupby("ticker").tail(3).index, col] = np.nan
    frame["feature_set_version"] = "v0.1"
    return frame


FRAME = _synthetic_frame()


def _specs() -> list[ModelSpec]:
    return [
        ModelSpec.from_config("logistic", {
            "kind": "classifier", "estimator": "logistic_regression",
            "preprocessing": "impute_median_scale",
            "params": {"C": 0.1, "max_iter": 500},
            "class_weight": "balanced",
        }),
        ModelSpec.from_config("hist_gbt", {
            "kind": "classifier", "estimator": "hist_gradient_boosting",
            "preprocessing": "none",
            "params": {"max_iter": 30, "learning_rate": 0.1,
                       "max_leaf_nodes": 15, "min_samples_leaf": 50,
                       "early_stopping": False},
            "class_weight": "balanced",
        }),
        ModelSpec.from_config("ridge", {
            "kind": "regressor", "estimator": "ridge",
            "preprocessing": "impute_median_scale",
            "params": {"alpha": 1.0},
        }),
        ModelSpec.from_config("hist_gbt_reg", {
            "kind": "regressor", "estimator": "hist_gradient_boosting_regressor",
            "preprocessing": "none",
            "params": {"max_iter": 30, "learning_rate": 0.1,
                       "max_leaf_nodes": 15, "min_samples_leaf": 50,
                       "early_stopping": False},
        }),
    ]


def _folds() -> list:
    return walk_forward_splits(
        SESSIONS, train_window_sessions=200, test_window_sessions=60,
        step_sessions=60, min_train_sessions=100,
    )


# ------------------------------------------------------------------- splits
def test_walk_forward_splits_windows() -> None:
    folds = _folds()
    assert len(folds) >= 5
    first = folds[0]
    assert first.test_start == SESSIONS[200]      # after the train window
    assert first.train_end == SESSIONS[199]
    # consecutive folds advance by exactly `step`
    for a, b in zip(folds, folds[1:], strict=False):
        assert (b.test_start - a.test_start) == pd.Timedelta(
            SESSIONS[260] - SESSIONS[200])
    # last fold's test window is truncated at the data end
    assert folds[-1].test_end == SESSIONS[-1]


def test_walk_forward_splits_expanding() -> None:
    folds = walk_forward_splits(SESSIONS, train_window_sessions=None,
                                test_window_sessions=60, step_sessions=120,
                                min_train_sessions=100)
    assert all(f.train_start == SESSIONS[0] for f in folds)


def test_purge_mask_excludes_labels_reaching_into_test() -> None:
    fold = _folds()[0]
    horizon = 3
    mask = purged_train_mask(FRAME, fold, SESSIONS, horizon, embargo_sessions=0)
    # every trainable row's label availability strictly before test start
    avail = pd.to_datetime(FRAME.loc[mask, "label_available_date_3d"])
    assert (avail < fold.test_start).all()
    assert_no_overlap(FRAME, fold, mask, horizon)

    # unpurged window would violate -> assert_no_overlap must raise
    date = FRAME["date"]
    raw_mask = (date >= fold.train_start) & (date <= fold.train_end)
    avail_raw = pd.to_datetime(FRAME.loc[raw_mask, "label_available_date_3d"])
    assert (avail_raw >= fold.test_start).any()  # the boundary IS leaky
    with pytest.raises(ValueError, match="purge violation"):
        assert_no_overlap(FRAME, fold, raw_mask, horizon)


def test_embargo_drops_boundary_sessions() -> None:
    """Embargo beyond the horizon drops extra boundary sessions.

    With horizon 3 the purge already removes the last 3 train sessions
    (their label availability reaches the test window), so an embargo of
    10 sessions drops 7 ADDITIONAL sessions in front of the test window.
    """
    fold = _folds()[0]
    plain = purged_train_mask(FRAME, fold, SESSIONS, 3, embargo_sessions=0)
    embargoed = purged_train_mask(FRAME, fold, SESSIONS, 3, embargo_sessions=10)
    dropped = pd.DatetimeIndex(
        FRAME.loc[plain & ~embargoed, "date"].unique()).sort_values()
    assert len(dropped) == 7
    test_pos = SESSIONS.searchsorted(fold.test_start)
    assert dropped.min() == SESSIONS[test_pos - 10]
    assert dropped.max() == SESSIONS[test_pos - 4]  # purge covers [-3, -1]


def test_embargo_within_horizon_adds_nothing() -> None:
    """embargo_sessions <= horizon is subsumed by the purge."""
    fold = _folds()[0]
    plain = purged_train_mask(FRAME, fold, SESSIONS, 3, embargo_sessions=0)
    embargoed = purged_train_mask(FRAME, fold, SESSIONS, 3, embargo_sessions=3)
    assert (plain == embargoed).all()


def test_purge_rejects_missing_availability() -> None:
    fold = _folds()[0]
    broken = FRAME.copy()
    broken_rows = broken.index[:5]   # first 5 rows: inside the train window
    broken.loc[broken_rows, "label_available_date_3d"] = pd.NaT
    mask = purged_train_mask(broken, fold, SESSIONS, 3, embargo_sessions=0)
    # rows with missing availability can never be proven leak-free
    assert not mask.loc[broken_rows].any()
    assert_no_overlap(broken, fold, mask, 3)  # the surviving mask is clean


def test_test_mask_and_no_overlap_nan_raises() -> None:
    fold = _folds()[0]
    mask = fold_test_mask(FRAME, fold)
    assert FRAME.loc[mask, "date"].between(fold.test_start,
                                           fold.test_end).all()
    # a claimed train row with missing availability is a hard error
    broken = FRAME.copy()
    broken.loc[mask, "label_available_date_3d"] = pd.NaT
    with pytest.raises(ValueError, match="cannot be proven leak-free"):
        assert_no_overlap(broken, fold, mask, 3)


# ----------------------------------------------------------------- columns
def test_feature_columns_policy() -> None:
    cols = feature_columns(FRAME)
    assert cols == ["alpha_signal", "noise_a", "noise_b"]

    # label-prefixed columns are always filtered out, even new ones
    guarded = feature_columns(FRAME.assign(**{"future_x": 1.0,
                                              "direction_x": 1.0}))
    assert "future_x" not in guarded and "direction_x" not in guarded

    meta_only = FRAME[["ticker", "date", "feature_set_version"]]
    with pytest.raises(ValueError, match="No numeric feature columns"):
        feature_columns(meta_only)

    cols2 = feature_columns(FRAME, extra_exclude=["noise_a"])
    assert cols2 == ["alpha_signal", "noise_b"]


# --------------------------------------------------------------------- CV
def test_walk_forward_cv_end_to_end() -> None:
    folds = _folds()
    result = run_walk_forward(
        FRAME, folds=folds, specs=_specs(), sessions=SESSIONS, horizon=3,
        classification_label="direction_3d",
        regression_label="future_return_3d",
        embargo_sessions=3,
        feature_cols=feature_columns(FRAME),
    )

    assert len(result.folds) == len(folds)
    for fold in result.folds:
        # every enabled model scored every fold
        assert set(fold.metrics) == {s.name for s in _specs()}
        # informative model beats chance out-of-sample
        assert fold.metrics["logistic"]["roc_auc"] > 0.60
        assert fold.metrics["hist_gbt"]["roc_auc"] > 0.55
        # regression recovers the signal direction
        assert fold.metrics["ridge"]["spearman"] > 0.30
        assert fold.metrics["hist_gbt_reg"]["spearman"] > 0.25
        # purge actually dropped boundary rows
        assert fold.n_purged > 0

    preds = result.predictions
    assert not preds.empty
    clf = preds[preds["model_kind"] == "classifier"]
    assert clf["prediction"].between(0.0, 1.0).all()
    assert preds["prediction"].notna().all()
    # every test row that has a label got a prediction from every model
    assert (preds.groupby("model_name")["fold_id"].nunique() == len(folds)).all()

    agg = aggregate_metrics(result)
    assert agg["logistic"]["roc_auc"] > 0.60
    assert agg["ridge"]["spearman"] > 0.3

    payload = metrics_to_payload(result)
    assert payload["n_folds"] == len(folds)
    assert "evaluation" in payload and "holdout_diagnostics_only" in payload


def test_cv_noise_model_does_not_beat_chance() -> None:
    """A model given ONLY noise features must not look informative."""
    noise_frame = FRAME[["ticker", "date", "noise_a", "noise_b",
                         "future_return_3d", "direction_3d",
                         "label_available_date_3d", "feature_set_version"]]
    spec = ModelSpec.from_config("noise_logistic", {
        "kind": "classifier", "estimator": "logistic_regression",
        "preprocessing": "impute_median_scale",
        "params": {"C": 0.1, "max_iter": 500},
        "class_weight": "balanced",
    })
    result = run_walk_forward(
        noise_frame, folds=_folds(), specs=[spec], sessions=SESSIONS,
        horizon=3, classification_label="direction_3d",
        regression_label="future_return_3d",
        feature_cols=feature_columns(noise_frame),
    )
    agg = aggregate_metrics(result)
    assert abs(agg["noise_logistic"]["roc_auc"] - 0.5) < 0.15


def test_latest_predictions_keep_last_fold() -> None:
    a = pd.DataFrame({
        "ticker": ["CBA.AX"] * 2, "date": [pd.Timestamp("2024-01-02")] * 2,
        "model_name": ["m"] * 2, "model_kind": ["classifier"] * 2,
        "fold_id": [1, 3], "trained_through": [pd.Timestamp("2023-01-01")] * 2,
        "prediction": [0.1, 0.9],
    })
    out = latest_predictions(a)
    assert len(out) == 1 and out["prediction"].iloc[0] == 0.9


def test_aggregate_splits_evaluation_and_holdout() -> None:
    folds = _folds()
    boundary = pd.Timestamp(SESSIONS[320])
    result = run_walk_forward(
        FRAME, folds=folds, specs=_specs()[:1], sessions=SESSIONS,
        horizon=3, classification_label="direction_3d",
        regression_label="future_return_3d",
        evaluation_test_end=boundary,
        feature_cols=feature_columns(FRAME),
    )
    eval_folds = [f for f in result.folds if f.group == "evaluation"]
    hold_folds = [f for f in result.folds if f.group == "holdout"]
    assert eval_folds and hold_folds
    assert all(pd.Timestamp(f.test_end) < boundary for f in eval_folds)
    assert all(pd.Timestamp(f.test_end) >= boundary for f in hold_folds)
    payload = metrics_to_payload(result)
    assert payload["n_folds_evaluation"] == len(eval_folds)
    assert payload["n_folds_holdout"] == len(hold_folds)


# ------------------------------------------------------------------- checks
def _fold_record(purge_verified: bool = True) -> dict:
    return {
        "fold_id": 0,
        "train": ["2023-01-02", "2023-10-10"],
        "test": ["2023-10-11", "2024-01-10"],
        "n_train": 100, "n_test": 30, "n_purged": 3, "base_rate": 0.52,
        "group": "evaluation",
        "metrics": {"logistic": {"roc_auc": 0.63, "log_loss": 0.68}},
        "purge_verified": purge_verified,
    }


def _predictions_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": ["CBA.AX"] * 3,
        "date": pd.bdate_range("2023-10-11", periods=3),
        "model_name": ["logistic"] * 3,
        "model_kind": ["classifier"] * 3,
        "fold_id": [0] * 3,
        "prediction": [0.2, 0.8, 0.5],
    })


def test_training_quality_clean_passes() -> None:
    report = model_checks.run_training_quality_checks(
        [_fold_record()], _predictions_frame(), ["logistic"],
        evaluation_test_end="2024-07-01",
        feature_set_version="v0.1", expected_feature_set_version="v0.1",
    )
    assert report.passed, [r.name for r in report.errors if not r.passed]


def test_training_quality_purge_violation_is_error() -> None:
    folds = [dict(_fold_record(), purge_verified=False)]
    report = model_checks.run_training_quality_checks(
        folds, _predictions_frame(), ["logistic"],
        evaluation_test_end="2024-07-01",
        feature_set_version="v0.1", expected_feature_set_version="v0.1")
    rule = next(r for r in report.results if r.name == "purge_verified_every_fold")
    assert rule.severity == "error" and not rule.passed


def test_training_quality_missing_model_in_fold_is_error() -> None:
    folds = [dict(_fold_record(), metrics={})]
    report = model_checks.run_training_quality_checks(
        folds, _predictions_frame(), ["logistic"],
        evaluation_test_end="2024-07-01",
        feature_set_version="v0.1", expected_feature_set_version="v0.1")
    rule = next(r for r in report.results
                if r.name == "all_models_scored_every_fold")
    assert rule.severity == "error" and not rule.passed


def test_training_quality_version_mismatch_is_error() -> None:
    report = model_checks.run_training_quality_checks(
        [_fold_record()], _predictions_frame(), ["logistic"],
        evaluation_test_end="2024-07-01",
        feature_set_version="v0.2", expected_feature_set_version="v0.1")
    rule = next(r for r in report.results
                if r.name == "feature_set_version_matches_config")
    assert rule.severity == "error" and not rule.passed


def test_training_quality_boundary_violation_is_error() -> None:
    folds = [dict(_fold_record(), test_end="2024-08-15")]
    report = model_checks.run_training_quality_checks(
        folds, _predictions_frame(), ["logistic"],
        evaluation_test_end="2024-07-01",
        feature_set_version="v0.1", expected_feature_set_version="v0.1")
    rule = next(r for r in report.results
                if r.name == "evaluation_boundary_respected")
    assert rule.severity == "error" and not rule.passed


def test_training_quality_nonfinite_and_out_of_bounds() -> None:
    bad = _predictions_frame()
    bad.loc[0, "prediction"] = float("nan")
    report = model_checks.run_training_quality_checks(
        [_fold_record()], bad, ["logistic"], evaluation_test_end="2024-07-01",
        feature_set_version="v0.1", expected_feature_set_version="v0.1")
    assert not next(r for r in report.results
                    if r.name == "predictions_finite").passed

    bad2 = _predictions_frame()
    bad2.loc[0, "prediction"] = 1.5
    report2 = model_checks.run_training_quality_checks(
        [_fold_record()], bad2, ["logistic"], evaluation_test_end="2024-07-01",
        feature_set_version="v0.1", expected_feature_set_version="v0.1")
    assert not next(r for r in report2.results
                    if r.name == "classifier_scores_are_probabilities").passed


def test_training_quality_empty_predictions_is_error() -> None:
    report = model_checks.run_training_quality_checks(
        [_fold_record()], pd.DataFrame(), ["logistic"],
        evaluation_test_end="2024-07-01",
        feature_set_version="v0.1", expected_feature_set_version="v0.1")
    assert not next(r for r in report.results
                    if r.name == "non_empty_predictions").passed


# ------------------------------------------------------- e2e train orchestrator
class _S:
    parquet_compression = "zstd"
    parquet_compression_level = 3


TEST_TRAIN_CFG = {
    "train": {
        "model_version": "v0.1-test",
        "features_dataset": "features",
        "targets_dataset": "target_labels",
        "feature_set_version": "v0.1",
        "target": {
            "horizon": 3,
            "classification_label": "direction_3d",
            "regression_label": "future_return_3d",
        },
        "columns": {"exclude": []},
        "cv": {
            "train_window_sessions": 200,
            "test_window_sessions": 60,
            "step_sessions": 60,
            "min_train_sessions": 100,
            "embargo_sessions": 3,
            "evaluation_test_end": str(SESSIONS[380].date()),
        },
        "models": {
            "logistic": {
                "enabled": True, "kind": "classifier",
                "estimator": "logistic_regression",
                "preprocessing": "impute_median_scale",
                "params": {"C": 0.1, "max_iter": 500},
                "class_weight": "balanced",
            },
            "hist_gbt_reg": {
                "enabled": True, "kind": "regressor",
                "estimator": "hist_gradient_boosting_regressor",
                "preprocessing": "none",
                "params": {"max_iter": 30, "learning_rate": 0.1,
                           "max_leaf_nodes": 15, "min_samples_leaf": 50,
                           "early_stopping": False},
            },
        },
        "outputs": {
            "predictions_dataset": "predictions",
            "save_final_artifacts": True,
        },
    },
}


def test_train_models_end_to_end(tmp_path, monkeypatch) -> None:
    from asx_quant.observability.runs import RunRegistry
    from asx_quant.storage.paths import DataLake

    monkeypatch.setattr("asx_quant.models.build.load_yaml",
                        lambda name: TEST_TRAIN_CFG)

    lake = DataLake(root=tmp_path / "data")
    lake.ensure_all()
    registry = RunRegistry.start(lake, config_version="test")

    features = FRAME[["ticker", "date", "alpha_signal", "noise_a", "noise_b",
                      "feature_set_version"]]
    targets = FRAME[["ticker", "date", "future_return_3d", "direction_3d",
                     "label_available_date_3d"]]
    features.to_parquet(lake.gold_dataset_path("features"), index=False)
    targets.to_parquet(lake.gold_dataset_path("target_labels"), index=False)

    from asx_quant.models.build import train_models

    result = train_models(lake, _S(), registry)
    assert result.n_folds >= 5
    assert result.n_models == 2
    assert "logistic" in result.evaluation_metrics
    assert result.evaluation_metrics["logistic"]["roc_auc"] > 0.60

    predictions = pd.read_parquet(result.predictions_path)
    assert not predictions.empty
    assert predictions["model_version"].eq("v0.1-test").all()
    # latest-fold-wins: one prediction per (ticker, date, model)
    assert not predictions.duplicated(
        subset=["ticker", "date", "model_name"]).any()
    assert predictions["prediction"].notna().all()

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["n_folds"] == result.n_folds
    assert metrics["evaluation"]["logistic"]["roc_auc"] > 0.60

    report = json.loads(
        (tmp_path / "data/reports/model_quality.json").read_text(encoding="utf-8"))
    assert report["status"] == "pass"

    # final artifacts saved with metadata
    model_path = tmp_path / "data/models/v0.1-test/logistic.joblib"
    assert model_path.exists()
    meta = json.loads(
        (tmp_path / "data/models/v0.1-test/logistic.json").read_text(
            encoding="utf-8"))
    assert meta["feature_columns"] == ["alpha_signal", "noise_a", "noise_b"]
    assert "walk_forward_evaluation" in meta


def test_train_models_rejects_wrong_feature_set_version(tmp_path,
                                                        monkeypatch) -> None:
    from asx_quant.observability.runs import RunRegistry
    from asx_quant.storage.paths import DataLake

    cfg = json.loads(json.dumps(TEST_TRAIN_CFG))
    cfg["train"]["feature_set_version"] = "v9.9"
    monkeypatch.setattr("asx_quant.models.build.load_yaml", lambda name: cfg)

    lake = DataLake(root=tmp_path / "data")
    lake.ensure_all()
    registry = RunRegistry.start(lake, config_version="test")

    features = FRAME[["ticker", "date", "alpha_signal", "noise_a", "noise_b",
                      "feature_set_version"]]
    targets = FRAME[["ticker", "date", "future_return_3d", "direction_3d",
                     "label_available_date_3d"]]
    features.to_parquet(lake.gold_dataset_path("features"), index=False)
    targets.to_parquet(lake.gold_dataset_path("target_labels"), index=False)

    from asx_quant.models.build import train_models

    with pytest.raises(AsxQuantError, match="feature_set_version"):
        train_models(lake, _S(), registry)


def test_train_config_file_is_valid() -> None:
    """The real config/train.yaml parses and is internally consistent."""
    from asx_quant.config.loader import load_yaml

    cfg = load_yaml("train")["train"]
    assert cfg["feature_set_version"] == "v0.1"
    cv = cfg["cv"]
    assert cv["test_window_sessions"] > cv["step_sessions"] > 0
    assert cv["embargo_sessions"] >= 0
    assert pd.Timestamp(cv["evaluation_test_end"]) == pd.Timestamp("2024-07-01")
    enabled = {n: m for n, m in cfg["models"].items() if m.get("enabled")}
    assert enabled, "at least one model must be enabled"
    for spec in enabled.values():
        assert spec["kind"] in ("classifier", "regressor")
        if "hist" in spec["estimator"]:
            assert spec["params"].get("early_stopping") is False
