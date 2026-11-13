# Training Methodology (Phase 5)

**Model version:** `v0.1` · **Config:** `config/train.yaml` · **Code:** `src/asx_quant/models/`

This document describes how models are trained and evaluated, and which
leakage controls make the reported numbers trustworthy. Everything here is
enforced by tests (`tests/test_train.py`), not by convention.

---

## 1. Inputs

* **Features:** `gold/features.parquet` (feature set `v0.1`, 66 numeric
  columns — see `docs/feature_catalogue.md`).
* **Labels:** `gold/target_labels.parquet` — the ONLY forward-looking
  columns in the platform, joined on `(ticker, date)`.
* **Primary horizon:** 3 sessions (`direction_3d` classification,
  `future_return_3d` regression). The 10d horizon can be trained the same
  way by changing `target.horizon` in config.
* The stamped `feature_set_version` must match the training config or the
  run aborts.
* Feature-matrix policy (hard, in code): meta columns
  (`ticker`, `date`, `feature_set_version`, `ingestion_run_id`,
  `retrieved_at`) and label-prefixed columns (`future_*`, `direction_*`,
  `label_available_date_*`) can never enter X.

## 2. Walk-forward cross-validation

Folds are contiguous in session time (`models/splits.py`), from config
`train.cv`:

| Parameter | Value | Meaning |
|---|---|---|
| `train_window_sessions` | 1008 | rolling training window (~4 years); `null` = expanding |
| `test_window_sessions` | 252 | evaluation window per fold |
| `step_sessions` | 63 | fold advance (~1 quarter) |
| `min_train_sessions` | 504 | shorter folds skipped |
| `embargo_sessions` | 3 | extra boundary gap |

A model in fold *k* is trained on sessions `[t0, ts)` and evaluated on
`[ts, te)` — it NEVER trains on a session at or after its evaluation
window, and preprocessing (imputation, scaling) is fit inside the training
rows only (sklearn Pipeline).

## 3. Purge and embargo (the core leakage controls)

**Purge.** A training row whose label window reaches into the test window
is removed: keep `label_available_date_3d < test_start`. Without this, a
training sample from the last sessions before the boundary has an outcome
partially determined by test-period prices — an optimistic bias that grows
with the horizon.

**Embargo.** Training rows within `embargo_sessions` of the test start are
also removed, guarding serial correlation of overlapping feature windows
(e.g. a 252-session rolling feature spans the boundary). With
`embargo_sessions <= horizon` the purge subsumes the embargo (tested);
raising the embargo beyond the horizon drops additional boundary sessions.

Both rules are machine-verified:
`assert_no_overlap` runs after EVERY fold's mask is built (raises on
violation), and the saved model-quality report records
`purge_verified_every_fold` as an error-severity rule.

**Label availability semantics.** `label_available_date_{h}d` is the
session whose close completes the label (Phase 4). Rows with a missing
availability date are excluded from training — a row that cannot be proven
leak-free is treated as leaky.

## 4. The held-out window

`train.cv.evaluation_test_end = 2024-07-01` equals
`backtest.yaml periods.test_start`. Folds whose test window ends before
that date form the **evaluation** group — the only numbers used for model
selection. Later folds still run (they produce the OOS predictions the
Phase 6/7 backtest needs, always trained strictly on earlier data), but
their metrics are reported separately as **holdout diagnostics** and must
never drive selection.

## 5. Model zoo (v0.1)

| Model | Kind | Notes |
|---|---|---|
| `logistic` | classifier | median-impute + standardise (fit on train only), L2, `class_weight=balanced` |
| `hist_gbt` | classifier | HistGradientBoosting, native NaN support, `class_weight=balanced` |
| `ridge` | regressor | median-impute + standardise, L2 |
| `hist_gbt_reg` | regressor | HistGradientBoostingRegressor, native NaN |
| `lightgbm` / `xgboost` | classifier | available, disabled by default (`scale_pos_weight=auto`) |

Class imbalance: classifiers use class weighting computed on training rows
only. **Early stopping is hard-disabled** for all boosting estimators —
sklearn's early stopping uses a RANDOM validation split, which would leak
future sessions into training through that split.

## 6. Metrics

* Classification: ROC-AUC (primary), log-loss, Brier, accuracy,
  precision/recall at the 0.5 threshold. Fold base rate reported for
  context; extreme base rates (<5% or >95%) raise a warning rule.
* Regression: Spearman rank IC (primary for ranking usefulness), RMSE,
  MAE, R². Spearman is computed without scipy (Pearson on ranks).

## 7. Outputs

* **`gold/predictions.parquet`** — long format: one row per
  (ticker, date, model). Overlapping test windows collapse
  latest-fold-wins, so each session carries the prediction of the model
  trained on the most data available at that time. Columns include
  `fold_id` and `trained_through` for traceability.
* **`data/reports/model_metrics.json`** — per-fold records, aggregated
  evaluation + holdout metrics, feature columns, model registry.
* **`data/reports/model_quality.json`** — error/warn rules (purge
  verification, prediction finiteness, probability bounds, boundary
  discipline). Saved BEFORE `assert_passed()`.
* **`data/models/<model_version>/<model>.joblib`** — final artifacts,
  refit on ALL labelled history for Phase 6+ inference, with JSON metadata
  (feature columns, trained-through date, walk-forward evaluation
  snapshot). These artifacts are NEVER used for evaluation — every
  reported metric comes from the walk-forward folds.

## 8. Known limitations (v0.1)

* **No hyperparameter search yet** — parameters are hand-set in config;
  any tuning must happen inside the evaluation folds only (a future
  Phase 5+ extension should search per fold, never globally).
* **Single target horizon** — the 10d horizon is not trained by default.
* **No feature selection** — all 66 features enter every model; NaN-heavy
  columns (e.g. `macro_aud_usd` before 2023) rely on native-NaN models or
  median imputation.
* **No transaction-cost-aware threshold** — classifiers score P(up);
  converting to trades (top-N ranking, thresholds) is Phase 6.
* **Survivorship-biased universe** — inherited from the Phase 2 universe
  source; documented in `docs/data_sources.md`.
* **Fold base rates drift** — reported per fold; the class weighting
  keeps training calibrated but probability calibration across regimes is
  not yet checked explicitly.
