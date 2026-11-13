"""Model factory: config-declared estimators -> sklearn pipelines.

Every model is wrapped so that ALL preprocessing is fit inside
``pipeline.fit(X_train)`` — scalers/imputers never see test data. GBDT
models use native NaN support with no preprocessing; linear models get
median imputation + standardisation (fit on training rows only).

Class imbalance: classifiers use ``class_weight='balanced'`` (or
``scale_pos_weight`` for boosted third-party estimators), computed per fold
on the training rows only.

IMPORTANT: histogram/boosting estimators are always configured with early
stopping DISABLED — sklearn's early stopping uses a random validation
split, which would let future sessions leak into training through the
validation fold. The config declares ``early_stopping: false`` and the
factory enforces it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from asx_quant.exceptions import ConfigError
from asx_quant.observability.logging import get_logger

logger = get_logger("models.fit")


@dataclass
class ModelSpec:
    """One configured model."""

    name: str
    kind: str                     # classifier | regressor
    estimator: str
    params: dict
    class_weight: str | None = None
    scale_pos_weight: str | None = None   # 'auto' for third-party boosters
    preprocessing: str = "none"   # none | impute_median_scale

    @classmethod
    def from_config(cls, name: str, cfg: dict) -> ModelSpec:
        kind = cfg.get("kind")
        if kind not in ("classifier", "regressor"):
            raise ConfigError(
                f"Model '{name}': kind must be 'classifier' or 'regressor', "
                f"got '{kind}'."
            )
        return cls(
            name=name,
            kind=kind,
            estimator=str(cfg.get("estimator", "")),
            params=dict(cfg.get("params", {})),
            class_weight=cfg.get("class_weight"),
            scale_pos_weight=cfg.get("scale_pos_weight"),
            preprocessing=str(cfg.get("preprocessing", "none")),
        )


_ESTIMATORS = {
    "logistic_regression": LogisticRegression,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
    "hist_gradient_boosting_regressor": HistGradientBoostingRegressor,
    "ridge": Ridge,
    "dummy_classifier": DummyClassifier,
    "dummy_regressor": DummyRegressor,
    "lightgbm": None,    # resolved lazily (third-party)
    "xgboost": None,
}


def _third_party(estimator: str):
    if estimator == "lightgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor

        return {"classifier": LGBMClassifier, "regressor": LGBMRegressor}
    if estimator == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor

        return {"classifier": XGBClassifier, "regressor": XGBRegressor}
    raise ConfigError(f"Unknown estimator '{estimator}'.")


def build_model(spec: ModelSpec, n_positive: int = 0, n_negative: int = 0):
    """Build an unfitted sklearn pipeline for ``spec``.

    Args:
        spec: The configured model.
        n_positive / n_negative: Training class counts, used for
            ``scale_pos_weight: auto`` (third-party boosters only).
    """
    if spec.estimator in ("lightgbm", "xgboost"):
        cls = _third_party(spec.estimator)[spec.kind]
        params = dict(spec.params)
        if spec.scale_pos_weight == "auto" and spec.kind == "classifier":
            if n_positive <= 0:
                raise ValueError(
                    "scale_pos_weight='auto' requires a positive class in "
                    "the training rows."
                )
            params["scale_pos_weight"] = float(n_negative) / n_positive
        return cls(**params)

    if spec.estimator not in _ESTIMATORS or _ESTIMATORS[spec.estimator] is None:
        raise ConfigError(f"Unknown estimator '{spec.estimator}'.")

    estimator_cls = _ESTIMATORS[spec.estimator]
    params = dict(spec.params)

    if spec.kind == "classifier":
        if spec.class_weight == "balanced" and "class_weight" not in params:
            supported = estimator_cls is not DummyClassifier
            if supported and estimator_cls in (LogisticRegression,
                                               HistGradientBoostingClassifier):
                params["class_weight"] = "balanced"
    if spec.estimator == "hist_gradient_boosting":
        # Random early-stopping splits leak future sessions — hard-disable.
        params["early_stopping"] = False
    if spec.estimator == "hist_gradient_boosting_regressor":
        params["early_stopping"] = False
    estimator = estimator_cls(**params)

    if spec.preprocessing == "impute_median_scale":
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", estimator),
        ])
    if spec.preprocessing == "none":
        return estimator
    raise ConfigError(
        f"Model '{spec.name}': unknown preprocessing '{spec.preprocessing}'."
    )


def predict_scores(model, X: np.ndarray | pd.DataFrame, kind: str) -> np.ndarray:
    """Score rows with a fitted model.

    Classifiers produce P(up) (``predict_proba`` column 1); regressors
    produce the expected future return (``predict``).
    """
    if kind == "classifier":
        proba = model.predict_proba(X)
        return proba[:, 1]
    return model.predict(X)
