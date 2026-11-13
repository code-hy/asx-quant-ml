"""Training dataset assembly: gold features + target labels -> model frame.

Column policy (hard assertions in code, tests in tests/test_train.py):

* Meta/identifier columns (``ticker``, ``date``, ``feature_set_version``,
  ``ingestion_run_id``, ``retrieved_at``) never enter the feature matrix.
* Label columns (``future_*``, ``direction_*``, ``label_available_date_*``)
  can NEVER enter the feature matrix — a forward-looking column in X would
  leak directly into training.
* Features are exactly the remaining NUMERIC columns.
* The stamped ``feature_set_version`` in the features dataset must match the
  training config, otherwise training aborts.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from asx_quant.exceptions import StorageError
from asx_quant.observability.logging import get_logger
from asx_quant.storage.parquet import read_parquet

logger = get_logger("models.dataset")

META_COLUMNS = frozenset({
    "ticker", "date", "feature_set_version", "ingestion_run_id", "retrieved_at",
})
LABEL_PREFIXES = ("future_", "direction_", "label_available_date_")


def load_training_frame(
    lake,  # noqa: ANN001 - DataLake (cycle-safe)
    features_dataset: str,
    targets_dataset: str,
    feature_set_version: str,
) -> pd.DataFrame:
    """Inner-join gold features and target labels on (ticker, date).

    Raises:
        StorageError: If either dataset is missing or the stamped feature
            set version does not match the requested one.
    """
    features_path = lake.gold_dataset_path(features_dataset)
    targets_path = lake.gold_dataset_path(targets_dataset)
    for path in (features_path, targets_path):
        if not Path(path).exists():
            raise StorageError(
                f"Gold dataset '{path.name}' not found; run the 'features' "
                "stage first."
            )

    features = read_parquet(features_path)
    targets = read_parquet(targets_path)

    stamped = features["feature_set_version"].unique()
    if list(stamped) != [feature_set_version]:
        raise StorageError(
            f"gold/{features_dataset} was built with feature_set_version "
            f"{stamped} but training config expects '{feature_set_version}'. "
            "Re-run the 'features' stage or update config/train.yaml."
        )

    label_cols = [c for c in targets.columns
                  if c not in ("ticker", "date", "ingestion_run_id",
                               "retrieved_at")]
    frame = features.merge(
        targets[["ticker", "date", *label_cols]],
        on=["ticker", "date"], how="inner", validate="one_to_one",
    )
    logger.info(
        "Training frame: %d rows, %d tickers, %s -> %s",
        len(frame), frame["ticker"].nunique(),
        frame["date"].min().date(), frame["date"].max().date(),
    )
    return frame


def feature_columns(
    frame: pd.DataFrame,
    extra_exclude: list[str] | None = None,
) -> list[str]:
    """Feature matrix columns: numeric, minus meta/label/excluded columns.

    Raises:
        ValueError: If a label column would sneak into X (impossible by the
            prefix filter, asserted anyway) or no feature columns remain.
    """
    excluded = set(META_COLUMNS) | set(extra_exclude or [])
    cols: list[str] = []
    for col in frame.columns:
        if col in excluded or not pd.api.types.is_numeric_dtype(frame[col]):
            continue
        if col.startswith(LABEL_PREFIXES):
            continue
        cols.append(col)
    if not cols:
        raise ValueError("No numeric feature columns found in the frame.")
    return cols


def feature_matrix(
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Float32 feature matrix (memory-friendly for ~450k x 66)."""
    X = frame[columns].astype("float32")
    if not X.columns.is_unique:
        raise ValueError("Feature matrix has duplicate column names.")
    return X
