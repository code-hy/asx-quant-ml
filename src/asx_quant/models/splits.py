"""Purged, embargoed walk-forward splits.

THE leakage control of model evaluation (enforced by tests, not convention):

* **Walk-forward**: folds move forward in time; a model is always evaluated
  on sessions strictly AFTER every session it trained on.
* **Purge**: a training sample whose label window reaches into the test
  window is removed — its outcome partially depends on test-period sessions.
* **Embargo**: training samples within ``embargo_sessions`` immediately
  before the test window are additionally removed, guarding against serial
  correlation between overlapping feature windows at the boundary.

The purge/embargo keys are the ``label_available_date_{h}d`` columns written
by Phase 4 into ``gold/target_labels.parquet``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from asx_quant.observability.logging import get_logger

logger = get_logger("models.splits")


@dataclass(frozen=True)
class Fold:
    """One contiguous walk-forward split, in session positions and dates."""

    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp      # exclusive
    test_start: pd.Timestamp
    test_end: pd.Timestamp       # exclusive


def walk_forward_splits(
    sessions: pd.DatetimeIndex,
    train_window_sessions: int | None,
    test_window_sessions: int,
    step_sessions: int,
    min_train_sessions: int = 1,
    max_folds: int | None = None,
) -> list[Fold]:
    """Contiguous walk-forward folds over a sorted session index.

    Args:
        sessions: Sorted unique trading sessions.
        train_window_sessions: Rolling training window length; ``None`` or 0
            means expanding (train from the first session).
        test_window_sessions: Length of each evaluation window.
        step_sessions: Advance of the test window between folds.
        min_train_sessions: Folds with a shorter training window are skipped.
        max_folds: Cap the number of folds (``None`` = all).

    Returns:
        Folds ordered by time. The first fold's test window starts at
        position ``train_window_sessions`` (rolling) or the earliest position
        with enough history (expanding); the last fold's test window is
        truncated at the end of the session index.
    """
    if len(sessions) == 0:
        return []
    n = len(sessions)
    test_window = max(1, int(test_window_sessions))
    step = max(1, int(step_sessions))
    min_train = max(1, int(min_train_sessions))
    start_pos = int(train_window_sessions or 0)

    folds: list[Fold] = []
    fold_id = 0
    test_start_pos = start_pos
    while test_start_pos < n - 1:
        train_start_pos = 0 if start_pos == 0 else test_start_pos - start_pos
        n_train = test_start_pos - train_start_pos
        if n_train >= min_train:
            test_end_pos = min(test_start_pos + test_window, n)
            folds.append(Fold(
                fold_id=fold_id,
                train_start=sessions[train_start_pos],
                train_end=sessions[test_start_pos - 1],
                test_start=sessions[test_start_pos],
                test_end=sessions[test_end_pos - 1],
            ))
            fold_id += 1
            if max_folds is not None and fold_id >= max_folds:
                break
        test_start_pos += step
    return folds


def purged_train_mask(
    frame: pd.DataFrame,
    fold: Fold,
    sessions: pd.DatetimeIndex,
    horizon: int,
    embargo_sessions: int = 0,
) -> pd.Series:
    """Boolean mask of rows usable for TRAINING in ``fold``.

    A row is trainable when all of the following hold:

    1. Its session lies in ``[train_start, train_end]``;
    2. **Purge** — its label availability date (the session whose close
       completes the label) is strictly before ``test_start``;
    3. **Embargo** — its session is at least ``embargo_sessions`` sessions
       before ``test_start``.

    Rows with a missing label availability date are excluded (they cannot be
    proven leak-free).

    Returns:
        Boolean Series aligned to ``frame`` (position index).
    """
    date = frame["date"]
    in_train_window = (date >= fold.train_start) & (date <= fold.train_end)

    avail_col = f"label_available_date_{horizon}d"
    if avail_col not in frame.columns:
        raise KeyError(
            f"Column '{avail_col}' missing — the purge rule requires the "
            "label availability dates from gold/target_labels.parquet."
        )
    avail = pd.to_datetime(frame[avail_col])
    purged = avail >= fold.test_start  # label window reaches into test
    missing_avail = avail.isna()

    # Embargo: drop sessions within `embargo_sessions` of the test start.
    test_pos = sessions.searchsorted(fold.test_start)
    embargo_cut = (
        sessions[max(0, test_pos - int(embargo_sessions))]
        if embargo_sessions > 0 and test_pos > 0 else fold.train_start
    )
    embargoed = date >= embargo_cut if embargo_sessions > 0 else pd.Series(False, index=frame.index)

    mask = in_train_window & ~purged & ~missing_avail & ~embargoed
    return mask.fillna(False)


def test_mask(frame: pd.DataFrame, fold: Fold) -> pd.Series:
    """Boolean mask of rows inside the fold's test window."""
    date = frame["date"]
    return (date >= fold.test_start) & (date <= fold.test_end)


def assert_no_overlap(
    frame: pd.DataFrame,
    fold: Fold,
    train_mask: pd.Series,
    horizon: int,
) -> None:
    """Raise ``ValueError`` if any training row's label reaches into test.

    This is the machine-checkable form of the purge rule; it runs after
    every fold's mask is built, in addition to the dataset-level quality
    checks.
    """
    avail = pd.to_datetime(frame.loc[train_mask, f"label_available_date_{horizon}d"])
    if avail.isna().any():
        raise ValueError(
            f"Fold {fold.fold_id}: training rows with missing "
            f"label_available_date_{horizon}d cannot be proven leak-free."
        )
    if not avail.empty and avail.max() >= fold.test_start:
        raise ValueError(
            f"Fold {fold.fold_id}: purge violation — a training row's label "
            f"availability ({avail.max().date()}) reaches into the test "
            f"window (from {fold.test_start.date()})."
        )
