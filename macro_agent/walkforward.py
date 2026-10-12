"""Purged expanding walk-forward validation and the naive mean benchmark.

Design: docs/design.md section 7. A fold's training rows must have exited before
its first test entry, so labels that were not yet realised cannot leak into the model.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np
import pandas as pd

_RESULT_COLUMNS = ["slot_id", "strategy_id", "pred", "fold", "n_train", "train_cutoff"]


class MeanBaseline:
    """Naive benchmark: each strategy's historical mean target.

    Strategies unseen in training get the overall mean. NaN targets are ignored.
    """

    def fit(self, X: pd.DataFrame, y) -> MeanBaseline:
        """Store each strategy's mean of y and the overall mean. y is matched to X by position."""
        values = np.asarray(y, dtype=float)
        known = ~np.isnan(values)
        if not known.any():
            raise ValueError("MeanBaseline needs at least one non-NaN target")
        frame = pd.DataFrame(
            {"strategy_id": X["strategy_id"].to_numpy()[known], "y": values[known]}
        )
        self.strategy_means_: pd.Series = frame.groupby("strategy_id")["y"].mean()
        self.overall_mean_: float = float(values[known].mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Each row's strategy mean, or the overall mean for a strategy not seen in training."""
        means = X["strategy_id"].map(self.strategy_means_)
        return means.fillna(self.overall_mean_).to_numpy(dtype=float)


def purged_splits(
    panel: pd.DataFrame, min_train_slots: int, step: int = 1
) -> Iterator[tuple[pd.Index, list[str]]]:
    """Yield (train_index, test_slot_ids) for each purged expanding fold.

    Slots are ordered by (entry_utc, slot_id). Fold k tests the slots ordered[k:k+step]
    and trains on panel rows whose exit_utc is strictly before the earliest test entry.
    Folds with no training rows are skipped.
    """
    for train_index, test_slots, _ in _folds(panel, min_train_slots, step):
        yield train_index, test_slots


def walk_forward(
    panel: pd.DataFrame,
    feature_cols: Sequence[str],
    target: str,
    make_model: Callable[[], Any],
    min_train_slots: int,
    step: int = 1,
    min_train_rows: int = 1,
    predict_method: str = "predict",
) -> pd.DataFrame:
    """Out-of-sample predictions from purged walk-forward folds.

    Each fold drops training rows with a NaN target, and is skipped if fewer than
    max(min_train_rows, 1) rows remain. A fresh make_model() is fitted on feature_cols
    and predicts every row of the fold's test slots. predict_method="predict_proba"
    uses column 1. For MeanBaseline, include "strategy_id" in feature_cols.

    Returns one row per predicted panel row, indexed by the panel index, with columns
    slot_id, strategy_id, pred, fold (0-based over folds actually run), n_train, and
    train_cutoff (the fold's t0). Sorted by entry_utc, then strategy_id and slot_id.
    """
    features = list(feature_cols)
    frames: list[pd.DataFrame] = []
    fold = 0
    for train_index, test_slots, t0 in _folds(panel, min_train_slots, step):
        train_index = train_index[panel.loc[train_index, target].notna().to_numpy()]
        if len(train_index) < max(min_train_rows, 1):
            continue
        model = make_model()
        model.fit(panel.loc[train_index, features], panel.loc[train_index, target])
        test_rows = panel.index[panel["slot_id"].isin(test_slots).to_numpy()]
        pred = _predict(model, panel.loc[test_rows, features], predict_method)
        frames.append(
            panel.loc[test_rows, ["slot_id", "strategy_id", "entry_utc"]].assign(
                pred=pred, fold=fold, n_train=len(train_index), train_cutoff=t0
            )
        )
        fold += 1
    if not frames:
        return _empty_result(panel)
    result = pd.concat(frames).sort_values(["entry_utc", "strategy_id", "slot_id"])
    return result.drop(columns="entry_utc")[_RESULT_COLUMNS]


def _folds(
    panel: pd.DataFrame, min_train_slots: int, step: int
) -> Iterator[tuple[pd.Index, list[str], pd.Timestamp]]:
    """Yield (train_index, test_slot_ids, t0) for each fold with a non-empty training set."""
    if min_train_slots < 0 or step < 1:
        raise ValueError("need min_train_slots >= 0 and step >= 1")
    if not panel.index.is_unique:
        raise ValueError("panel index must be unique")
    slots = (
        panel.groupby("slot_id")["entry_utc"]
        .min()
        .reset_index()
        .sort_values(["entry_utc", "slot_id"])
    )
    ordered = slots["slot_id"].tolist()
    entries = slots.set_index("slot_id")["entry_utc"]
    for k in range(min_train_slots, len(ordered), step):
        test_slots = ordered[k : k + step]
        t0 = entries.loc[test_slots].min()
        train_index = panel.index[(panel["exit_utc"] < t0).to_numpy()]
        if len(train_index) > 0:
            yield train_index, test_slots, t0


def _predict(model: Any, X: pd.DataFrame, predict_method: str) -> np.ndarray:
    """Call the named model method. For predict_proba, keep the positive-class column."""
    raw = np.asarray(getattr(model, predict_method)(X), dtype=float)
    if predict_method == "predict_proba":
        raw = raw[:, 1]
    return raw.reshape(len(X))


def _empty_result(panel: pd.DataFrame) -> pd.DataFrame:
    """Zero-row result with the documented columns and dtypes."""
    index = panel.index[:0]
    dtypes = {
        "slot_id": panel["slot_id"].dtype,
        "strategy_id": panel["strategy_id"].dtype,
        "pred": np.float64,
        "fold": np.int64,
        "n_train": np.int64,
        "train_cutoff": panel["entry_utc"].dtype,
    }
    return pd.DataFrame(
        {col: pd.Series(dtype=dtypes[col], index=index) for col in _RESULT_COLUMNS},
        index=index,
    )
