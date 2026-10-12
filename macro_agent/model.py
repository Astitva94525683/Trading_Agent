"""Predictive models and metrics for the purged walk-forward evaluation.

Design: docs/design.md section 7. Regressors predict net_ret and classifiers predict win.
Each model is refit on the purged expanding folds from macro_agent.walkforward, so a
prediction only uses trades that had exited before its slot's entry.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from macro_agent.features import CATEGORICAL_FEATURES, feature_columns
from macro_agent.walkforward import MeanBaseline, walk_forward

REGRESSORS = ("baseline", "ridge", "hgb")
CLASSIFIERS = ("baseline_clf", "logit", "hgb_clf")

_HGB_PARAMS: dict[str, Any] = {
    "max_depth": 3,
    "learning_rate": 0.05,
    "max_iter": 150,
    "min_samples_leaf": 40,
    "l2_regularization": 1.0,
    "random_state": 0,
}
_PRED_COLUMNS = [
    "model",
    "target",
    "row",
    "slot_id",
    "strategy_id",
    "pred",
    "fold",
    "n_train",
    "train_cutoff",
]
_METRIC_COLUMNS = [
    "model",
    "target",
    "n_slots",
    "n_preds",
    "rank_ic",
    "rank_ic_t",
    "hit_rate",
    "mae",
    "auc",
    "brier",
]


def model_inputs(panel: pd.DataFrame) -> tuple[list[str], list[str]]:
    """The numeric model features and the categorical features that panel has."""
    categorical = [name for name in CATEGORICAL_FEATURES if name in panel.columns]
    return feature_columns(panel), categorical


class _DropEmptyColumns(BaseEstimator, TransformerMixin):
    """Keeps the columns that have a non-NaN value in the training data.

    HistGradientBoosting raises on an all-NaN column, so those are dropped. Other columns,
    NaN included, pass through unchanged.
    """

    def fit(self, X: pd.DataFrame, y: object = None) -> _DropEmptyColumns:
        """Record which columns have at least one non-NaN value."""
        self.keep_ = ~np.isnan(np.asarray(X, dtype=float)).all(axis=0)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Keep the recorded columns, in their original order."""
        return np.asarray(X, dtype=float)[:, self.keep_]


def _preprocessor(
    numeric: Sequence[str], categorical: Sequence[str], scale: bool
) -> ColumnTransformer:
    """Numeric columns, then one-hot categorical columns. Other columns are dropped.

    With scale, numeric columns are median-imputed and standardised for the linear models.
    Without it, gradient boosting gets them with NaN intact, minus any all-NaN columns.
    """
    numeric_step: BaseEstimator | str = _DropEmptyColumns()
    if scale:
        numeric_step = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scale", StandardScaler()),
            ]
        )
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        [
            ("numeric", numeric_step, list(numeric)),
            ("categorical", encoder, list(categorical)),
        ],
        remainder="drop",
    )


class BaseRateFallback:
    """Binary classifier that falls back to the training base rate when it sees one class.

    fit casts y to int and stores base_rate_, its mean. With a single class there is nothing
    to learn, so no model is fitted. Otherwise a clone of estimator is fitted. predict_proba
    returns an (n, 2) array of [1 - p, p], where p is the base rate or the fitted model's
    probability of class 1.
    """

    def __init__(self, estimator: BaseEstimator) -> None:
        self.estimator = estimator

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> BaseRateFallback:
        """Store the base rate of y and, when y has two classes, fit a clone of the estimator."""
        target = np.asarray(y).astype(int)
        self.base_rate_ = float(target.mean())
        self.model_ = None
        if np.unique(target).size > 1:
            self.model_ = clone(self.estimator).fit(X, target)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """(n, 2) array of [1 - p, p] for each row of X."""
        if self.model_ is None:
            p = np.full(len(X), self.base_rate_)
        else:
            p = self.model_.predict_proba(X)[:, 1]
        return np.column_stack([1.0 - p, p])


class MeanRateClassifier:
    """Naive classifier: each strategy's historical win rate, as a two-column probability."""

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> MeanRateClassifier:
        """Fit MeanBaseline on y as floats, matched to X by position."""
        self.baseline_ = MeanBaseline().fit(X, np.asarray(y, dtype=float))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """(n, 2) array of [1 - p, p], with p the strategy's win rate clipped to [0, 1]."""
        p = np.clip(self.baseline_.predict(X), 0.0, 1.0)
        return np.column_stack([1.0 - p, p])


def make_regressor(
    name: str, numeric: Sequence[str], categorical: Sequence[str]
) -> MeanBaseline | Pipeline:
    """A fresh unfitted regressor for net_ret. Raises ValueError for an unknown name."""
    if name == "baseline":
        return MeanBaseline()
    if name == "ridge":
        return Pipeline(
            [
                ("prep", _preprocessor(numeric, categorical, scale=True)),
                ("model", Ridge(alpha=10.0)),
            ]
        )
    if name == "hgb":
        return Pipeline(
            [
                ("prep", _preprocessor(numeric, categorical, scale=False)),
                ("model", HistGradientBoostingRegressor(**_HGB_PARAMS)),
            ]
        )
    raise ValueError(f"unknown regressor {name!r}; expected one of {REGRESSORS}")


def make_classifier(
    name: str, numeric: Sequence[str], categorical: Sequence[str]
) -> MeanRateClassifier | BaseRateFallback:
    """A fresh unfitted classifier for win. Raises ValueError for an unknown name."""
    if name == "baseline_clf":
        return MeanRateClassifier()
    if name == "logit":
        return BaseRateFallback(
            Pipeline(
                [
                    ("prep", _preprocessor(numeric, categorical, scale=True)),
                    ("model", LogisticRegression(C=0.1, max_iter=2000)),
                ]
            )
        )
    if name == "hgb_clf":
        return BaseRateFallback(
            Pipeline(
                [
                    ("prep", _preprocessor(numeric, categorical, scale=False)),
                    ("model", HistGradientBoostingClassifier(**_HGB_PARAMS)),
                ]
            )
        )
    raise ValueError(f"unknown classifier {name!r}; expected one of {CLASSIFIERS}")


def walk_forward_predictions(
    panel: pd.DataFrame,
    min_train_slots: int,
    regressors: Sequence[str] = REGRESSORS,
    classifiers: Sequence[str] = CLASSIFIERS,
    step: int = 1,
) -> pd.DataFrame:
    """Out-of-sample predictions from every named model, stacked in long form.

    Regressors predict net_ret and classifiers predict the probability of win. Each model is
    refit on the purged folds of walk_forward. Regressors come first, then classifiers, each
    in the order given. Returns the _PRED_COLUMNS, with row set to the panel index label.
    """
    unknown = [name for name in regressors if name not in REGRESSORS]
    unknown += [name for name in classifiers if name not in CLASSIFIERS]
    if unknown:
        raise ValueError(f"unknown models: {unknown}")
    numeric, categorical = model_inputs(panel)
    features = [*numeric, *categorical, "strategy_id"]
    frames: list[pd.DataFrame] = []
    for name in regressors:
        make = partial(make_regressor, name, numeric, categorical)
        result = walk_forward(
            panel, features, "net_ret", make, min_train_slots, step=step
        )
        frames.append(_tag(result, name, "net_ret"))
    for name in classifiers:
        make = partial(make_classifier, name, numeric, categorical)
        result = walk_forward(
            panel,
            features,
            "win",
            make,
            min_train_slots,
            step=step,
            predict_method="predict_proba",
        )
        frames.append(_tag(result, name, "win"))
    if not frames:
        return pd.DataFrame(columns=_PRED_COLUMNS)
    return pd.concat(frames, ignore_index=True)


def _tag(result: pd.DataFrame, model: str, target: str) -> pd.DataFrame:
    """Add the model name, target and panel row label to one walk_forward result."""
    tagged = result.assign(model=model, target=target, row=result.index.to_numpy())
    return tagged[_PRED_COLUMNS]


def predictive_metrics(preds: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Accuracy metrics for each model in preds, one row per model in order of appearance.

    Actual net_ret and win are read from panel.loc[row]. Every model gets rank_ic and rank_ic_t:
    the mean across slots of the rank correlation between pred and net_ret, and its t-statistic.
    A net_ret model also gets hit_rate and mae. A win model also gets auc, brier and hit_rate.
    Metrics that do not apply are NaN, and rows with a NaN pred or actual are left out.
    """
    rows: list[dict[str, object]] = []
    for model, group in preds.groupby("model", sort=False):
        target = str(group["target"].iloc[0])
        labels = group["row"].to_numpy()
        pred = group["pred"].to_numpy(dtype=float)
        net_ret = panel.loc[labels, "net_ret"].to_numpy(dtype=float)
        rank_ic, rank_ic_t = _rank_ic(group["slot_id"].to_numpy(), pred, net_ret)
        row: dict[str, object] = {
            "model": model,
            "target": target,
            "n_slots": group["slot_id"].nunique(),
            "n_preds": len(group),
            "rank_ic": rank_ic,
            "rank_ic_t": rank_ic_t,
            "hit_rate": np.nan,
            "mae": np.nan,
            "auc": np.nan,
            "brier": np.nan,
        }
        if target == "net_ret":
            row.update(_net_ret_metrics(pred, net_ret))
        elif target == "win":
            row.update(
                _win_metrics(pred, panel.loc[labels, "win"].to_numpy(dtype=float))
            )
        else:
            raise ValueError(f"unknown target {target!r}")
        rows.append(row)
    return pd.DataFrame(rows, columns=_METRIC_COLUMNS)


def _rank_ic(
    slot_ids: np.ndarray, pred: np.ndarray, net_ret: np.ndarray
) -> tuple[float, float]:
    """Mean per-slot rank correlation of pred with net_ret, and its t-statistic.

    A slot counts only with at least 3 rows and a non-constant pred and net_ret. The
    t-statistic is mean / std(ddof=1) * sqrt(count) over the counted slots.
    """
    frame = pd.DataFrame({"slot": slot_ids, "pred": pred, "net_ret": net_ret}).dropna()
    ics: list[float] = []
    for _, group in frame.groupby("slot", sort=False):
        if len(group) < 3:
            continue
        ranked_pred = group["pred"].rank()
        ranked_ret = group["net_ret"].rank()
        if ranked_pred.nunique() < 2 or ranked_ret.nunique() < 2:
            continue
        ics.append(float(np.corrcoef(ranked_pred, ranked_ret)[0, 1]))
    count = len(ics)
    if count == 0:
        return np.nan, np.nan
    mean = float(np.mean(ics))
    spread = float(np.std(ics, ddof=1)) if count > 1 else 0.0
    if spread == 0.0:
        return mean, np.nan
    return mean, mean / spread * np.sqrt(count)


def _net_ret_metrics(pred: np.ndarray, net_ret: np.ndarray) -> dict[str, float]:
    """Hit rate over rows with a non-zero prediction, and mean absolute error."""
    keep = np.isfinite(pred) & np.isfinite(net_ret)
    pred, net_ret = pred[keep], net_ret[keep]
    traded = pred != 0
    hits = np.sign(pred[traded]) == np.sign(net_ret[traded])
    return {"hit_rate": _mean(hits), "mae": _mean(np.abs(pred - net_ret))}


def _win_metrics(pred: np.ndarray, win: np.ndarray) -> dict[str, float]:
    """AUC, Brier score and hit rate at a 0.5 threshold, for predicted win probabilities."""
    keep = np.isfinite(pred) & np.isfinite(win)
    pred, win = pred[keep], win[keep]
    auc = roc_auc_score(win, pred) if np.unique(win).size == 2 else np.nan
    return {
        "hit_rate": _mean((pred > 0.5) == (win == 1)),
        "auc": float(auc),
        "brier": _mean((pred - win) ** 2),
    }


def _mean(values: np.ndarray) -> float:
    """Mean of values, or NaN when there are none."""
    return float(values.mean()) if values.size else float("nan")
