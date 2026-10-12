"""Tests for macro_agent.model on a small synthetic panel (design: docs/design.md section 7)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from macro_agent.model import (
    CLASSIFIERS,
    REGRESSORS,
    BaseRateFallback,
    MeanRateClassifier,
    make_classifier,
    make_regressor,
    predictive_metrics,
    walk_forward_predictions,
)

N_SLOTS = 40
# strategy_id, rule, horizon, sign
STRATEGIES = (
    ("s0", "breakout", 1, 1.0),
    ("s1", "breakout", 1, -1.0),
    ("s2", "fade", 5, 1.0),
    ("s3", "fade", 5, -1.0),
)
ASSETS = ("ES", "ZN", "CL")


def _panel(seed: int = 0) -> pd.DataFrame:
    """A (slot, strategy) panel where net_ret = 0.02 * surprise_z * sign + noise.

    news_tone_7d is all NaN, so the handling of empty feature columns is exercised.
    """
    rng = np.random.default_rng(seed)
    entries = pd.date_range("2024-01-01", periods=N_SLOTS, freq="D", tz="UTC")
    rows = []
    for k, entry in enumerate(entries):
        surprise_z = float(rng.normal())
        for j, (strategy_id, rule, horizon, sign) in enumerate(STRATEGIES):
            net_ret = 0.02 * surprise_z * sign + 0.01 * float(rng.normal())
            rows.append(
                {
                    "slot_id": f"slot{k:03d}",
                    "strategy_id": strategy_id,
                    "asset": ASSETS[(k + j) % len(ASSETS)],
                    "rule": rule,
                    "horizon": float(horizon),
                    "sign": sign,
                    "entry_utc": entry,
                    "exit_utc": entry + pd.Timedelta(days=1),
                    "net_ret": net_ret,
                    "win": net_ret > 0,
                    "surprise_z": surprise_z,
                    "news_tone_7d": np.nan,
                }
            )
    panel = pd.DataFrame(rows)
    return panel.sort_values(["entry_utc", "strategy_id", "slot_id"], ignore_index=True)


def _as_preds(
    panel: pd.DataFrame, model: str, target: str, pred: np.ndarray
) -> pd.DataFrame:
    """One prediction per panel row, in the layout walk_forward_predictions returns."""
    return pd.DataFrame(
        {
            "model": model,
            "target": target,
            "row": panel.index.to_numpy(),
            "slot_id": panel["slot_id"].to_numpy(),
            "strategy_id": panel["strategy_id"].to_numpy(),
            "pred": pred,
            "fold": 0,
            "n_train": 0,
            "train_cutoff": panel["entry_utc"].to_numpy(),
        }
    )


def test_walk_forward_predictions_covers_all_models_without_leakage():
    panel = _panel()
    preds = walk_forward_predictions(panel, min_train_slots=10, step=5)

    assert list(dict.fromkeys(preds["model"])) == [*REGRESSORS, *CLASSIFIERS]
    classifier_preds = preds.loc[preds["model"].isin(CLASSIFIERS), "pred"]
    assert classifier_preds.between(0.0, 1.0).all()
    entry_by_slot = panel.groupby("slot_id")["entry_utc"].first()
    assert (preds["train_cutoff"] <= preds["slot_id"].map(entry_by_slot)).all()


@pytest.mark.parametrize("value", [0, 1])
def test_base_rate_fallback_on_single_class_returns_base_rate(value: int):
    X = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]})
    y = pd.Series([value] * 4)
    model = BaseRateFallback(LogisticRegression()).fit(X, y)
    proba = model.predict_proba(X)
    np.testing.assert_allclose(proba, np.tile([1.0 - value, float(value)], (4, 1)))


def test_mean_rate_classifier_returns_per_strategy_rates():
    train = pd.DataFrame({"strategy_id": ["a", "a", "b", "b", "b"]})
    y = pd.Series([True, False, True, True, False])
    model = MeanRateClassifier().fit(train, y)
    new = pd.DataFrame({"strategy_id": ["a", "b", "unseen"]})
    proba = model.predict_proba(new)
    # a: 1/2, b: 2/3, unseen strategy: overall mean 3/5
    np.testing.assert_allclose(proba[:, 1], [0.5, 2 / 3, 0.6])
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_predictive_metrics_scores_perfect_predictions_as_perfect():
    panel = _panel()
    preds = pd.concat(
        [
            _as_preds(panel, "oracle", "net_ret", panel["net_ret"].to_numpy()),
            _as_preds(panel, "oracle_win", "win", panel["win"].to_numpy(dtype=float)),
        ],
        ignore_index=True,
    )
    metrics = predictive_metrics(preds, panel).set_index("model")
    assert list(metrics.index) == ["oracle", "oracle_win"]
    assert metrics.loc["oracle", "rank_ic"] == pytest.approx(1.0)
    assert metrics.loc["oracle", "hit_rate"] == pytest.approx(1.0)
    assert metrics.loc["oracle_win", "auc"] == pytest.approx(1.0)


def test_unknown_model_names_raise_value_error():
    numeric, categorical = ["horizon"], ["asset"]
    with pytest.raises(ValueError):
        make_regressor("nope", numeric, categorical)
    with pytest.raises(ValueError):
        make_classifier("nope", numeric, categorical)
    with pytest.raises(ValueError):
        walk_forward_predictions(
            _panel(), min_train_slots=10, regressors=("nope",), step=5
        )
