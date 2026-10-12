"""Tests for macro_agent/walkforward.py: purged splits, MeanBaseline and walk_forward."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from macro_agent.walkforward import MeanBaseline, purged_splits, walk_forward

START = pd.Timestamp("2026-01-05T14:00:00Z")
RESULT_COLUMNS = ["slot_id", "strategy_id", "pred", "fold", "n_train", "train_cutoff"]
IRREGULAR_HORIZONS = {
    "A": pd.Timedelta(hours=24),
    "B": pd.Timedelta(hours=72),
    "C": pd.Timedelta(hours=120),
}


def build_panel(entries, horizons, target=lambda i, strategy: 0.0) -> pd.DataFrame:
    """One row per (slot, strategy). Rows of a slot share entry_utc but may differ in exit."""
    rows = []
    for i, entry in enumerate(entries):
        for strategy, horizon in horizons.items():
            rows.append(
                {
                    "slot_id": f"s{i:03d}",
                    "strategy_id": strategy,
                    "entry_utc": entry,
                    "exit_utc": entry + horizon,
                    "x": float(i),
                    "y": target(i, strategy),
                }
            )
    return pd.DataFrame(rows)


def daily_entries(n: int) -> list[pd.Timestamp]:
    return [START + pd.Timedelta(days=i) for i in range(n)]


def irregular_entries(n: int, seed: int = 0) -> list[pd.Timestamp]:
    gaps_hours = np.random.default_rng(seed).integers(1, 96, size=n)
    return list(START + pd.to_timedelta(np.cumsum(gaps_hours), unit="h"))


class RateClassifier:
    """Tiny classifier: P(class 1) is each strategy's training win rate."""

    def fit(self, X: pd.DataFrame, y) -> RateClassifier:
        y = pd.Series(np.asarray(y, dtype=float))
        self.rates_ = y.groupby(X["strategy_id"].to_numpy()).mean()
        self.overall_ = float(y.mean())
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = X["strategy_id"].map(self.rates_).fillna(self.overall_).to_numpy(dtype=float)
        return np.column_stack([1.0 - p, p])


def test_training_uses_only_labels_realised_before_each_fold():
    panel = build_panel(irregular_entries(60), IRREGULAR_HORIZONS)
    for step in (1, 4):
        folds = list(purged_splits(panel, min_train_slots=8, step=step))
        assert folds
        for train_index, test_slots in folds:
            train = panel.loc[train_index]
            test_rows = panel[panel["slot_id"].isin(test_slots)]
            assert len(train) > 0
            assert train["exit_utc"].max() < test_rows["entry_utc"].min()
            assert not train["slot_id"].isin(test_slots).any()


def test_overlapping_horizons_are_purged_by_exit_time():
    # Daily slots, every trade held 5 days: a slot exits at its entry + 5 days.
    panel = build_panel(daily_entries(15), {"A": pd.Timedelta(days=5)})
    train_index, test_slots = next(purged_splits(panel, min_train_slots=10))
    assert test_slots == ["s010"]
    train_slots = set(panel.loc[train_index, "slot_id"])
    assert "s004" in train_slots  # entered 6 days before, exits 1 day before the test entry
    assert "s008" not in train_slots  # entered 2 days before, still open at the test entry
    assert "s005" not in train_slots  # exits exactly at the test entry, not strictly before
    assert train_slots == {f"s{i:03d}" for i in range(5)}


def test_purged_splits_skips_folds_with_empty_training_set():
    # With a 5-day horizon, the first training row appears only once slot 6 is the test slot.
    panel = build_panel(daily_entries(15), {"A": pd.Timedelta(days=5)})
    folds = list(purged_splits(panel, min_train_slots=0))
    assert folds[0][1] == ["s006"]
    assert all(len(train_index) > 0 for train_index, _ in folds)


@pytest.mark.parametrize("step", [1, 3])
def test_every_post_warmup_slot_is_tested_exactly_once(step):
    # A 1-day horizon means every fold has training data, so no fold is skipped.
    panel = build_panel(daily_entries(22), {"A": pd.Timedelta(days=1), "B": pd.Timedelta(days=1)})
    warmup = 5
    tested = Counter(
        slot
        for _, test_slots in purged_splits(panel, min_train_slots=warmup, step=step)
        for slot in test_slots
    )
    assert set(tested) == {f"s{i:03d}" for i in range(warmup, 22)}
    assert all(count == 1 for count in tested.values())


def test_mean_baseline_uses_strategy_means_and_overall_mean_for_unseen():
    X = pd.DataFrame({"strategy_id": ["A", "A", "A", "B"]})
    model = MeanBaseline().fit(X, [1.0, 3.0, np.nan, 10.0])  # the NaN target is ignored
    pred = model.predict(pd.DataFrame({"strategy_id": ["A", "B", "C"]}))
    overall = (1.0 + 3.0 + 10.0) / 3
    np.testing.assert_allclose(pred, [2.0, 10.0, overall])


def test_walk_forward_ranks_positive_strategy_above_negative_one():
    panel = build_panel(
        daily_entries(30),
        {"A": pd.Timedelta(days=5), "B": pd.Timedelta(days=5)},
        target=lambda i, strategy: 0.01 if strategy == "A" else -0.01,
    )
    res = walk_forward(panel, ["strategy_id"], "y", MeanBaseline, min_train_slots=10)
    assert list(res.columns) == RESULT_COLUMNS
    wide = res.pivot(index="slot_id", columns="strategy_id", values="pred")
    assert (wide["A"] > wide["B"]).all()
    np.testing.assert_allclose(wide["A"], 0.01)
    np.testing.assert_allclose(wide["B"], -0.01)

    # Sorted by entry_utc, then strategy_id.
    keys = list(zip(panel.loc[res.index, "entry_utc"], res["strategy_id"]))
    assert keys == sorted(keys)
    # Each fold's cutoff is the earliest entry among its test rows.
    for _, rows in res.groupby("fold"):
        assert rows["train_cutoff"].iloc[0] == panel.loc[rows.index, "entry_utc"].min()


def test_fold_predictions_ignore_targets_realised_after_its_cutoff():
    panel = build_panel(
        daily_entries(20),
        {"A": pd.Timedelta(days=5), "B": pd.Timedelta(days=5)},
        target=lambda i, strategy: float(i) if strategy == "A" else -float(i),
    )
    kwargs = dict(
        feature_cols=["strategy_id"], target="y", make_model=MeanBaseline, min_train_slots=5
    )
    base = walk_forward(panel, **kwargs)

    fold = 3
    cutoff = base.loc[base["fold"] == fold, "train_cutoff"].iloc[0]
    perturbed = panel.copy()
    future = (perturbed["exit_utc"] >= cutoff).to_numpy()
    rng = np.random.default_rng(0)
    perturbed.loc[future, "y"] = rng.normal(scale=100.0, size=int(future.sum()))
    out = walk_forward(perturbed, **kwargs)

    pd.testing.assert_frame_equal(base[base["fold"] == fold], out[out["fold"] == fold])
    # Sanity check: the perturbation does reach later folds, so the test is not vacuous.
    last = base["fold"].max()
    assert not np.allclose(base[base["fold"] == last]["pred"], out[out["fold"] == last]["pred"])


def test_predict_proba_uses_positive_class_column():
    panel = build_panel(
        daily_entries(20),
        {"A": pd.Timedelta(days=5), "B": pd.Timedelta(days=5)},
        target=lambda i, strategy: 1.0 if strategy == "A" else 0.0,
    )
    res = walk_forward(
        panel,
        ["strategy_id"],
        "y",
        RateClassifier,
        min_train_slots=10,
        predict_method="predict_proba",
    )
    assert not res.empty
    assert (res.loc[res["strategy_id"] == "A", "pred"] == 1.0).all()
    assert (res.loc[res["strategy_id"] == "B", "pred"] == 0.0).all()


def test_nan_targets_are_dropped_from_training():
    panel = build_panel(
        daily_entries(12),
        {"A": pd.Timedelta(days=1), "B": pd.Timedelta(days=1)},
        target=lambda i, strategy: np.nan if (i, strategy) in {(2, "A"), (4, "B")} else float(i),
    )
    nan_labels = set(panel.index[panel["y"].isna()])
    fitted: list[pd.Series] = []

    class Recording(MeanBaseline):
        def fit(self, X, y):
            fitted.append(pd.Series(y).copy())
            return super().fit(X, y)

    res = walk_forward(panel, ["strategy_id"], "y", Recording, min_train_slots=3)
    assert fitted
    for y in fitted:
        assert y.notna().all()
        assert not nan_labels & set(y.index)
    assert res.groupby("fold")["n_train"].first().tolist() == [len(y) for y in fitted]


def test_folds_with_too_few_training_rows_are_skipped_and_not_numbered():
    panel = build_panel(
        daily_entries(10),
        {"A": pd.Timedelta(days=1), "B": pd.Timedelta(days=1)},
        target=lambda i, strategy: np.nan if i == 0 else 1.0,
    )
    res = walk_forward(
        panel, ["strategy_id"], "y", MeanBaseline, min_train_slots=2, min_train_rows=2
    )
    # Slot 2 trains only on slot 0, whose targets are NaN, so its fold is skipped.
    assert "s002" not in set(res["slot_id"])
    first = res[res["fold"] == 0]
    assert set(first["slot_id"]) == {"s003"}
    assert (first["n_train"] == 2).all()
    # Folds for slots 3..9 run and are numbered 0..6 with no gaps.
    assert sorted(res["fold"].unique()) == list(range(7))


def test_empty_result_keeps_documented_columns():
    panel = build_panel(daily_entries(5), {"A": pd.Timedelta(days=1)})
    assert list(purged_splits(panel, min_train_slots=50)) == []
    res = walk_forward(panel, ["strategy_id"], "y", MeanBaseline, min_train_slots=50)
    assert res.empty
    assert list(res.columns) == RESULT_COLUMNS


def test_result_does_not_depend_on_row_order():
    panel = build_panel(
        irregular_entries(40),
        IRREGULAR_HORIZONS,
        target=lambda i, strategy: float(i % 7) - 3.0 if strategy != "C" else 0.5,
    )
    shuffled = panel.sample(frac=1.0, random_state=1)
    a = walk_forward(panel, ["strategy_id"], "y", MeanBaseline, min_train_slots=5)
    b = walk_forward(shuffled, ["strategy_id"], "y", MeanBaseline, min_train_slots=5)
    pd.testing.assert_frame_equal(a, b)


def test_duplicate_panel_index_is_rejected():
    panel = build_panel(daily_entries(10), {"A": pd.Timedelta(days=1)})
    panel.index = [0] * len(panel)
    with pytest.raises(ValueError, match="unique"):
        walk_forward(panel, ["strategy_id"], "y", MeanBaseline, min_train_slots=2)


@pytest.mark.parametrize(("min_train_slots", "step"), [(-1, 1), (2, 0)])
def test_invalid_split_arguments_raise(min_train_slots, step):
    panel = build_panel(daily_entries(5), {"A": pd.Timedelta(days=1)})
    with pytest.raises(ValueError):
        list(purged_splits(panel, min_train_slots=min_train_slots, step=step))
