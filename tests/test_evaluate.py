"""Out-of-sample evaluation on the synthetic pipeline from test_integration."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from macro_agent.config import ASSETS, VIX_SYMBOL, Settings
from macro_agent.evaluate import (
    BUY_HOLD,
    LONG_SPY,
    MODEL_LABELS,
    SUMMARY_COLUMNS,
    evaluate,
    top_strategies,
)
from macro_agent.features import (
    build_panel,
    build_slots,
    market_features,
    news_features,
    prepare_events,
    surprise_index,
    time_features,
)
from macro_agent.strategies import compute_labels, pivot_prices, strategy_grid
from macro_agent.timeutils import SessionCalendar
from test_integration import _events, _news, _prices

REGRESSORS = ("baseline", "ridge")
CLASSIFIERS = ("baseline_clf",)


def _synthetic_panel():
    """Panel, opens and closes, wired the same way as test_integration."""
    sessions = list(pd.bdate_range("2026-03-02", "2026-07-31").date)
    calendar = SessionCalendar.from_dates(sessions)
    opens, closes = pivot_prices(_prices(sessions))
    events = prepare_events(_events(sessions))
    slots = build_slots(events, calendar)
    fomc = [ts.tz_convert("America/New_York").date()
            for ts in events.loc[events["category"] == "fed", "release_utc"]]
    slots = surprise_index(time_features(slots, fomc))
    slots = news_features(slots, _news(sessions))
    labels = compute_labels(slots, opens, closes, calendar, strategy_grid(), cost_bps=2.0)
    market = market_features(slots, opens, closes, tuple(ASSETS), VIX_SYMBOL)
    return build_panel(slots, labels, market), opens, closes


@pytest.fixture(scope="module")
def run():
    """One evaluation of the synthetic panel, shared by the tests below."""
    panel, opens, closes = _synthetic_panel()
    settings = Settings(min_train_slots=10)
    ev = evaluate(panel, opens, closes, settings, regressors=REGRESSORS,
                  classifiers=CLASSIFIERS, step=5)
    return panel, opens, closes, settings, ev


def test_summary_has_one_row_per_label_in_order(run):
    *_, ev = run
    labels = [MODEL_LABELS["baseline"], MODEL_LABELS["ridge"], LONG_SPY, BUY_HOLD]
    assert ev.summary["strategy"].tolist() == labels
    assert list(ev.summary.columns) == list(SUMMARY_COLUMNS)
    assert math.isfinite(ev.summary.set_index("strategy").loc[labels[0], "sharpe"])
    assert (ev.summary["max_drawdown"] <= 0).all()


@pytest.mark.parametrize("model", REGRESSORS)
def test_model_trades_are_one_per_slot_with_positive_pred(run, model):
    *_, ev = run
    trades = ev.trades[MODEL_LABELS[model]]
    assert len(trades) > 0
    assert trades["slot_id"].is_unique
    assert (trades["pred"] > 0).all()


def test_long_spy_trades_are_spy_one_session(run):
    *_, ev = run
    trades = ev.trades[LONG_SPY]
    assert len(trades) > 0
    assert (trades["strategy_id"] == "SPY|long|1").all()


def test_buy_hold_matches_spy_close_to_close(run):
    _, _, closes, _, ev = run
    sessions = closes.index
    in_window = sessions[(sessions >= ev.oos_start) & (sessions <= ev.oos_end)]
    before = sessions[sessions < ev.oos_start][-1]
    expected = closes.loc[in_window[-1], "SPY"] / closes.loc[before, "SPY"] - 1
    total = ev.summary.set_index("strategy").loc[BUY_HOLD, "total_return"]
    assert abs(total - expected) < 1e-9
    assert ev.trades[BUY_HOLD].empty


def test_strategy_pnl_is_the_direct_out_of_sample_mean(run):
    panel, _, _, _, ev = run
    oos = panel[panel["slot_id"].isin(ev.oos_slots)]
    expected = oos.groupby("strategy_id")["net_ret"].mean()
    got = ev.strategy_pnl.set_index("strategy_id")["value"]
    pd.testing.assert_series_equal(got.sort_index(), expected.sort_index(), check_names=False)
    assert ev.strategy_pnl["value"].is_monotonic_decreasing


def test_top_strategies_has_the_four_columns(run):
    *_, ev = run
    top = top_strategies(ev.strategy_pnl, n=3)
    assert list(top.columns) == ["strategy_id", "mean_ret", "win_rate", "n"]
    assert len(top) == 3


def test_too_few_slots_raises(run):
    panel, opens, closes, _, _ = run
    with pytest.raises(ValueError, match="not enough slots"):
        evaluate(panel, opens, closes, Settings(min_train_slots=10_000),
                 regressors=REGRESSORS, classifiers=CLASSIFIERS, step=5)
