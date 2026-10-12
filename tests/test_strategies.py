"""Tests for macro_agent.strategies (docs/design.md section 5)."""
from __future__ import annotations

import math
from datetime import date

import numpy as np
import pandas as pd
import pytest

from macro_agent.config import ASSETS, HORIZONS, RULES
from macro_agent.strategies import (
    LABEL_COLUMNS,
    Strategy,
    compute_labels,
    pivot_prices,
    position_sign,
    strategy_grid,
)
from macro_agent.timeutils import SessionCalendar

# Ten weekdays, 5 to 16 January 2026, no holidays. Session i is DAYS[i].
DAYS = [date(2026, 1, d) for d in (5, 6, 7, 8, 9, 12, 13, 14, 15, 16)]
CAL = SessionCalendar.from_dates(DAYS)

SPY_OPEN = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
SPY_CLOSE = [101, 103, 102, 106, 105, 104, 110, 108, 112, 111]
TLT_OPEN = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
TLT_CLOSE = [41, 40, 44, 42, 46, 45, 47, 50, 49, 50]
GLD_OPEN = [200, 198, 196, 194, 192, 190, 188, 186, 184, 182]
GLD_CLOSE = [198, 196, 194, 192, 190, 188, 186, 184, 182, 180]
UUP_OPEN = [25.0, 25.1, 25.2, 25.3, 25.4, 25.5, 25.6, 25.7, 25.8, 25.9]
UUP_CLOSE = [25.05, 25.15, 25.25, 25.35, 25.45, 25.55, 25.65, 25.75, 25.85, 25.95]
BASE_SERIES = {
    "SPY": (SPY_OPEN, SPY_CLOSE),
    "TLT": (TLT_OPEN, TLT_CLOSE),
    "GLD": (GLD_OPEN, GLD_CLOSE),
    "UUP": (UUP_OPEN, UUP_CLOSE),
}
SLOT_COLUMNS = ["slot_id", "entry_session", "entry_kind", "entry_utc", "surprise_z"]


def _price_frames(series):
    """Wide (opens, closes) from {symbol: (opens, closes)} over DAYS, built via pivot_prices."""
    rows = [
        {"symbol": symbol, "date": day.isoformat(), "open": o, "close": c}
        for symbol, (opens, closes) in series.items()
        for day, o, c in zip(DAYS, opens, closes)
    ]
    return pivot_prices(pd.DataFrame(rows))


OPENS, CLOSES = _price_frames(BASE_SERIES)


def _slot(slot_id, day, kind, surprise=math.nan):
    """A slot row whose entry_utc is the calendar's open or close for its kind."""
    entry_utc = CAL.open_at(day) if kind == "open" else CAL.close_at(day)
    return {
        "slot_id": slot_id,
        "entry_session": day,
        "entry_kind": kind,
        "entry_utc": entry_utc,
        "surprise_z": surprise,
    }


def _slots(*rows):
    return pd.DataFrame(list(rows), columns=SLOT_COLUMNS)


def _labels(slots, strategies, cost_bps=0.0, opens=None, closes=None):
    return compute_labels(
        slots,
        OPENS if opens is None else opens,
        CLOSES if closes is None else closes,
        CAL,
        strategies,
        cost_bps,
    )


# --- Strategy and strategy_grid ---


def test_default_grid_has_48_unique_strategies():
    grid = strategy_grid()
    assert len(grid) == 48
    assert len({s.id for s in grid}) == 48
    assert {s.asset for s in grid} == set(ASSETS)
    assert {s.rule for s in grid} == set(RULES)
    assert {s.horizon for s in grid} == set(HORIZONS)


def test_grid_order_is_asset_then_rule_then_horizon():
    ids = [s.id for s in strategy_grid(("SPY", "TLT"), ("long", "short"), (1, 3))]
    assert ids == [
        "SPY|long|1",
        "SPY|long|3",
        "SPY|short|1",
        "SPY|short|3",
        "TLT|long|1",
        "TLT|long|3",
        "TLT|short|1",
        "TLT|short|3",
    ]
    grid = strategy_grid()
    assert grid[0].id == "SPY|long|1"
    assert grid[-1].id == "UUP|short_on_beat|5"
    per_asset = len(RULES) * len(HORIZONS)
    assert [s.asset for s in grid] == [a for a in ASSETS for _ in range(per_asset)]


def test_strategy_id_joins_asset_rule_and_horizon():
    assert Strategy("GLD", "long_on_beat", 3).id == "GLD|long_on_beat|3"


def test_strategy_rejects_unknown_rule():
    with pytest.raises(ValueError):
        Strategy("SPY", "buy", 1)


@pytest.mark.parametrize("horizon", [0, -1, 1.5, 2.0, "3"])
def test_strategy_rejects_horizon_that_is_not_a_positive_integer(horizon):
    with pytest.raises(ValueError):
        Strategy("SPY", "long", horizon)


def test_numpy_integer_horizon_is_accepted():
    grid = strategy_grid(("SPY",), ("long",), np.arange(1, 4))
    assert [s.id for s in grid] == ["SPY|long|1", "SPY|long|2", "SPY|long|3"]


# --- position_sign ---


@pytest.mark.parametrize("surprise", [None, math.nan, 0.0, 1.5, -2.0])
def test_long_and_short_ignore_the_surprise(surprise):
    assert position_sign("long", surprise) == 1
    assert position_sign("short", surprise) == -1


@pytest.mark.parametrize(
    ("surprise", "expected"),
    [(1.5, 1), (-0.7, -1), (0.0, 0), (None, 0), (math.nan, 0)],
)
def test_long_on_beat_follows_the_surprise_sign(surprise, expected):
    assert position_sign("long_on_beat", surprise) == expected


@pytest.mark.parametrize(
    ("surprise", "expected"),
    [(1.5, -1), (-0.7, 1), (0.0, 0), (None, 0), (math.nan, 0)],
)
def test_short_on_beat_is_the_negated_surprise_sign(surprise, expected):
    assert position_sign("short_on_beat", surprise) == expected


def test_unknown_rule_raises_with_or_without_a_surprise():
    with pytest.raises(ValueError):
        position_sign("buy", 1.0)
    with pytest.raises(ValueError):
        position_sign("buy", None)


# --- pivot_prices ---


@pytest.mark.parametrize(
    "as_input",
    [lambda d: d.isoformat(), lambda d: d, lambda d: pd.Timestamp(d)],
    ids=["string", "date", "timestamp"],
)
def test_pivot_prices_sorts_dates_and_pivots_symbols(as_input):
    prices = pd.DataFrame(
        {
            "symbol": ["SPY", "SPY", "TLT"],
            "date": [as_input(DAYS[1]), as_input(DAYS[0]), as_input(DAYS[0])],
            "open": [2.0, 1.0, 3.0],
            "close": [2.5, 1.5, 3.5],
            "volume": [10, 20, 30],  # extra columns are ignored
        }
    )
    opens, closes = pivot_prices(prices)
    assert opens.index.tolist() == DAYS[:2]
    assert all(type(d) is date for d in opens.index)
    assert opens.columns.tolist() == ["SPY", "TLT"]
    assert closes.index.tolist() == DAYS[:2]
    assert opens.at[DAYS[0], "SPY"] == 1.0
    assert closes.at[DAYS[1], "SPY"] == 2.5
    assert opens.at[DAYS[0], "TLT"] == 3.0
    assert math.isnan(opens.at[DAYS[1], "TLT"])


def test_pivot_prices_rejects_missing_columns():
    prices = pd.DataFrame({"symbol": ["SPY"], "date": ["2026-01-05"], "open": [1.0]})
    with pytest.raises(ValueError, match="missing columns"):
        pivot_prices(prices)


def test_pivot_prices_rejects_a_repeated_symbol_and_date():
    prices = pd.DataFrame(
        {
            "symbol": ["SPY", "SPY"],
            "date": ["2026-01-05", "2026-01-05"],
            "open": [1.0, 1.1],
            "close": [1.2, 1.3],
        }
    )
    with pytest.raises(ValueError):
        pivot_prices(prices)


def test_pivot_prices_of_no_rows_is_empty():
    opens, closes = pivot_prices(pd.DataFrame(columns=["symbol", "date", "open", "close"]))
    assert opens.empty and closes.empty


# --- compute_labels ---


def test_open_entry_with_horizon_1_trades_open_to_same_day_close():
    out = _labels(_slots(_slot("a", DAYS[0], "open")), [Strategy("SPY", "long", 1)])
    assert len(out) == 1
    row = out.iloc[0]
    assert row["entry_price"] == 100  # SPY open on DAYS[0]
    assert row["exit_price"] == 101  # SPY close on the same day
    assert row["entry_session"] == DAYS[0]
    assert row["exit_session"] == DAYS[0]
    assert row["exit_utc"] == CAL.close_at(DAYS[0])
    assert row["gross_ret"] == pytest.approx(0.01)
    assert row["net_ret"] == pytest.approx(0.01)
    assert row["win"]


def test_open_entry_with_horizon_3_exits_at_the_close_two_sessions_later():
    out = _labels(_slots(_slot("a", DAYS[0], "open")), [Strategy("SPY", "long", 3)])
    row = out.iloc[0]
    assert row["exit_session"] == DAYS[2]
    assert row["exit_price"] == 102  # SPY close on DAYS[2]
    assert row["exit_utc"] == CAL.close_at(DAYS[2])
    assert row["gross_ret"] == pytest.approx(0.02)


def test_close_entry_with_horizon_1_trades_close_to_next_close():
    out = _labels(_slots(_slot("b", DAYS[1], "close")), [Strategy("SPY", "long", 1)])
    row = out.iloc[0]
    assert row["entry_kind"] == "close"
    assert row["entry_utc"] == CAL.close_at(DAYS[1])
    assert row["entry_price"] == 103  # SPY close on DAYS[1]
    assert row["exit_session"] == DAYS[2]
    assert row["exit_price"] == 102  # SPY close on DAYS[2]
    assert row["gross_ret"] == pytest.approx(102 / 103 - 1)
    assert not row["win"]


def test_short_sign_negates_the_return_and_wins_when_price_falls():
    out = _labels(_slots(_slot("c", DAYS[5], "open")), [Strategy("SPY", "short", 1)])
    row = out.iloc[0]
    assert row["sign"] == -1
    assert row["entry_price"] == 105
    assert row["exit_price"] == 104
    assert row["gross_ret"] == pytest.approx(-(104 / 105 - 1))
    assert row["win"]


def test_beat_rules_follow_the_surprise_sign_and_skip_zero_or_missing_surprise():
    slots = _slots(
        _slot("pos", DAYS[0], "open", 1.0),
        _slot("neg", DAYS[0], "open", -0.5),
        _slot("zero", DAYS[0], "open", 0.0),
        _slot("nan", DAYS[0], "open", math.nan),
    )
    strategies = [
        Strategy("SPY", "long_on_beat", 1),
        Strategy("SPY", "short_on_beat", 1),
        Strategy("SPY", "long", 1),
    ]
    out = _labels(slots, strategies)
    got = set(zip(out["slot_id"].tolist(), out["rule"].tolist(), out["sign"].tolist()))
    assert got == {
        ("pos", "long_on_beat", 1),
        ("pos", "short_on_beat", -1),
        ("pos", "long", 1),
        ("neg", "long_on_beat", -1),
        ("neg", "short_on_beat", 1),
        ("neg", "long", 1),
        ("zero", "long", 1),  # plain long still trades without a surprise
        ("nan", "long", 1),
    }
    assert len(out) == 8


def test_cost_is_charged_on_both_sides_and_decides_the_win():
    slots = _slots(_slot("a", DAYS[0], "open"))
    out = _labels(slots, [Strategy("SPY", "long", 1)], cost_bps=10.0)
    row = out.iloc[0]
    assert row["gross_ret"] == pytest.approx(0.01)
    assert row["net_ret"] == pytest.approx(0.01 - 2 * 10.0 / 1e4)  # 0.008
    assert row["win"]

    out = _labels(slots, [Strategy("SPY", "long", 1)], cost_bps=60.0)
    row = out.iloc[0]
    assert row["net_ret"] == pytest.approx(0.01 - 2 * 60.0 / 1e4)  # -0.002
    assert not row["win"]


def test_trade_whose_exit_is_beyond_the_calendar_is_skipped():
    # Open on DAYS[8]: horizon 1 exits on DAYS[8]; horizons 3 and 5 would need days past DAYS[9].
    strategies = [Strategy("SPY", "long", h) for h in (1, 3, 5)]
    out = _labels(_slots(_slot("a", DAYS[8], "open")), strategies)
    assert out["horizon"].tolist() == [1]

    # A close entry on the last session has no next session to exit on.
    out = _labels(_slots(_slot("b", DAYS[9], "close")), [Strategy("SPY", "long", 1)])
    assert out.empty


def test_trade_with_nan_exit_price_is_skipped():
    series = {**BASE_SERIES, "SPY": (SPY_OPEN, SPY_CLOSE[:2] + [math.nan] + SPY_CLOSE[3:])}
    opens, closes = _price_frames(series)
    strategies = [Strategy("SPY", "long", 1), Strategy("SPY", "long", 3)]
    out = _labels(_slots(_slot("a", DAYS[0], "open")), strategies, opens=opens, closes=closes)
    assert out["horizon"].tolist() == [1]  # horizon 3 exits on DAYS[2], whose close is NaN


def test_trade_with_non_positive_entry_price_is_skipped():
    series = {**BASE_SERIES, "TLT": ([0] + TLT_OPEN[1:], TLT_CLOSE)}
    opens, closes = _price_frames(series)
    strategies = [Strategy("TLT", "long", 1), Strategy("SPY", "long", 1)]
    out = _labels(_slots(_slot("a", DAYS[0], "open")), strategies, opens=opens, closes=closes)
    assert out["asset"].tolist() == ["SPY"]


def test_trade_on_an_asset_missing_from_the_price_frames_is_skipped():
    series = {k: v for k, v in BASE_SERIES.items() if k != "GLD"}
    opens, closes = _price_frames(series)
    strategies = [Strategy("GLD", "long", 1), Strategy("SPY", "long", 1)]
    out = _labels(_slots(_slot("a", DAYS[0], "open")), strategies, opens=opens, closes=closes)
    assert out["asset"].tolist() == ["SPY"]


def test_trade_whose_exit_date_is_not_in_the_price_frames_is_skipped():
    opens, closes = OPENS.drop(index=DAYS[4]), CLOSES.drop(index=DAYS[4])
    strategies = [Strategy("SPY", "long", 1), Strategy("SPY", "long", 3)]
    out = _labels(_slots(_slot("a", DAYS[2], "open")), strategies, opens=opens, closes=closes)
    assert out["horizon"].tolist() == [1]  # horizon 3 exits on DAYS[4], which is absent


def test_output_has_label_columns_in_order_and_rows_in_slot_then_strategy_order():
    slots = _slots(_slot("s1", DAYS[0], "open"), _slot("s2", DAYS[1], "close"))
    strategies = [Strategy("SPY", "long", 1), Strategy("SPY", "short", 3)]
    out = _labels(slots, strategies)
    assert list(out.columns) == [
        "slot_id",
        "strategy_id",
        "asset",
        "rule",
        "horizon",
        "sign",
        "entry_kind",
        "entry_session",
        "entry_utc",
        "exit_session",
        "exit_utc",
        "entry_price",
        "exit_price",
        "gross_ret",
        "net_ret",
        "win",
    ]
    assert list(out.columns) == list(LABEL_COLUMNS)
    assert out[["slot_id", "strategy_id"]].values.tolist() == [
        ["s1", "SPY|long|1"],
        ["s1", "SPY|short|3"],
        ["s2", "SPY|long|1"],
        ["s2", "SPY|short|3"],
    ]
    assert out["win"].dtype == bool

    # The close entry on DAYS[1] with horizon 3 exits on DAYS[4], at SPY close 105.
    row = out.iloc[3]
    assert row["sign"] == -1
    assert row["horizon"] == 3
    assert row["entry_price"] == 103
    assert row["exit_session"] == DAYS[4]
    assert row["exit_price"] == 105
    assert row["exit_utc"] == CAL.close_at(DAYS[4])


def test_no_trades_gives_an_empty_frame_with_the_label_columns():
    out = _labels(_slots(), [Strategy("SPY", "long", 1)])
    assert out.empty
    assert list(out.columns) == list(LABEL_COLUMNS)

    # A beat rule on a zero surprise trades nothing.
    out = _labels(_slots(_slot("a", DAYS[0], "open", 0.0)), [Strategy("SPY", "long_on_beat", 1)])
    assert out.empty
    assert list(out.columns) == list(LABEL_COLUMNS)


def test_unknown_entry_kind_raises():
    slot = {
        "slot_id": "a",
        "entry_session": DAYS[0],
        "entry_kind": "midday",
        "entry_utc": CAL.open_at(DAYS[0]),
        "surprise_z": 1.0,
    }
    with pytest.raises(ValueError):
        _labels(_slots(slot), [Strategy("SPY", "long", 1)])


def test_entry_session_outside_the_calendar_raises():
    weekend = date(2026, 1, 10)  # a Saturday, not a session in CAL
    slot = {
        "slot_id": "a",
        "entry_session": weekend,
        "entry_kind": "open",
        "entry_utc": pd.Timestamp("2026-01-10T14:30:00Z"),
        "surprise_z": 1.0,
    }
    with pytest.raises(ValueError):
        _labels(_slots(slot), [Strategy("SPY", "long", 1)])
