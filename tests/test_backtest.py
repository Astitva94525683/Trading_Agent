"""Tests for macro_agent.backtest: trade selection, trade returns and the calendar-time portfolio.

Prices used throughout (SPY), with cost_bps = 2 so each side costs C = 0.0002:
    session   D1    D2    D3    D4    D5
    open      100   102   101   103   104
    close     101    99   104   102   105
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from macro_agent.backtest import portfolio_returns, select_trades, trade_daily_returns

D1, D2, D3, D4, D5 = (date(2026, 1, day) for day in range(5, 10))
SESSIONS = [D1, D2, D3, D4, D5]
NOT_A_SESSION = date(2026, 1, 10)
COST_BPS = 2.0
C = COST_BPS / 1e4


@pytest.fixture
def prices():
    opens = pd.DataFrame({"SPY": [100.0, 102.0, 101.0, 103.0, 104.0]}, index=SESSIONS)
    closes = pd.DataFrame({"SPY": [101.0, 99.0, 104.0, 102.0, 105.0]}, index=SESSIONS)
    return opens, closes


def _trade(entry, exit_, kind="open", sign=1, asset="SPY"):
    return {
        "asset": asset,
        "sign": sign,
        "entry_kind": kind,
        "entry_session": entry,
        "exit_session": exit_,
    }


# ---------------------------------------------------------------- select_trades

PREDS = pd.DataFrame(
    {
        "slot_id": [1, 1, 1, 1, 2, 2, 3],
        "strategy_id": [3, 1, 2, 4, 5, 6, 7],
        "pred": [0.02, 0.05, 0.05, -0.01, 0.03, 0.0, np.nan],
        "asset": ["GLD", "SPY", "TLT", "UUP", "SPY", "TLT", "GLD"],
    }
)


def _keys(frame: pd.DataFrame) -> list[tuple[int, int]]:
    return list(zip(frame["slot_id"], frame["strategy_id"]))


@pytest.mark.parametrize(
    "top_k, min_pred, expected",
    [
        (1, 0.0, [(1, 1), (2, 5)]),  # slot 1 tie at 0.05 goes to strategy 1; NaN and 0.0 excluded
        (2, 0.0, [(1, 1), (1, 2), (2, 5)]),
        (3, 0.0, [(1, 1), (1, 2), (1, 3), (2, 5)]),
        (3, 0.02, [(1, 1), (1, 2), (2, 5)]),  # min_pred is strict: 0.02 is not above 0.02
        (3, 0.04, [(1, 1), (1, 2)]),
        (3, 0.05, []),
    ],
)
def test_select_trades_top_k_and_min_pred(top_k, min_pred, expected):
    assert _keys(select_trades(PREDS, top_k=top_k, min_pred=min_pred)) == expected


def test_select_trades_breaks_ties_by_strategy_id_not_input_order():
    reversed_rows = PREDS.iloc[::-1]
    assert _keys(select_trades(reversed_rows, top_k=1)) == [(1, 1), (2, 5)]
    assert _keys(select_trades(reversed_rows, top_k=2)) == [(1, 1), (1, 2), (2, 5)]


def test_select_trades_keeps_columns_and_sorts_by_slot_then_score():
    out = select_trades(PREDS, top_k=3)
    assert list(out.columns) == list(PREDS.columns)
    assert list(out.index) == list(range(len(out)))
    assert list(out["pred"]) == [0.05, 0.05, 0.02, 0.03]


def test_select_trades_uses_score_col():
    preds = PREDS.rename(columns={"pred": "score"})
    assert _keys(select_trades(preds, top_k=1, score_col="score")) == [(1, 1), (2, 5)]


def test_select_trades_rejects_bad_arguments():
    with pytest.raises(ValueError):
        select_trades(PREDS, top_k=0)
    with pytest.raises(ValueError):
        select_trades(PREDS.drop(columns="strategy_id"))


# ------------------------------------------------------- trade_daily_returns

@pytest.mark.parametrize(
    "trade, expected_days, expected",
    [
        # open entry D1, horizon 1: entry and exit on the same day pay both costs
        (_trade(D1, D1), [D1], [101 / 100 - 1 - 2 * C]),
        # long open entry D1 to D3 (horizon 3): cost on D1 and on D3 only
        (
            _trade(D1, D3),
            [D1, D2, D3],
            [101 / 100 - 1 - C, 99 / 101 - 1, 104 / 99 - 1 - C],
        ),
        # short open entry D1 to D3: signs flip, costs unchanged
        (
            _trade(D1, D3, sign=-1),
            [D1, D2, D3],
            [-(101 / 100 - 1) - C, -(99 / 101 - 1), -(104 / 99 - 1) - C],
        ),
        # long close entry on D2 (close 99) to D4 (horizon 2): exposed D3 and D4
        (
            _trade(D2, D4, kind="close"),
            [D3, D4],
            [104 / 99 - 1 - C, 102 / 104 - 1 - C],
        ),
        # short close entry on D1 (close 101) to D4 (horizon 3): exposed D2, D3, D4
        (
            _trade(D1, D4, kind="close", sign=-1),
            [D2, D3, D4],
            [-(99 / 101 - 1) - C, -(104 / 99 - 1), -(102 / 104 - 1) - C],
        ),
        # close entry with one exposed session: both costs fall on D2
        (_trade(D1, D2, kind="close"), [D2], [99 / 101 - 1 - 2 * C]),
    ],
)
def test_trade_daily_returns(prices, trade, expected_days, expected):
    opens, closes = prices
    out = trade_daily_returns(trade, opens, closes, COST_BPS)
    assert list(out.index) == expected_days
    assert all(isinstance(day, date) for day in out.index)
    np.testing.assert_allclose(out.to_numpy(), expected, rtol=0, atol=1e-12)


def test_trade_daily_returns_nan_close_in_window_raises(prices):
    opens, closes = prices
    closes = closes.copy()
    closes.at[D3, "SPY"] = np.nan
    with pytest.raises(ValueError):
        trade_daily_returns(_trade(D1, D3), opens, closes, COST_BPS)


def test_trade_daily_returns_nan_open_raises(prices):
    opens, closes = prices
    opens = opens.copy()
    opens.at[D1, "SPY"] = np.nan
    with pytest.raises(ValueError):
        trade_daily_returns(_trade(D1, D2), opens, closes, COST_BPS)


def test_trade_daily_returns_empty_window_raises(prices):
    opens, closes = prices
    with pytest.raises(ValueError):
        trade_daily_returns(_trade(D2, D2, kind="close"), opens, closes, COST_BPS)
    with pytest.raises(ValueError):
        trade_daily_returns(_trade(D3, D2), opens, closes, COST_BPS)


def test_trade_daily_returns_rejects_dates_that_are_not_sessions(prices):
    opens, closes = prices
    with pytest.raises(ValueError):
        trade_daily_returns(_trade(NOT_A_SESSION, D3), opens, closes, COST_BPS)
    with pytest.raises(ValueError):
        trade_daily_returns(_trade(D1, NOT_A_SESSION), opens, closes, COST_BPS)


def test_trade_daily_returns_rejects_bad_sign_or_entry_kind(prices):
    opens, closes = prices
    with pytest.raises(ValueError):
        trade_daily_returns(_trade(D1, D2, sign=0), opens, closes, COST_BPS)
    with pytest.raises(ValueError):
        trade_daily_returns(_trade(D1, D2, kind="midday"), opens, closes, COST_BPS)


# --------------------------------------------------------- portfolio_returns

TRADE_A = _trade(D1, D2)  # long, open D1 -> D2
TRADE_B = _trade(D2, D3, sign=-1)  # short, open D2 -> D3: overlaps A on D2
TRADE_C = _trade(D5, D5)  # long, open D5 -> D5: after the flat day D4

A_D1 = 101 / 100 - 1 - C
A_D2 = 99 / 101 - 1 - C
B_D2 = -(99 / 102 - 1) - C
B_D3 = -(104 / 99 - 1) - C
C_D5 = 105 / 104 - 1 - 2 * C


def test_portfolio_averages_overlapping_trades_and_counts_active(prices):
    opens, closes = prices
    trades = pd.DataFrame([TRADE_A, TRADE_B, TRADE_C])
    out = portfolio_returns(trades, opens, closes, COST_BPS)
    assert list(out.index) == SESSIONS
    assert list(out.columns) == ["ret", "n_active"]
    np.testing.assert_allclose(
        out["ret"].to_numpy(),
        [A_D1, (A_D2 + B_D2) / 2, B_D3, 0.0, C_D5],
        rtol=0,
        atol=1e-12,
    )
    assert list(out["n_active"]) == [1, 2, 1, 0, 1]
    assert out["n_active"].dtype.kind == "i"


def test_portfolio_default_range_ends_at_last_exit(prices):
    opens, closes = prices
    out = portfolio_returns(pd.DataFrame([TRADE_A, TRADE_B]), opens, closes, COST_BPS)
    assert list(out.index) == [D1, D2, D3]


def test_portfolio_flat_days_are_zero_when_range_is_extended(prices):
    opens, closes = prices
    trades = pd.DataFrame([TRADE_A, TRADE_B])
    out = portfolio_returns(trades, opens, closes, COST_BPS, start=D1, end=D5)
    assert list(out.index) == SESSIONS
    assert out.loc[D4, "ret"] == 0.0
    assert out.loc[D4, "n_active"] == 0
    assert out.loc[D5, "ret"] == 0.0
    assert out.loc[D5, "n_active"] == 0


def test_portfolio_without_trades(prices):
    opens, closes = prices
    columns = ["asset", "sign", "entry_kind", "entry_session", "exit_session"]
    no_trades = pd.DataFrame(columns=columns)
    empty = portfolio_returns(no_trades, opens, closes, COST_BPS)
    assert list(empty.columns) == ["ret", "n_active"]
    assert len(empty) == 0

    flat = portfolio_returns(no_trades, opens, closes, COST_BPS, start=D2, end=D4)
    assert list(flat.index) == [D2, D3, D4]
    assert (flat["ret"] == 0.0).all()
    assert (flat["n_active"] == 0).all()
