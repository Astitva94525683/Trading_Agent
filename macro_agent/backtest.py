"""Trade selection and the calendar-time portfolio.

A trade is a mapping (or row) with asset, sign (+1 long, -1 short), entry_kind
("open" or "close"), entry_session and exit_session (datetime.date). Price frames
are wide: index = sorted session dates (datetime.date), one column per symbol.
Conventions are in docs/design.md section 8.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping

import numpy as np
import pandas as pd

_TRADE_KEYS = ("asset", "sign", "entry_kind", "entry_session", "exit_session")


def select_trades(
    preds: pd.DataFrame,
    top_k: int = 1,
    min_pred: float = 0.0,
    score_col: str = "pred",
) -> pd.DataFrame:
    """Keep the top_k strategies per slot whose score is above min_pred.

    preds needs slot_id, strategy_id and score_col columns, with one row per
    (slot_id, strategy_id). Ties are broken by strategy_id ascending. The result
    keeps all columns, has a fresh index, and is sorted by slot_id, then score descending.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    missing = [c for c in ("slot_id", "strategy_id", score_col) if c not in preds.columns]
    if missing:
        raise ValueError(f"preds is missing columns: {missing}")
    eligible = preds[preds[score_col] > min_pred]
    ranked = eligible.sort_values([score_col, "strategy_id"], ascending=[False, True])
    top = ranked.groupby("slot_id", sort=False).head(top_k)
    result = top.sort_values(
        ["slot_id", score_col, "strategy_id"], ascending=[True, False, True]
    )
    return result.reset_index(drop=True)


def trade_daily_returns(
    trade: Mapping,
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    cost_bps: float,
) -> pd.Series:
    """Net daily returns of one trade over its exposure window.

    An open entry is exposed from entry_session to exit_session inclusive. Its first
    return is close / open - 1 on the entry session. A close entry is exposed on the
    sessions after entry_session up to exit_session. Its first return is measured from
    the entry session's close. Later days are close-to-close. Each return is multiplied
    by sign, and cost_bps / 1e4 is subtracted on the first and on the last exposed day
    (twice on one day when the window is a single session).

    Returns a Series indexed by session date. Raises ValueError for an unknown sign or
    entry_kind, an entry or exit session missing from closes, an empty exposure window,
    or a missing, NaN or non-positive price.
    """
    absent = [k for k in _TRADE_KEYS if k not in trade]
    if absent:
        raise ValueError(f"trade is missing {absent}")
    asset, sign, kind = trade["asset"], trade["sign"], trade["entry_kind"]
    if sign not in (1, -1):
        raise ValueError(f"sign must be +1 or -1, got {sign!r}")
    if kind not in ("open", "close"):
        raise ValueError(f"entry_kind must be 'open' or 'close', got {kind!r}")
    if asset not in closes.columns or (kind == "open" and asset not in opens.columns):
        raise ValueError(f"no price column for {asset!r}")
    entry, exit_ = _as_date(trade["entry_session"]), _as_date(trade["exit_session"])
    if entry not in closes.index or exit_ not in closes.index:
        raise ValueError(f"entry {entry} or exit {exit_} is not a session in closes")

    if kind == "open":
        exposed = [d for d in closes.index if entry <= d <= exit_]
        base = _price(opens, asset, entry, "open")
    else:
        exposed = [d for d in closes.index if entry < d <= exit_]
        base = _price(closes, asset, entry, "close")
    if not exposed:
        raise ValueError(f"empty exposure window {entry}..{exit_} for {asset}")

    path = np.array([_price(closes, asset, d, "close") for d in exposed])
    prev = np.concatenate(([base], path[:-1]))
    daily = float(sign) * (path / prev - 1.0)
    cost = cost_bps / 1e4
    daily[0] -= cost
    daily[-1] -= cost
    return pd.Series(daily, index=exposed, dtype=float)


def portfolio_returns(
    trades: pd.DataFrame,
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    cost_bps: float,
    start: dt.date | None = None,
    end: dt.date | None = None,
) -> pd.DataFrame:
    """Calendar-time, equal-weight portfolio of trades.

    Every session in closes between start and end (inclusive) gets two columns:
    ret, the mean of the daily returns of the trades active that session (0.0 when
    none are), and n_active, the number of active trades. By default the range runs
    from the first exposed session to the last exit session over all trades. With no
    trades and no start or end, the result is an empty frame with the same columns.
    """
    per_trade = [
        trade_daily_returns(trade, opens, closes, cost_bps)
        for trade in trades.to_dict("records")
    ]
    if per_trade:
        first = min(s.index[0] for s in per_trade)
        last = max(s.index[-1] for s in per_trade)
    elif (start is None and end is None) or len(closes.index) == 0:
        return _empty_portfolio()
    else:
        first, last = closes.index[0], closes.index[-1]
    lo = first if start is None else _as_date(start)
    hi = last if end is None else _as_date(end)
    days = [d for d in closes.index if lo <= d <= hi]

    by_day: dict[dt.date, list[float]] = {}
    for series in per_trade:
        for day, value in series.items():
            by_day.setdefault(day, []).append(float(value))
    ret = [float(np.mean(by_day[d])) if d in by_day else 0.0 for d in days]
    n_active = [len(by_day.get(d, ())) for d in days]
    return pd.DataFrame(
        {
            "ret": np.asarray(ret, dtype=float),
            "n_active": np.asarray(n_active, dtype="int64"),
        },
        index=pd.Index(days, dtype=object),
    )


def _price(prices: pd.DataFrame, asset: str, day: dt.date, label: str) -> float:
    """Return one price, raising ValueError if it is missing, NaN or non-positive."""
    try:
        value = float(prices.at[day, asset])
    except KeyError:
        value = float("nan")
    if not (np.isfinite(value) and value > 0.0):
        raise ValueError(f"missing or non-positive {label} price for {asset} on {day}")
    return value


def _as_date(value: object) -> dt.date:
    """Coerce a date-like value (date, datetime, Timestamp or ISO string) to datetime.date."""
    if type(value) is dt.date:
        return value
    return pd.Timestamp(value).date()


def _empty_portfolio() -> pd.DataFrame:
    """Empty portfolio frame with the ret and n_active columns."""
    return pd.DataFrame(
        {"ret": np.array([], dtype=float), "n_active": np.array([], dtype="int64")},
        index=pd.Index([], dtype=object),
    )
