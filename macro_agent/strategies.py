"""Strategy grid, price pivots and trade labels.

Specification: docs/design.md section 5. A strategy is (asset, rule, horizon). A label is one
strategy's trade on one slot: the sign from the rule, the entry and exit from the session
calendar, and the return after costs.
"""
from __future__ import annotations

import math
import numbers
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import pandas as pd

from macro_agent.config import ASSETS, HORIZONS, RULES
from macro_agent.timeutils import Entry, SessionCalendar

LABEL_COLUMNS = (
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
)
_LABEL_DTYPES = {
    "sign": "int64",
    "horizon": "int64",
    "entry_price": "float64",
    "exit_price": "float64",
    "gross_ret": "float64",
    "net_ret": "float64",
    "win": "bool",
}
_SLOT_COLUMNS = ("slot_id", "entry_session", "entry_kind", "entry_utc", "surprise_z")
_PRICE_COLUMNS = ("symbol", "date", "open", "close")


@dataclass(frozen=True)
class Strategy:
    """A permutation to test: trade `asset` under `rule`, holding for `horizon` sessions."""

    asset: str
    rule: str
    horizon: int

    def __post_init__(self) -> None:
        if self.rule not in RULES:
            raise ValueError(f"unknown rule: {self.rule!r}")
        if not isinstance(self.horizon, numbers.Integral) or self.horizon < 1:
            raise ValueError(f"horizon must be an integer >= 1: {self.horizon!r}")

    @property
    def id(self) -> str:
        """Stable key such as 'SPY|long_on_beat|3'."""
        return f"{self.asset}|{self.rule}|{self.horizon}"


def strategy_grid(
    assets: Sequence[str] = tuple(ASSETS),
    rules: Sequence[str] = RULES,
    horizons: Sequence[int] = HORIZONS,
) -> list[Strategy]:
    """All (asset, rule, horizon) combinations, asset outermost and horizon innermost."""
    assets, rules, horizons = tuple(assets), tuple(rules), tuple(horizons)
    return [
        Strategy(asset, rule, horizon)
        for asset in assets
        for rule in rules
        for horizon in horizons
    ]


def _direction(surprise: float | None) -> int:
    """Sign of a surprise: 1 or -1, or 0 when it is None, NaN or exactly zero."""
    if surprise is None or pd.isna(surprise):
        return 0
    if surprise > 0:
        return 1
    if surprise < 0:
        return -1
    return 0


def position_sign(rule: str, surprise: float | None) -> int:
    """Position for a slot under `rule`: 1 long, -1 short or 0 flat.

    long_on_beat follows the sign of the surprise and short_on_beat its negative. Both stay
    flat when the surprise is None, NaN or zero. Raises ValueError for an unknown rule.
    """
    if rule == "long":
        return 1
    if rule == "short":
        return -1
    if rule == "long_on_beat":
        return _direction(surprise)
    if rule == "short_on_beat":
        return -_direction(surprise)
    raise ValueError(f"unknown rule: {rule!r}")


def _as_date(value: object) -> date:
    """Coerce a date, Timestamp or 'YYYY-MM-DD' string to a datetime.date."""
    if not isinstance(value, (str, date)):
        raise ValueError(f"unsupported date type: {type(value).__name__}")
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"cannot parse date: {value!r}") from exc
    if pd.isna(ts):
        raise ValueError(f"missing date: {value!r}")
    return ts.date()


def pivot_prices(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wide (opens, closes) frames from long prices with symbol, date, open and close.

    Dates may be datetime.date objects, 'YYYY-MM-DD' strings or Timestamps. Each frame is
    indexed by the sorted session dates and has one column per symbol; absent cells are NaN.
    Raises ValueError for a missing column or a repeated (symbol, date) pair.
    """
    missing = [c for c in _PRICE_COLUMNS if c not in prices.columns]
    if missing:
        raise ValueError(f"prices is missing columns: {missing}")
    frame = prices.loc[:, list(_PRICE_COLUMNS)].assign(
        date=[_as_date(value) for value in prices["date"]]
    )
    if frame.duplicated(["symbol", "date"]).any():
        raise ValueError("prices has more than one row for a symbol and date")
    opens = frame.pivot(index="date", columns="symbol", values="open")
    closes = frame.pivot(index="date", columns="symbol", values="close")
    return opens, closes


def _price(frame: pd.DataFrame, day: date, asset: str) -> float | None:
    """Price for `asset` on `day`, or None when it is absent, NaN, infinite or not positive."""
    if day not in frame.index or asset not in frame.columns:
        return None
    value = frame.at[day, asset]
    if pd.isna(value):
        return None
    price = float(value)
    if not math.isfinite(price) or price <= 0:
        return None
    return price


def compute_labels(
    slots: pd.DataFrame,
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    calendar: SessionCalendar,
    strategies: Sequence[Strategy],
    cost_bps: float,
) -> pd.DataFrame:
    """Label each slot under each strategy with its return after round-trip costs.

    `slots` needs columns slot_id, entry_session, entry_kind ("open" or "close"), entry_utc
    and surprise_z. `opens` and `closes` are the frames from pivot_prices. A trade is skipped
    when its position sign is 0, when the calendar has no exit, or when a price it needs is
    missing or not positive. Costs are cost_bps per side. Rows follow slot order, then
    strategy order, with the columns in LABEL_COLUMNS.
    """
    missing = [c for c in _SLOT_COLUMNS if c not in slots.columns]
    if missing:
        raise ValueError(f"slots is missing columns: {missing}")
    strategies = tuple(strategies)
    cost = 2 * cost_bps / 1e4
    rows = []
    for slot in slots.itertuples(index=False):
        day = _as_date(slot.entry_session)
        entry = Entry(day, slot.entry_kind, slot.entry_utc)
        entry_book = opens if entry.kind == "open" else closes
        for strategy in strategies:
            sign = position_sign(strategy.rule, slot.surprise_z)
            if sign == 0:
                continue
            exit_point = calendar.exit_for(entry, strategy.horizon)
            if exit_point is None:
                continue
            exit_day, exit_utc = exit_point
            entry_price = _price(entry_book, day, strategy.asset)
            exit_price = _price(closes, exit_day, strategy.asset)
            if entry_price is None or exit_price is None:
                continue
            gross = sign * (exit_price / entry_price - 1)
            net = gross - cost
            rows.append(
                {
                    "slot_id": slot.slot_id,
                    "strategy_id": strategy.id,
                    "asset": strategy.asset,
                    "rule": strategy.rule,
                    "horizon": strategy.horizon,
                    "sign": sign,
                    "entry_kind": entry.kind,
                    "entry_session": day,
                    "entry_utc": entry.at_utc,
                    "exit_session": exit_day,
                    "exit_utc": exit_utc,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "gross_ret": gross,
                    "net_ret": net,
                    "win": net > 0,
                }
            )
    return pd.DataFrame(rows, columns=list(LABEL_COLUMNS)).astype(_LABEL_DTYPES)
