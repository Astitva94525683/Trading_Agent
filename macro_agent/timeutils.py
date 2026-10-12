"""Time alignment: release timestamps in UTC and New York time, and the session calendar.

Specification: docs/design.md section 3. Stored instants are UTC. Naive inputs are read
in New York time unless `assume_tz` says otherwise.
"""
from __future__ import annotations

import bisect
import datetime
import operator
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, time, tzinfo

import pandas as pd

from macro_agent.config import ET

UTC = datetime.timezone.utc


def _parse(value: object) -> pd.Timestamp:
    """Parse a str, date or datetime (Timestamp included). Rejects other types and NaT."""
    if not isinstance(value, (str, date)):
        raise ValueError(f"unsupported timestamp type: {type(value).__name__}")
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"cannot parse timestamp: {value!r}") from exc
    if ts is pd.NaT:
        raise ValueError(f"missing timestamp: {value!r}")
    return ts


def _localize(ts: pd.Timestamp, tz: str | tzinfo) -> pd.Timestamp:
    """Attach a zone to a naive timestamp. DST gaps and repeated hours raise ValueError."""
    try:
        return ts.tz_localize(tz, ambiguous="raise", nonexistent="raise")
    except ValueError as exc:
        raise ValueError(f"{ts} does not exist or is ambiguous in {tz}") from exc


def _utc_stamp(value: object, label: str) -> pd.Timestamp:
    """Parse a timezone-aware value and return it in UTC."""
    ts = _parse(value)
    if ts.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware: {value!r}")
    return ts.tz_convert(UTC)


def _to_date(value: object) -> date:
    """Coerce a date, datetime, Timestamp or ISO string to a calendar date."""
    return _parse(value).date()


def _count(value: object, name: str) -> int:
    """Require an integer (numpy integers included)."""
    try:
        return operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must be an integer: {value!r}") from None


def to_utc(value: str | date, assume_tz: str = "America/New_York") -> pd.Timestamp:
    """Return `value` as a UTC Timestamp.

    Aware values convert directly. Naive values are localized to `assume_tz` first.
    Raises ValueError for unparseable input and for local times that do not exist or repeat.
    """
    ts = _parse(value)
    if ts.tzinfo is None:
        ts = _localize(ts, assume_tz)
    return ts.tz_convert(UTC)


def to_et(value: str | date, assume_tz: str = "America/New_York") -> pd.Timestamp:
    """Return `value` as a New York (ET) Timestamp. Input rules as in `to_utc`."""
    return to_utc(value, assume_tz).tz_convert(ET)


def iso_utc(value: str | date) -> str:
    """Format `value` as 'YYYY-MM-DDTHH:MM:SSZ' in UTC, to the second. Naive values are ET."""
    return to_utc(value).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class Entry:
    """A trade entry: its session, its kind ("open" or "close") and its UTC time."""

    session: date
    kind: str  # "open" | "close"
    at_utc: pd.Timestamp

    def __post_init__(self) -> None:
        if self.kind not in ("open", "close"):
            raise ValueError(f"kind must be 'open' or 'close': {self.kind!r}")


class SessionCalendar:
    """Trading sessions in date order, each with a UTC open and close."""

    def __init__(self, sessions: pd.DataFrame) -> None:
        """Build from a frame with columns `date`, `open_utc` and `close_utc`.

        Raises ValueError if a column is missing, dates are not strictly ascending,
        a time is not timezone-aware, or a session does not open before it closes.
        """
        missing = [c for c in ("date", "open_utc", "close_utc") if c not in sessions.columns]
        if missing:
            raise ValueError(f"sessions is missing columns: {missing}")
        days = [_to_date(v) for v in sessions["date"]]
        opens = [_utc_stamp(v, "open_utc") for v in sessions["open_utc"]]
        closes = [_utc_stamp(v, "close_utc") for v in sessions["close_utc"]]
        for prev, cur in zip(days, days[1:]):
            if cur <= prev:
                raise ValueError("session dates must be unique and in ascending order")
        for day, opened, closed in zip(days, opens, closes):
            if opened >= closed:
                raise ValueError(f"open must be before close on {day}")
        self._dates = days
        self._opens = opens
        self._closes = closes
        self._pos = {day: i for i, day in enumerate(days)}

    @classmethod
    def from_dates(
        cls,
        dates: Iterable[date | str],
        open_et: time = time(9, 30),
        close_et: time = time(16, 0),
        early_closes: Mapping[date, time] | None = None,
    ) -> SessionCalendar:
        """Build a calendar from session dates, in any order, with duplicates removed.

        Open and close are built in ET and converted to UTC, so they follow DST.
        `early_closes` maps a session date to its ET close. Entries for non-session
        dates are ignored.
        """
        days = sorted({_to_date(d) for d in dates})
        overrides = {_to_date(k): v for k, v in (early_closes or {}).items()}
        rows = []
        for day in days:
            opened = _localize(pd.Timestamp(datetime.datetime.combine(day, open_et)), ET)
            close_time = overrides.get(day, close_et)
            closed = _localize(pd.Timestamp(datetime.datetime.combine(day, close_time)), ET)
            rows.append((day, opened.tz_convert(UTC), closed.tz_convert(UTC)))
        return cls(pd.DataFrame(rows, columns=["date", "open_utc", "close_utc"]))

    @property
    def dates(self) -> list[date]:
        """Session dates in ascending order."""
        return list(self._dates)

    def __len__(self) -> int:
        return len(self._dates)

    def __contains__(self, d: object) -> bool:
        try:
            return _to_date(d) in self._pos
        except ValueError:
            return False

    def _index(self, d: object) -> int:
        day = _to_date(d)
        if day not in self._pos:
            raise KeyError(f"{day} is not a session")
        return self._pos[day]

    def open_at(self, d: date) -> pd.Timestamp:
        """UTC open of session `d`. KeyError if `d` is not a session."""
        return self._opens[self._index(d)]

    def close_at(self, d: date) -> pd.Timestamp:
        """UTC close of session `d`. KeyError if `d` is not a session."""
        return self._closes[self._index(d)]

    def next_session(self, d: date, inclusive: bool = True) -> date | None:
        """First session on or after `d`, or strictly after if not inclusive. None if none."""
        day = _to_date(d)
        i = bisect.bisect_left(self._dates, day)
        if not inclusive and i < len(self._dates) and self._dates[i] == day:
            i += 1
        return self._dates[i] if i < len(self._dates) else None

    def offset(self, d: date, n: int) -> date | None:
        """Session `n` positions after session `d` (n=0 gives `d`). None if past the last session.

        Raises ValueError if `d` is not a session or `n` is negative.
        """
        steps = _count(n, "n")
        if steps < 0:
            raise ValueError("n must be >= 0")
        day = _to_date(d)
        if day not in self._pos:
            raise ValueError(f"{day} is not a session")
        j = self._pos[day] + steps
        return self._dates[j] if j < len(self._dates) else None

    def entry_for(self, release: str | date) -> Entry | None:
        """Entry for a release time, with no look-ahead. Naive values are ET.

        Before the open of a session day, enter at its open. From the open up to the close,
        enter at its close. At or after the close, or on a non-session day, enter at the
        next session's open. None if no later session exists.
        """
        stamp = to_utc(release)
        day = stamp.tz_convert(ET).date()
        i = self._pos.get(day)
        if i is not None:
            if stamp < self._opens[i]:
                return Entry(day, "open", self._opens[i])
            if stamp < self._closes[i]:
                return Entry(day, "close", self._closes[i])
        nxt = self.next_session(day, inclusive=False)
        if nxt is None:
            return None
        return Entry(nxt, "open", self.open_at(nxt))

    def exit_for(self, entry: Entry, horizon: int) -> tuple[date, pd.Timestamp] | None:
        """Session and UTC close where a trade exits after `horizon` sessions.

        An open entry counts its own session, so it exits at session + (horizon - 1).
        A close entry exits at session + horizon. None if beyond the calendar.
        Raises ValueError if `horizon` is below 1, or if the entry's session is not in
        the calendar.
        """
        count = _count(horizon, "horizon")
        if count < 1:
            raise ValueError("horizon must be >= 1")
        steps = count - 1 if entry.kind == "open" else count
        session = self.offset(entry.session, steps)
        if session is None:
            return None
        return session, self.close_at(session)
