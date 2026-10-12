"""Event preparation, slots, time features and macro surprise features.

Specification: docs/design.md sections 4 and 6. Every feature uses only information known at
a slot's entry time. News, market and panel features are added in later steps.
"""

from __future__ import annotations

import bisect
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from macro_agent.config import CATEGORIES, ET
from macro_agent.parsing import impact_weight
from macro_agent.timeutils import SessionCalendar, to_utc

_NAN = float("nan")
_DAY = pd.Timedelta(days=1)
_EPOCH = pd.Timestamp(0, tz="UTC")
_CATEGORY_RANK = {name: rank for rank, name in enumerate(CATEGORIES)}

PREPARE_INPUT_COLUMNS = (
    "event_id", "event_name", "category", "impact", "release_utc", "actual", "forecast", "inverted",
)
SLOT_INPUT_COLUMNS = ("event_id", "event_name", "category", "weight", "release_utc", "surprise_z")
SLOT_COLUMNS = (
    "slot_id", "entry_session", "entry_kind", "entry_utc", "last_release_utc", "n_events",
    "n_high", "primary_event", "primary_category", "surprise_z", "abs_surprise_z", "has_surprise",
    *(f"cat_{name}" for name in CATEGORIES),
    "event_ids",
)


# --- Helpers -----------------------------------------------------------------


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    """Raise ValueError if `frame` lacks any of `columns`."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def _utc_series(values: Iterable[object], index: pd.Index) -> pd.Series:
    """Convert each value with `to_utc`, so naive values are read as New York time."""
    return pd.Series([to_utc(v) for v in values], index=index, dtype="datetime64[ns, UTC]")


def _days(stamps: pd.Series) -> np.ndarray:
    """Days since 1970-01-01 UTC as floats; NaN for missing values."""
    utc = pd.to_datetime(stamps, utc=True)
    return ((utc - _EPOCH) / _DAY).to_numpy(dtype=float, na_value=np.nan)


def _as_date(value: object) -> date:
    """Return the calendar date of a date, datetime, Timestamp or ISO string."""
    if type(value) is date:
        return value
    return pd.Timestamp(value).date()


def _number(value: object) -> float:
    """Return `value` as a float, with missing or non-numeric input as NaN."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return _NAN


def _flag(value: object) -> bool:
    """Read an `inverted` flag. A missing value counts as False."""
    return False if pd.isna(value) else bool(value)


def _weighted_mean(weights: np.ndarray, values: np.ndarray) -> float:
    """Weighted mean of `values`. NaN when there are none or the weights sum to zero."""
    total = float(weights.sum())
    return float((weights * values).sum() / total) if total > 0 else _NAN


# --- Events ------------------------------------------------------------------


def _surprise_z(names: pd.Series, release: pd.Series, surprise: np.ndarray) -> np.ndarray:
    """Scale each surprise by the median |surprise| of its event's earlier releases.

    `release` and `surprise` must be sorted by release time. Only releases strictly earlier
    than the row count as priors. With fewer than two priors, or a zero median, the value
    falls back to its sign. A NaN surprise gives a NaN value.
    """
    z = np.full(surprise.shape, _NAN)
    times = pd.DatetimeIndex(release).asi8
    groups: dict[object, list[int]] = {}
    for pos, name in enumerate(names):
        groups.setdefault(None if pd.isna(name) else name, []).append(pos)
    for members in groups.values():
        idx = np.asarray(members, dtype=np.intp)
        group_times = times[idx]
        group_values = surprise[idx]
        for k, pos in enumerate(idx):
            current = group_values[k]
            if np.isnan(current):
                continue
            end = np.searchsorted(group_times, group_times[k], side="left")
            prior = np.abs(group_values[:end])
            prior = prior[~np.isnan(prior)]
            scale = float(np.median(prior)) if prior.size >= 2 else 0.0
            if scale > 0.0:
                z[pos] = float(np.clip(current / scale, -3.0, 3.0))
            else:
                z[pos] = float(np.sign(current))
    return z


def prepare_events(events: pd.DataFrame, min_weight: float = 0.5) -> pd.DataFrame:
    """Prepare raw calendar events: UTC release time, weight, surprise and scaled surprise.

    Rows with weight below `min_weight` are dropped first. The output is sorted by
    (release_utc, event_id). Its `surprise_z` uses only earlier kept releases of the same
    event_name. Naive release times are read as New York time (see `to_utc`). Non-numeric
    actual or forecast values count as missing. Raises ValueError for missing columns or
    an unparseable release time.
    """
    _require_columns(events, PREPARE_INPUT_COLUMNS, "events")
    frame = events.reset_index(drop=True).copy()
    frame["release_utc"] = _utc_series(frame["release_utc"], frame.index)
    frame["weight"] = pd.Series(
        [impact_weight(v) for v in frame["impact"]], index=frame.index, dtype=float
    )
    frame = frame[frame["weight"] >= min_weight]
    frame = frame.sort_values(["release_utc", "event_id"]).reset_index(drop=True)

    actual = pd.to_numeric(frame["actual"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    forecast = pd.to_numeric(frame["forecast"], errors="coerce").to_numpy(
        dtype=float, na_value=np.nan
    )
    raw = actual - forecast
    flipped = np.array([_flag(v) for v in frame["inverted"]], dtype=bool)
    surprise = np.where(flipped, -raw, raw)

    frame["surprise_raw"] = raw
    frame["surprise"] = surprise
    frame["surprise_z"] = _surprise_z(frame["event_name"], frame["release_utc"], surprise)
    return frame


# --- Slots -------------------------------------------------------------------


def _primary_key(member: dict[str, Any]) -> tuple[float, int, pd.Timestamp, str]:
    """Sort key for the primary event: weight, then category order, earlier release and name."""
    rank = _CATEGORY_RANK.get(member["category"], len(CATEGORIES))
    return (-member["weight"], rank, member["release_utc"], str(member["event_name"]))


def _slot_row(session: date, kind: str, members: list[dict[str, Any]]) -> dict[str, Any]:
    """Build one slot row from the events that enter together."""
    members = sorted(members, key=lambda m: m["release_utc"])  # stable: input order breaks ties
    primary = min(members, key=_primary_key)
    scored = [(m["weight"], m["surprise_z"]) for m in members if not math.isnan(m["surprise_z"])]
    surprise_z = _weighted_mean(
        np.array([w for w, _ in scored], dtype=float),
        np.array([z for _, z in scored], dtype=float),
    )
    row: dict[str, Any] = {
        "slot_id": f"{session.isoformat()}_{kind}",
        "entry_session": session,
        "entry_kind": kind,
        "entry_utc": members[0]["entry_utc"],
        "last_release_utc": members[-1]["release_utc"],
        "n_events": len(members),
        "n_high": sum(m["weight"] == 1.0 for m in members),
        "primary_event": str(primary["event_name"]),
        "primary_category": primary["category"],
        "surprise_z": surprise_z,
        "abs_surprise_z": abs(surprise_z),
        "has_surprise": (not math.isnan(surprise_z)) and surprise_z != 0.0,
        "event_ids": ",".join(str(m["event_id"]) for m in members),
    }
    for name in CATEGORIES:
        row[f"cat_{name}"] = sum(m["category"] == name for m in members)
    return row


def build_slots(
    events: pd.DataFrame, calendar: SessionCalendar, max_gap_days: int = 4
) -> pd.DataFrame:
    """Group prepared events into slots by entry point (design.md section 4).

    Each event enters at `calendar.entry_for(release_utc)`. Events with no entry, or whose
    entry session is more than `max_gap_days` calendar days after the release's New York
    date, are dropped. Events with the same (entry session, entry kind) form one slot.
    Output is one row per slot, sorted by entry_utc.
    """
    _require_columns(events, SLOT_INPUT_COLUMNS, "events")
    groups: dict[tuple[date, str], list[dict[str, Any]]] = {}
    for record in events.to_dict("records"):
        release = to_utc(record["release_utc"])
        entry = calendar.entry_for(release)
        if entry is None:
            continue
        release_day = release.tz_convert(ET).date()
        if (entry.session - release_day).days > max_gap_days:
            continue
        groups.setdefault((entry.session, entry.kind), []).append({
            "event_id": record["event_id"],
            "event_name": record["event_name"],
            "category": record["category"],
            "weight": _number(record["weight"]),
            "surprise_z": _number(record["surprise_z"]),
            "release_utc": release,
            "entry_utc": entry.at_utc,
        })

    rows = [_slot_row(session, kind, members) for (session, kind), members in groups.items()]
    slots = pd.DataFrame(rows, columns=list(SLOT_COLUMNS))
    slots = slots.sort_values(["entry_utc", "slot_id"]).reset_index(drop=True)
    slots["entry_utc"] = pd.to_datetime(slots["entry_utc"], utc=True)
    slots["last_release_utc"] = pd.to_datetime(slots["last_release_utc"], utc=True)
    return slots.astype({
        "n_events": "int64",
        "n_high": "int64",
        "surprise_z": "float64",
        "abs_surprise_z": "float64",
        "has_surprise": "bool",
        **{f"cat_{name}": "int64" for name in CATEGORIES},
    })


# --- Time features -----------------------------------------------------------


def time_features(slots: pd.DataFrame, fomc_dates: Sequence[date]) -> pd.DataFrame:
    """Add time-based features to slots (design.md section 6).

    Adds the New York release hour, the entry weekday, day of month, week of month and
    month, the open/close flag, days since the previous slot and since the previous slot of
    the same primary category, days to the next FOMC date and since the last one (strictly
    before the entry session), and the number of slots in the same ISO week. The result
    keeps the input row order. Spacing features follow entry-time order.
    """
    _require_columns(
        slots,
        ("entry_session", "entry_kind", "entry_utc", "last_release_utc", "primary_category"),
        "slots",
    )
    out = slots.copy()
    n = len(out)
    sessions = [_as_date(v) for v in out["entry_session"]]
    kinds = out["entry_kind"].tolist()
    cats = out["primary_category"].tolist()
    entry = _days(out["entry_utc"])
    released_et = pd.to_datetime(out["last_release_utc"], utc=True).dt.tz_convert(ET)
    release_hour = (released_et.dt.hour + released_et.dt.minute / 60).to_numpy(
        dtype=float, na_value=np.nan
    )

    days_prev = np.full(n, _NAN)
    days_prev_cat = np.full(n, _NAN)
    previous: float | None = None
    last_by_cat: dict[object, float] = {}
    for i in sorted(range(n), key=lambda k: entry[k]):
        if previous is not None:
            days_prev[i] = entry[i] - previous
        if cats[i] in last_by_cat:
            days_prev_cat[i] = entry[i] - last_by_cat[cats[i]]
        previous = entry[i]
        last_by_cat[cats[i]] = entry[i]

    fomc = sorted({_as_date(d) for d in fomc_dates})
    to_next = np.full(n, _NAN)
    since_last = np.full(n, _NAN)
    for i, day in enumerate(sessions):
        j = bisect.bisect_left(fomc, day)  # first FOMC date on or after the session
        if j < len(fomc):
            to_next[i] = (fomc[j] - day).days
        if j > 0:
            since_last[i] = (day - fomc[j - 1]).days

    weeks = [day.isocalendar()[:2] for day in sessions]
    week_counts = Counter(weeks)
    dom = np.array([day.day for day in sessions], dtype="int64")

    out["release_hour_et"] = release_hour
    out["dow"] = np.array([day.weekday() for day in sessions], dtype="int64")
    out["dom"] = dom
    out["week_of_month"] = (dom - 1) // 7 + 1
    out["month"] = np.array([day.month for day in sessions], dtype="int64")
    out["is_open_entry"] = np.array([1 if kind == "open" else 0 for kind in kinds], dtype="int64")
    out["days_since_prev_slot"] = days_prev
    out["days_since_same_category"] = days_prev_cat
    out["days_to_next_fomc"] = to_next
    out["days_since_last_fomc"] = since_last
    out["slots_same_week"] = np.array([week_counts[week] for week in weeks], dtype="int64")
    return out


# --- Surprise index ----------------------------------------------------------


def surprise_index(
    slots: pd.DataFrame,
    halflife_days: float = 30.0,
    categories: Sequence[str] = ("inflation", "labor", "growth", "fed"),
) -> pd.DataFrame:
    """Add exponentially weighted surprise indexes built from earlier slots (design.md section 6).

    For each slot, the earlier slots (strictly earlier by entry_utc) with a non-NaN
    surprise_z are averaged with weight 0.5 ** (age_days / halflife_days). The age runs from
    the earlier slot's last release to this slot's entry. `si_all` uses every such slot.
    `si_<category>` uses only those whose primary_category matches. A value is NaN when no
    slot qualifies. The result keeps the input row order.
    """
    _require_columns(
        slots, ("entry_utc", "last_release_utc", "surprise_z", "primary_category"), "slots"
    )
    if not halflife_days > 0:
        raise ValueError(f"halflife_days must be positive: {halflife_days!r}")
    out = slots.copy()
    n = len(out)
    entry = _days(out["entry_utc"])
    released = _days(out["last_release_utc"])
    z = out["surprise_z"].to_numpy(dtype=float, na_value=np.nan)
    cats = np.array(out["primary_category"].tolist(), dtype=object)
    usable = ~np.isnan(z)

    results = {"si_all": np.full(n, _NAN)}
    for name in categories:
        results[f"si_{name}"] = np.full(n, _NAN)
    for i in range(n):
        earlier = np.flatnonzero(usable & (entry < entry[i]))
        if earlier.size == 0:
            continue
        weights = 0.5 ** ((entry[i] - released[earlier]) / halflife_days)
        values = z[earlier]
        results["si_all"][i] = _weighted_mean(weights, values)
        for name in categories:
            same = cats[earlier] == name
            results[f"si_{name}"][i] = _weighted_mean(weights[same], values[same])

    for name, column in results.items():
        out[name] = column
    return out
