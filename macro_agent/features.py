"""Event preparation, slots, time, surprise, news and market features, and the model panel.

Specification: docs/design.md sections 4 and 6. Every feature uses only information known at
a slot's entry time. The panel joins these features with the trade labels from strategies.py.
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


# --- News features -----------------------------------------------------------

NEWS_INPUT_COLUMNS = ("published_utc", "theme", "sentiment", "macro_tone")
_NAT_NS = int(np.iinfo(np.int64).min)  # the int64 value that DatetimeIndex.asi8 gives NaT
_DAY_NS = 86_400 * 10**9


def _utc_ns(stamps: pd.Series) -> np.ndarray:
    """UTC instants as int64 nanoseconds. Naive values are read as UTC. NaT is `_NAT_NS`."""
    return pd.DatetimeIndex(pd.to_datetime(stamps, utc=True)).as_unit("ns").asi8


def _mean_of(values: np.ndarray) -> float:
    """Mean of the values that are not NaN. NaN when there are none."""
    present = values[~np.isnan(values)]
    return float(present.mean()) if present.size else _NAN


def _news_arrays(news: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Articles that have a publication time, sorted by it: (ns times, sentiment, tone, theme)."""
    if news.empty:
        return np.empty(0, dtype="int64"), np.empty(0), np.empty(0), np.empty(0, dtype=object)
    _require_columns(news, NEWS_INPUT_COLUMNS, "news")
    published = _utc_ns(news["published_utc"])
    kept = np.flatnonzero(published != _NAT_NS)
    order = kept[np.argsort(published[kept], kind="stable")]

    def numbers(column: str) -> np.ndarray:
        values = pd.to_numeric(news[column], errors="coerce")
        return values.to_numpy(dtype=float, na_value=np.nan)[order]

    themes = news["theme"].to_numpy(dtype=object)[order]
    return published[order], numbers("sentiment"), numbers("macro_tone"), themes


def news_features(
    slots: pd.DataFrame,
    news: pd.DataFrame,
    themes: Sequence[str] = ("fed", "inflation", "labor", "growth"),
) -> pd.DataFrame:
    """Add article count, sentiment and tone features to slots (design.md section 6).

    An article counts for a slot when entry_utc - W <= published_utc < entry_utc, where W is 7 or
    30 days. Adds news_n_7d (the article count), the means news_sent_7d, news_tone_7d,
    news_sent_30d and news_tone_30d, the 7-day minus 30-day changes news_sent_chg and
    news_tone_chg, and news_tone_7d_<theme> for each theme. Means skip NaN values and are NaN
    when none are left. Articles without a publication time are ignored. Returns a copy of
    slots in input order. Raises ValueError for a missing column or a missing entry_utc.
    """
    _require_columns(slots, ("entry_utc",), "slots")
    themes = tuple(dict.fromkeys(themes))
    out = slots.copy()
    entry = _utc_ns(out["entry_utc"])
    if (entry == _NAT_NS).any():
        raise ValueError("slots has missing entry_utc values")
    published, sentiment, tone, theme_of = _news_arrays(news)
    tone_of = {
        theme: np.where(pd.Series(theme_of).eq(theme).to_numpy(dtype=bool), tone, _NAN)
        for theme in themes
    }
    n = len(out)
    results: dict[str, np.ndarray] = {
        "news_n_7d": np.zeros(n, dtype="int64"),
        "news_sent_7d": np.full(n, _NAN),
        "news_tone_7d": np.full(n, _NAN),
        "news_sent_30d": np.full(n, _NAN),
        "news_tone_30d": np.full(n, _NAN),
        **{f"news_tone_7d_{theme}": np.full(n, _NAN) for theme in themes},
    }
    for i, stamp in enumerate(entry):
        end = np.searchsorted(published, stamp, side="left")  # articles before `end` predate entry
        start_7d = np.searchsorted(published, stamp - 7 * _DAY_NS, side="left")
        start_30d = np.searchsorted(published, stamp - 30 * _DAY_NS, side="left")
        results["news_n_7d"][i] = end - start_7d
        results["news_sent_7d"][i] = _mean_of(sentiment[start_7d:end])
        results["news_tone_7d"][i] = _mean_of(tone[start_7d:end])
        results["news_sent_30d"][i] = _mean_of(sentiment[start_30d:end])
        results["news_tone_30d"][i] = _mean_of(tone[start_30d:end])
        for theme in themes:
            results[f"news_tone_7d_{theme}"][i] = _mean_of(tone_of[theme][start_7d:end])
    results["news_sent_chg"] = results["news_sent_7d"] - results["news_sent_30d"]
    results["news_tone_chg"] = results["news_tone_7d"] - results["news_tone_30d"]
    names = [
        "news_n_7d", "news_sent_7d", "news_tone_7d", "news_sent_30d", "news_tone_30d",
        "news_sent_chg", "news_tone_chg", *(f"news_tone_7d_{theme}" for theme in themes),
    ]
    for name in names:
        out[name] = results[name]
    return out


# --- Market features ---------------------------------------------------------

MARKET_COLUMNS = (
    "slot_id", "asset", "prev_close", "ret_5d", "ret_20d", "vol_20d", "reaction", "vix_prev",
)
_TRADING_DAYS = 252
_ENTRY_KINDS = ("open", "close")


def _price_book(frame: pd.DataFrame) -> pd.DataFrame:
    """Float copy of a wide price frame, indexed by sorted dates.

    Missing, zero, negative and infinite prices become NaN, so they never form a return.
    """
    book = frame.astype("float64")
    book.index = [_as_date(value) for value in frame.index]
    book = book.sort_index()
    return book.where(np.isfinite(book) & (book > 0))


def _column(book: pd.DataFrame, name: str | None) -> np.ndarray:
    """One price column as floats. All NaN when `name` is None or not a column of `book`."""
    if name is None or name not in book.columns:
        return np.full(len(book.index), _NAN)
    return book[name].to_numpy(dtype=float)


def _value_at(book: pd.DataFrame, day: date, name: str) -> float:
    """The value in column `name` on `day`, NaN when either is missing."""
    if name not in book.columns or day not in book.index:
        return _NAN
    return float(book.at[day, name])


def _annual_volatility(closes: np.ndarray) -> float:
    """Annualised sample standard deviation of the log returns between consecutive closes.

    Twenty-one closes give twenty returns. NaN when any close is missing.
    """
    if np.isnan(closes).any():
        return _NAN
    returns = np.diff(np.log(closes))
    return float(returns.std(ddof=1) * math.sqrt(_TRADING_DAYS))


def market_features(
    slots: pd.DataFrame,
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    assets: Sequence[str],
    vix_symbol: str | None = "^VIX",
) -> pd.DataFrame:
    """Market-state features in long format, one row per (slot, asset) (design.md section 6).

    opens and closes are wide frames indexed by date, as from pivot_prices. For a slot whose
    entry session is E, p is the last session in closes before E. prev_close is the close on p.
    ret_5d and ret_20d are the returns over 5 and 20 sessions up to p. vol_20d is the annualised
    standard deviation of the 20 daily log returns ending at p. reaction is the entry price over
    prev_close, minus 1: the open on E for "open" entries and the close on E for "close" entries.
    vix_prev is the VIX close on p, NaN when vix_symbol is None or not a column. Too little
    history or a missing price gives NaN. Raises ValueError for missing slot columns or an
    entry_kind other than "open" or "close".
    """
    _require_columns(slots, ("slot_id", "entry_session", "entry_kind"), "slots")
    kinds = {str(kind) for kind in slots["entry_kind"]}
    if not kinds <= set(_ENTRY_KINDS):
        raise ValueError(f"entry_kind must be 'open' or 'close': {sorted(kinds)}")
    asset_list = tuple(dict.fromkeys(assets))
    open_book, close_book = _price_book(opens), _price_book(closes)
    sessions = list(close_book.index)
    series = {asset: _column(close_book, asset) for asset in asset_list}
    vix = _column(close_book, vix_symbol)
    rows: list[dict[str, object]] = []
    for slot_id, session, kind in zip(
        slots["slot_id"], slots["entry_session"], slots["entry_kind"], strict=True
    ):
        day = _as_date(session)
        p = bisect.bisect_left(sessions, day) - 1  # position of p, or -1 when there is none
        book = open_book if kind == "open" else close_book
        for asset in asset_list:
            prices = series[asset]
            if p >= 0:
                prev_close = prices[p]
                ret_5d = prev_close / prices[p - 5] - 1 if p >= 5 else _NAN
                ret_20d = prev_close / prices[p - 20] - 1 if p >= 20 else _NAN
                vol_20d = _annual_volatility(prices[p - 20 : p + 1]) if p >= 20 else _NAN
                vix_prev = vix[p]
            else:
                prev_close = ret_5d = ret_20d = vol_20d = vix_prev = _NAN
            reaction = _value_at(book, day, asset) / prev_close - 1
            rows.append({
                "slot_id": slot_id,
                "asset": asset,
                "prev_close": float(prev_close),
                "ret_5d": float(ret_5d),
                "ret_20d": float(ret_20d),
                "vol_20d": float(vol_20d),
                "reaction": float(reaction),
                "vix_prev": float(vix_prev),
            })
    out = pd.DataFrame(rows, columns=list(MARKET_COLUMNS))
    return out.astype({name: "float64" for name in MARKET_COLUMNS[2:]})


# --- Model panel -------------------------------------------------------------

CATEGORICAL_FEATURES = ("asset", "rule")
MODEL_FEATURES = (
    "horizon", "sign",
    "release_hour_et", "dow", "dom", "week_of_month", "is_open_entry",
    "days_since_prev_slot", "days_since_same_category", "days_to_next_fomc",
    "days_since_last_fomc", "slots_same_week",
    "n_events", "n_high", "surprise_z", "abs_surprise_z",
    "cat_fed", "cat_inflation", "cat_labor", "cat_growth", "cat_consumer", "cat_housing",
    "cat_other",
    "si_all", "si_inflation", "si_labor", "si_growth", "si_fed",
    "news_n_7d", "news_sent_7d", "news_tone_7d", "news_sent_30d", "news_tone_30d",
    "news_sent_chg", "news_tone_chg",
    "news_tone_7d_fed", "news_tone_7d_inflation", "news_tone_7d_labor", "news_tone_7d_growth",
    "ret_5d", "ret_20d", "vol_20d", "reaction", "vix_prev",
    "signed_reaction", "signed_surprise", "signed_ret_5d",
)


def build_panel(
    slots_feat: pd.DataFrame, labels: pd.DataFrame, market: pd.DataFrame
) -> pd.DataFrame:
    """One row per (slot, strategy): labels joined with slot and market features.

    Slot columns that labels already have, except slot_id, are dropped first, so the labels
    keep their own entry_* columns. Market features join on (slot_id, asset), and a market
    column that the panel already has is not repeated. Adds signed_reaction, signed_surprise
    (a NaN surprise counts as 0) and signed_ret_5d, each multiplied by the position sign. Rows
    are sorted by entry_utc, then strategy_id, and indexed 0 to n-1. Raises ValueError for
    missing columns, or for repeated slot_id or (slot_id, asset) keys.
    """
    _require_columns(labels, ("slot_id", "strategy_id", "asset", "sign", "entry_utc"), "labels")
    _require_columns(slots_feat, ("slot_id", "surprise_z"), "slots_feat")
    _require_columns(market, ("slot_id", "asset", "reaction", "ret_5d"), "market")
    if slots_feat["slot_id"].duplicated().any():
        raise ValueError("slots_feat has repeated slot_id values")
    if market.duplicated(["slot_id", "asset"]).any():
        raise ValueError("market has repeated (slot_id, asset) rows")
    slot_cols = [c for c in slots_feat.columns if c == "slot_id" or c not in labels.columns]
    panel = labels.merge(slots_feat.loc[:, slot_cols], on="slot_id", how="inner")
    market_cols = [c for c in market.columns if c in ("slot_id", "asset") or c not in panel.columns]
    panel = panel.merge(market.loc[:, market_cols], on=["slot_id", "asset"], how="left")
    sign = panel["sign"]
    panel["signed_reaction"] = sign * panel["reaction"]
    panel["signed_surprise"] = sign * panel["surprise_z"].fillna(0.0)
    panel["signed_ret_5d"] = sign * panel["ret_5d"]
    return panel.sort_values(["entry_utc", "strategy_id", "slot_id"], ignore_index=True)


def feature_columns(panel: pd.DataFrame) -> list[str]:
    """The MODEL_FEATURES that `panel` has, in MODEL_FEATURES order. Labels are never listed."""
    return [name for name in MODEL_FEATURES if name in panel.columns]
