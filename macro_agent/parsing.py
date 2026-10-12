"""Parsing helpers for calendar values, event metadata and news timestamps.

Rules follow docs/design.md section 4. Every function here is pure.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import math
import numbers
import re
from decimal import Decimal

import pandas as pd
from pandas.api.types import is_scalar

# --- Calendar values --------------------------------------------------------

# Unicode minus (U+2212) and en dash (U+2013) become "-"; thousands commas are dropped.
_CLEAN = str.maketrans({"\u2212": "-", "\u2013": "-", ",": None})
# Comparison and approximation markers that may precede a number, e.g. "<0.1%".
_LEADING_MARKERS = "+<>~\u2248"
# Placeholders that carry no value (\u2014 is the em dash).
_PLACEHOLDERS = frozenset({"", "-", "\u2014", "N/A", "n/a", "tentative"})
_NUMBER_RE = re.compile(r"[+-]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_MULTIPLIERS = {"K": 10**3, "M": 10**6, "B": 10**9, "T": 10**12}


def parse_value(raw: object) -> tuple[float | None, str | None]:
    """Parse a calendar value into ``(value, unit)``.

    ``%`` values keep the printed number. K/M/B/T values are multiplied out and
    the letter becomes the unit. Missing or unparseable input gives ``(None, None)``.
    Infinite numbers count as missing.
    """
    if isinstance(raw, numbers.Real):
        number = float(raw)
        return (number, None) if math.isfinite(number) else (None, None)
    if not isinstance(raw, str):
        return None, None

    text = raw.translate(_CLEAN).strip().lstrip(_LEADING_MARKERS).strip()
    if text in _PLACEHOLDERS:
        return None, None

    unit: str | None = None
    multiplier = 1
    if text.endswith("%"):
        unit, text = "%", text[:-1].rstrip()
    elif text[-1:].upper() in _MULTIPLIERS:
        unit = text[-1:].upper()
        multiplier = _MULTIPLIERS[unit]
        text = text[:-1].rstrip()
    if not _NUMBER_RE.fullmatch(text):
        return None, None
    try:
        # Decimal keeps the product exact: "1.005K" gives 1005.0, not 1004.9999999999999.
        value = float(Decimal(text) * multiplier)
    except ArithmeticError:
        return None, None
    return (value, unit) if math.isfinite(value) else (None, None)


# --- Event metadata ---------------------------------------------------------

# Categories in match order: the first category with a matching keyword wins.
_CATEGORY_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "fed",
        ("fomc", "federal funds", "fed chair", "beige book", "monetary policy",
         "interest rate decision"),
    ),
    ("inflation", ("cpi", "ppi", "pce", "price index", "inflation", "prices")),
    (
        "labor",
        ("non-farm", "nonfarm", "payroll", "employment", "unemployment", "jobless",
         "claims", "jolts", "job openings", "hourly earnings", "labor", "adp"),
    ),
    (
        "housing",
        ("housing", "building permits", "home sales", "pending home", "hpi", "construction"),
    ),
    (
        "consumer",
        ("consumer sentiment", "consumer confidence", "uom", "personal spending",
         "personal income", "consumer credit"),
    ),
    (
        "growth",
        ("gdp", "ism", "pmi", "retail sales", "industrial production", "durable goods",
         "factory orders", "manufacturing", "empire state", "business inventories",
         "productivity"),
    ),
)

# Indicators where a higher reading means a weaker economy.
_INVERTED_PHRASES = (
    "unemployment rate",
    "unemployment claims",
    "jobless claims",
    "continuing claims",
    "initial claims",
)

# Numeric impact ranks: 3 = high, 2 = medium, 1 = low.
_NUMERIC_IMPACT = {3: 1.0, 2: 0.5, 1: 0.25}


def categorize_event(name: str) -> str:
    """Return the event category (fed, inflation, labor, housing, consumer, growth, other)."""
    text = name.lower() if isinstance(name, str) else ""
    for category, keywords in _CATEGORY_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return category
    return "other"


def is_inverted(name: str) -> bool:
    """True when a higher reading is weaker (unemployment rate, jobless or other claims)."""
    text = name.lower() if isinstance(name, str) else ""
    return any(phrase in text for phrase in _INVERTED_PHRASES)


def impact_weight(impact: object) -> float:
    """Weight an impact label or rank: high 1.0, medium 0.5, low 0.25, otherwise 0.0."""
    if isinstance(impact, str):
        text = impact.lower()
        if "high" in text:
            return 1.0
        if "med" in text:
            return 0.5
        if "low" in text:
            return 0.25
        return 0.0
    if isinstance(impact, numbers.Real):
        # NaN is never a key, so it falls through to 0.0.
        return _NUMERIC_IMPACT.get(impact, 0.0)
    return 0.0


def make_event_id(
    source: str | None, currency: str | None, name: str | None, release_utc_iso: str | None
) -> str:
    """Return a deterministic 16-character hex id; missing parts count as empty strings."""
    parts = [_as_text(part) for part in (source, currency, name, release_utc_iso)]
    digest = hashlib.sha1("|".join(parts).encode("utf-8"), usedforsecurity=False)
    return digest.hexdigest()[:16]


def _as_text(value: object) -> str:
    """Return str(value), with None, NaN and NA as an empty string."""
    if value is None or (is_scalar(value) and pd.isna(value)):
        return ""
    return str(value)


# --- News timestamps --------------------------------------------------------

# A clock time such as "12:30" or "8am". A string without one is date-only.
_CLOCK_RE = re.compile(r"\d{1,2}:\d{2}|\d{1,2}\s*[ap]\.?m\b", re.IGNORECASE)
# A four-digit year. Without one, pandas would invent a date, so such text is rejected.
_YEAR_RE = re.compile(r"(?:19|20|21)\d{2}")


def conservative_published_utc(raw: object) -> pd.Timestamp | None:
    """Parse a news timestamp to a UTC Timestamp, erring late for date-only values.

    A date-only value (a string with no clock time, or a ``datetime.date``) becomes
    23:59:59 UTC on that date, so it never counts as published before a same-day
    entry. An exact midnight is treated the same way, because sources such as Exa
    stamp date-only articles "T00:00:00Z". A time without a zone is read as UTC;
    aware values are converted to UTC.
    Strings without a four-digit year (time-only text, for example) have no reliable
    date and give ``None``, as do missing or unparseable values.
    """
    if isinstance(raw, dt.date) and not isinstance(raw, dt.datetime):
        return _end_of_day_utc(raw.year, raw.month, raw.day)
    if isinstance(raw, str):
        if not _YEAR_RE.search(raw):
            return None
        stamp = _to_timestamp(raw)
        if stamp is None:
            return None
        if not _CLOCK_RE.search(raw):
            return _end_of_day_utc(stamp.year, stamp.month, stamp.day)
    elif isinstance(raw, dt.datetime):
        stamp = _to_timestamp(raw)
        if stamp is None:
            return None
    else:
        return None
    if stamp == stamp.normalize():
        # An exact midnight usually means the source only knew the date (Exa returns
        # "...T00:00:00.000Z"), so treat it as date-only.
        return _end_of_day_utc(stamp.year, stamp.month, stamp.day)
    try:
        return stamp.tz_localize("UTC") if stamp.tz is None else stamp.tz_convert("UTC")
    except (ValueError, OverflowError):
        return None


def _to_timestamp(value: str | dt.datetime) -> pd.Timestamp | None:
    """Parse a string or datetime to a Timestamp; None when empty, NaT or invalid."""
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return None if pd.isna(stamp) else stamp


def _end_of_day_utc(year: int, month: int, day: int) -> pd.Timestamp | None:
    """Return 23:59:59 UTC on the given calendar day, or None if it is out of range."""
    try:
        return pd.Timestamp(
            year=year, month=month, day=day, hour=23, minute=59, second=59, tz="UTC"
        )
    except (ValueError, OverflowError):
        return None
