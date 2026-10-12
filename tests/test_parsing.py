"""Tests for macro_agent.parsing (design.md section 4)."""

from __future__ import annotations

import datetime as dt
import hashlib

import numpy as np
import pandas as pd
import pytest

from macro_agent.parsing import (
    categorize_event,
    conservative_published_utc,
    impact_weight,
    is_inverted,
    make_event_id,
    parse_value,
)

MINUS = "\u2212"
EN_DASH = "\u2013"
EM_DASH = "\u2014"
APPROX = "\u2248"


# --- parse_value ------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("3.2%", (3.2, "%")),
        ("-0.1%", (-0.1, "%")),
        ("256K", (256000.0, "K")),
        ("1.25M", (1250000.0, "M")),
        (f"{MINUS}12.5B", (-1.25e10, "B")),
        ("<0.1%", (0.1, "%")),
        ("1,234.5", (1234.5, None)),
        ("52.3", (52.3, None)),
        ("abc", (None, None)),
    ],
)
def test_parse_value_spec_examples(raw, expected):
    assert parse_value(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [None, float("nan"), np.nan, pd.NA, "", "-", EM_DASH, "N/A", "n/a", "tentative",
     "  ", "abc", "inf", "nan", float("inf")],
)
def test_parse_value_missing_or_unparseable(raw):
    assert parse_value(raw) == (None, None)


def test_parse_value_numbers_have_no_unit():
    assert parse_value(5) == (5.0, None)
    assert parse_value(-0.25) == (-0.25, None)
    assert parse_value(np.int64(3)) == (3.0, None)
    assert parse_value(np.float64(1.5)) == (1.5, None)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (" 3.2% ", (3.2, "%")),
        ("+3.2%", (3.2, "%")),
        ("~2", (2.0, None)),
        (f"{APPROX} 1.5K", (1500.0, "K")),
        ("> -0.4%", (-0.4, "%")),
        (f"{EN_DASH}0.3%", (-0.3, "%")),
        ("256k", (256000.0, "K")),
        ("1.25m", (1250000.0, "M")),
        ("2b", (2e9, "B")),
        ("1.5t", (1.5e12, "T")),
        ("3.2 %", (3.2, "%")),
        ("-2.5 M", (-2500000.0, "M")),
        ("1.005K", (1005.0, "K")),
    ],
)
def test_parse_value_markers_dashes_case_and_exact_scaling(raw, expected):
    assert parse_value(raw) == expected


@pytest.mark.parametrize(
    "raw", ["%", "K", "1.2.3", "12%K", "1 234", "--1", "5x", "1e", f"{EM_DASH}%"]
)
def test_parse_value_rejects_malformed(raw):
    assert parse_value(raw) == (None, None)


# --- categorize_event -------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Philly Fed Manufacturing Index", "growth"),
        ("Fed Chair Powell Speaks", "fed"),
        ("Federal Funds Rate", "fed"),
        ("CPI m/m", "inflation"),
        ("Core PCE Price Index m/m", "inflation"),
        ("Non-Farm Employment Change", "labor"),
        ("Unemployment Claims", "labor"),
        ("Prelim UoM Consumer Sentiment", "consumer"),
        ("Prelim UoM Inflation Expectations", "inflation"),
        ("ISM Services PMI", "growth"),
        ("Crude Oil Inventories", "other"),
    ],
)
def test_categorize_event_spec_examples(name, expected):
    assert categorize_event(name) == expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("FOMC Statement", "fed"),
        ("Beige Book", "fed"),
        ("Import Prices m/m", "inflation"),
        ("Average Hourly Earnings m/m", "labor"),
        ("JOLTs Job Openings", "labor"),
        ("Existing Home Sales", "housing"),
        ("Building Permits Prelim", "housing"),
        ("Personal Spending m/m", "consumer"),
        ("Consumer Credit m/m", "consumer"),
        ("Retail Sales m/m", "growth"),
        ("Durable Goods Orders", "growth"),
        ("Empire State Manufacturing Index", "growth"),
    ],
)
def test_categorize_event_other_keywords(name, expected):
    assert categorize_event(name) == expected


def test_categorize_event_is_case_insensitive():
    assert categorize_event("cpi M/M") == "inflation"
    assert categorize_event("FOMC MINUTES") == "fed"


def test_categorize_event_missing_name_is_other():
    assert categorize_event(None) == "other"  # type: ignore[arg-type]


# --- is_inverted ------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Unemployment Rate", True),
        ("Unemployment Claims", True),
        ("Initial Jobless Claims", True),
        ("Continuing Claims", True),
        ("Initial Claims", True),
        ("Non-Farm Employment Change", False),
        ("CPI m/m", False),
        ("ISM Manufacturing PMI", False),
    ],
)
def test_is_inverted(name, expected):
    assert is_inverted(name) is expected


# --- impact_weight ----------------------------------------------------------


@pytest.mark.parametrize(
    ("impact", "expected"),
    [
        ("High Impact Expected", 1.0),
        ("HIGH", 1.0),
        ("Medium", 0.5),
        ("Low", 0.25),
        ("Non-Economic", 0.0),
        ("holiday", 0.0),
        (3, 1.0),
        (2, 0.5),
        (1, 0.25),
        (0, 0.0),
        (7, 0.0),
        (3.0, 1.0),
        (np.int64(2), 0.5),
        (None, 0.0),
        (float("nan"), 0.0),
        (pd.NA, 0.0),
    ],
)
def test_impact_weight(impact, expected):
    assert impact_weight(impact) == expected


# --- make_event_id ----------------------------------------------------------


def test_make_event_id_is_deterministic_16_hex_chars():
    first = make_event_id("forexfactory", "USD", "CPI m/m", "2026-06-10T12:30:00Z")
    second = make_event_id("forexfactory", "USD", "CPI m/m", "2026-06-10T12:30:00Z")
    assert first == second
    assert len(first) == 16
    assert set(first) <= set("0123456789abcdef")


def test_make_event_id_matches_sha1_of_joined_fields():
    expected = hashlib.sha1(b"forexfactory|USD|CPI m/m|2026-06-10T12:30:00Z").hexdigest()[:16]
    assert make_event_id("forexfactory", "USD", "CPI m/m", "2026-06-10T12:30:00Z") == expected


def test_make_event_id_treats_none_as_empty():
    assert make_event_id("src", None, "name", None) == make_event_id("src", "", "name", "")
    assert make_event_id(None, None, None, None) == hashlib.sha1(b"|||").hexdigest()[:16]


def test_make_event_id_changes_when_any_field_changes():
    base = make_event_id("src", "USD", "CPI m/m", "2026-06-10T12:30:00Z")
    assert make_event_id("other", "USD", "CPI m/m", "2026-06-10T12:30:00Z") != base
    assert make_event_id("src", "EUR", "CPI m/m", "2026-06-10T12:30:00Z") != base
    assert make_event_id("src", "USD", "PPI m/m", "2026-06-10T12:30:00Z") != base
    assert make_event_id("src", "USD", "CPI m/m", "2026-06-10T12:31:00Z") != base


# --- conservative_published_utc --------------------------------------------


def test_date_only_string_is_end_of_day_utc():
    expected = pd.Timestamp("2026-06-10 23:59:59", tz="UTC")
    assert conservative_published_utc("2026-06-10") == expected


def test_date_object_is_end_of_day_utc():
    expected = pd.Timestamp("2026-06-10 23:59:59", tz="UTC")
    assert conservative_published_utc(dt.date(2026, 6, 10)) == expected


def test_date_only_cannot_leak_into_a_same_day_close():
    # The 16:00 ET close on 2026-06-10 (EDT, UTC-4) is 20:00 UTC.
    close = pd.Timestamp("2026-06-10 20:00", tz="UTC")
    assert conservative_published_utc("2026-06-10") > close


def test_time_without_zone_is_utc():
    expected = pd.Timestamp("2026-06-10 12:30", tz="UTC")
    assert conservative_published_utc("2026-06-10 12:30") == expected
    assert conservative_published_utc("Jun 10, 2026 12:30 PM") == expected


def test_zulu_and_offset_strings_are_converted_to_utc():
    expected = pd.Timestamp("2026-06-10 12:30", tz="UTC")
    assert conservative_published_utc("2026-06-10T12:30:00Z") == expected
    assert conservative_published_utc("2026-06-10T14:30:00+02:00") == expected


def test_aware_datetime_is_converted_to_utc():
    value = dt.datetime(2026, 6, 10, 8, 30, tzinfo=dt.timezone(dt.timedelta(hours=-4)))
    assert conservative_published_utc(value) == pd.Timestamp("2026-06-10 12:30", tz="UTC")


def test_naive_datetime_is_read_as_utc():
    value = dt.datetime(2026, 6, 10, 12, 30)
    assert conservative_published_utc(value) == pd.Timestamp("2026-06-10 12:30", tz="UTC")


def test_result_is_utc_aware():
    result = conservative_published_utc("2026-06-10T14:30:00+02:00")
    assert isinstance(result, pd.Timestamp)
    assert result.utcoffset() == dt.timedelta(0)


@pytest.mark.parametrize("raw", ["10:30", "8am", "Jun 10"])
def test_text_without_a_year_is_none(raw):
    # Pandas would otherwise invent a date (today's, or year 1) for these.
    assert conservative_published_utc(raw) is None


@pytest.mark.parametrize(
    "raw", [None, "", "   ", "not a date", "N/A", float("nan"), pd.NA, pd.NaT, 12345]
)
def test_missing_or_unparseable_is_none(raw):
    assert conservative_published_utc(raw) is None


@pytest.mark.parametrize(
    "raw",
    [
        "2026-06-10T00:00:00.000Z",  # Exa's date-only format
        "2026-06-10T00:00:00",
        dt.datetime(2026, 6, 10),
        pd.Timestamp("2026-06-10"),
        pd.Timestamp("2026-06-10", tz="UTC"),
    ],
)
def test_conservative_published_utc_treats_exact_midnight_as_date_only(raw):
    assert conservative_published_utc(raw) == pd.Timestamp("2026-06-10T23:59:59Z")


def test_conservative_published_utc_keeps_a_real_time_just_after_midnight():
    assert conservative_published_utc("2026-06-10T00:00:01Z") == pd.Timestamp(
        "2026-06-10T00:00:01Z"
    )
