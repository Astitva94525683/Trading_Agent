"""Tests for macro_agent.timeutils (docs/design.md section 3)."""
from __future__ import annotations

import dataclasses
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from macro_agent.config import ET
from macro_agent.timeutils import Entry, SessionCalendar, iso_utc, to_et, to_utc

# Mon 8 June to Wed 17 June 2026: eight sessions, no holidays.
JUNE_WEEK = [date(2026, 6, d) for d in (8, 9, 10, 11, 12, 15, 16, 17)]
CAL = SessionCalendar.from_dates(JUNE_WEEK)
OPEN_ENTRY = CAL.entry_for("2026-06-08 08:30")  # pre-market release on Mon 8 June
CLOSE_ENTRY = CAL.entry_for("2026-06-08 14:00")  # in-session release on Mon 8 June

DAYS = [date(2026, 6, 8), date(2026, 6, 9)]
OPENS = ["2026-06-08T13:30:00Z", "2026-06-09T13:30:00Z"]
CLOSES = ["2026-06-08T20:00:00Z", "2026-06-09T20:00:00Z"]


def _frame(days, opens, closes):
    return pd.DataFrame(
        {
            "date": days,
            "open_utc": [pd.Timestamp(x) for x in opens],
            "close_utc": [pd.Timestamp(x) for x in closes],
        }
    )


# --- to_utc, to_et, iso_utc ---


def test_naive_string_is_read_as_new_york_time():
    assert to_utc("2026-06-10 08:30") == pd.Timestamp("2026-06-10T12:30:00Z")  # EDT
    assert to_utc("2026-02-10 08:30") == pd.Timestamp("2026-02-10T13:30:00Z")  # EST


@pytest.mark.parametrize(
    "local, expected",
    [
        ("2026-03-06 08:30", "2026-03-06T13:30:00Z"),  # EST, Friday before DST starts
        ("2026-03-09 08:30", "2026-03-09T12:30:00Z"),  # EDT, Monday after DST starts
        ("2026-11-02 08:30", "2026-11-02T13:30:00Z"),  # EST again after 1 November
    ],
)
def test_release_times_across_dst_changes(local, expected):
    assert to_utc(local) == pd.Timestamp(expected)


def test_naive_datetime_and_timestamp_are_read_as_new_york_time():
    expected = pd.Timestamp("2026-06-10T12:30:00Z")
    assert to_utc(datetime(2026, 6, 10, 8, 30)) == expected
    assert to_utc(pd.Timestamp("2026-06-10 08:30")) == expected


def test_assume_tz_replaces_new_york_for_naive_values():
    assert to_utc("2026-06-10 08:30", "Europe/London") == pd.Timestamp("2026-06-10T07:30:00Z")


def test_aware_string_is_converted_to_utc_and_ignores_assume_tz():
    assert to_utc("2026-06-10T12:30:00Z") == pd.Timestamp("2026-06-10T12:30:00Z")
    assert to_utc("2026-06-10T08:30:00-04:00") == pd.Timestamp("2026-06-10T12:30:00Z")
    assert to_utc("2026-06-10T12:30:00Z", "Asia/Tokyo") == pd.Timestamp("2026-06-10T12:30:00Z")


def test_aware_values_in_other_zones_are_converted_to_utc():
    tokyo = datetime(2026, 6, 10, 21, 30, tzinfo=ZoneInfo("Asia/Tokyo"))
    assert to_utc(tokyo) == pd.Timestamp("2026-06-10T12:30:00Z")
    ny = pd.Timestamp("2026-06-10 08:30", tz="America/New_York")
    assert to_utc(ny) == pd.Timestamp("2026-06-10T12:30:00Z")


def test_to_utc_result_has_zero_offset():
    assert to_utc("2026-06-10 08:30").utcoffset() == timedelta(0)


def test_to_et_converts_to_new_york_wall_time():
    summer = to_et("2026-06-10T12:30:00Z")
    assert (summer.hour, summer.minute) == (8, 30)
    assert str(summer.tz) == "America/New_York"
    assert summer.utcoffset() == timedelta(hours=-4)
    assert to_et("2026-01-15 08:30").utcoffset() == timedelta(hours=-5)


def test_iso_utc_format_to_the_second():
    assert iso_utc("2026-06-10 08:30") == "2026-06-10T12:30:00Z"
    assert iso_utc(datetime(2026, 6, 10, 8, 30, tzinfo=ET)) == "2026-06-10T12:30:00Z"
    assert iso_utc("2026-06-10T12:30:00.750Z") == "2026-06-10T12:30:00Z"


@pytest.mark.parametrize("bad", ["not a date", "", None, 12345, "2026-13-45 08:30"])
def test_unparseable_input_raises(bad):
    with pytest.raises(ValueError):
        to_utc(bad)


def test_nonexistent_local_time_raises():
    with pytest.raises(ValueError):  # clocks jump from 02:00 to 03:00 on 8 March 2026
        to_utc("2026-03-08 02:30")


def test_ambiguous_local_time_raises():
    with pytest.raises(ValueError):  # 01:00 to 02:00 happens twice on 1 November 2026
        to_utc("2026-11-01 01:30")


# --- SessionCalendar construction ---


def test_from_dates_sorts_and_drops_duplicates():
    cal = SessionCalendar.from_dates(
        [date(2026, 6, 10), "2026-06-08", pd.Timestamp("2026-06-09"), "2026-06-08"]
    )
    assert cal.dates == [date(2026, 6, 8), date(2026, 6, 9), date(2026, 6, 10)]
    assert len(cal) == 3
    assert date(2026, 6, 9) in cal
    assert date(2026, 6, 13) not in cal


def test_default_session_is_0930_to_1600_et_and_follows_dst():
    assert CAL.open_at(date(2026, 6, 10)) == pd.Timestamp("2026-06-10T13:30:00Z")
    assert CAL.close_at(date(2026, 6, 10)) == pd.Timestamp("2026-06-10T20:00:00Z")
    winter = SessionCalendar.from_dates([date(2026, 2, 10)])
    assert winter.open_at(date(2026, 2, 10)) == pd.Timestamp("2026-02-10T14:30:00Z")
    assert winter.close_at(date(2026, 2, 10)) == pd.Timestamp("2026-02-10T21:00:00Z")


def test_open_and_close_follow_dst_across_the_change():
    cal = SessionCalendar.from_dates([date(2026, 3, 6), date(2026, 3, 9), date(2026, 11, 2)])
    assert cal.open_at(date(2026, 3, 6)) == pd.Timestamp("2026-03-06T14:30:00Z")
    assert cal.close_at(date(2026, 3, 6)) == pd.Timestamp("2026-03-06T21:00:00Z")
    assert cal.open_at(date(2026, 3, 9)) == pd.Timestamp("2026-03-09T13:30:00Z")
    assert cal.close_at(date(2026, 3, 9)) == pd.Timestamp("2026-03-09T20:00:00Z")
    assert cal.open_at(date(2026, 11, 2)) == pd.Timestamp("2026-11-02T14:30:00Z")


def test_custom_session_times_are_built_in_et():
    cal = SessionCalendar.from_dates([date(2026, 6, 10)], open_et=time(8, 0), close_et=time(12, 0))
    assert cal.open_at(date(2026, 6, 10)) == pd.Timestamp("2026-06-10T12:00:00Z")
    assert cal.close_at(date(2026, 6, 10)) == pd.Timestamp("2026-06-10T16:00:00Z")


def test_constructor_accepts_frame_and_stores_utc():
    cal = SessionCalendar(_frame(DAYS, ["2026-06-08T09:30:00-04:00", OPENS[1]], CLOSES))
    assert cal.dates == DAYS
    assert cal.open_at(date(2026, 6, 8)) == pd.Timestamp("2026-06-08T13:30:00Z")
    assert cal.open_at(date(2026, 6, 8)).utcoffset() == timedelta(0)


def test_constructor_rejects_unsorted_dates():
    with pytest.raises(ValueError):
        SessionCalendar(_frame(DAYS[::-1], OPENS[::-1], CLOSES[::-1]))


def test_constructor_rejects_duplicate_dates():
    with pytest.raises(ValueError):
        SessionCalendar(_frame([DAYS[0], DAYS[0]], OPENS, CLOSES))


def test_constructor_rejects_open_not_before_close():
    with pytest.raises(ValueError):
        SessionCalendar(_frame(DAYS, OPENS, ["2026-06-08T12:00:00Z", CLOSES[1]]))


def test_constructor_rejects_missing_column():
    with pytest.raises(ValueError):
        SessionCalendar(_frame(DAYS, OPENS, CLOSES).drop(columns=["close_utc"]))


def test_constructor_rejects_naive_times():
    with pytest.raises(ValueError):
        SessionCalendar(_frame(DAYS, ["2026-06-08 09:30", OPENS[1]], CLOSES))


def test_membership_length_and_dates():
    assert len(CAL) == 8
    assert CAL.dates == JUNE_WEEK
    assert date(2026, 6, 10) in CAL
    assert date(2026, 6, 13) not in CAL  # Saturday
    assert None not in CAL


def test_next_session_inclusive_and_exclusive():
    assert CAL.next_session(date(2026, 6, 13)) == date(2026, 6, 15)  # Saturday to Monday
    assert CAL.next_session(date(2026, 6, 15)) == date(2026, 6, 15)
    assert CAL.next_session(date(2026, 6, 15), inclusive=False) == date(2026, 6, 16)
    assert CAL.next_session(date(2026, 6, 17), inclusive=False) is None
    assert CAL.next_session(date(2026, 6, 18)) is None


def test_offset_counts_sessions_not_calendar_days():
    assert CAL.offset(date(2026, 6, 12), 0) == date(2026, 6, 12)
    assert CAL.offset(date(2026, 6, 12), 2) == date(2026, 6, 16)  # skips the weekend
    assert CAL.offset(date(2026, 6, 8), 7) == date(2026, 6, 17)
    assert CAL.offset(date(2026, 6, 8), 8) is None


def test_open_and_close_of_non_session_raise_key_error():
    with pytest.raises(KeyError):
        CAL.open_at(date(2026, 6, 13))
    with pytest.raises(KeyError):
        CAL.close_at(date(2026, 6, 13))


# --- entry_for ---


def test_open_entry_on_edt_session():
    cal = SessionCalendar.from_dates([date(2026, 6, 10)])
    assert cal.entry_for("2026-06-10 08:30") == Entry(
        date(2026, 6, 10), "open", pd.Timestamp("2026-06-10T13:30:00Z")
    )


def test_open_entry_on_est_session():
    cal = SessionCalendar.from_dates([date(2026, 2, 10)])
    assert cal.entry_for("2026-02-10 08:30") == Entry(
        date(2026, 2, 10), "open", pd.Timestamp("2026-02-10T14:30:00Z")
    )


def test_fomc_style_release_inside_session_enters_at_close():
    assert CAL.entry_for("2026-06-17 14:00") == Entry(
        date(2026, 6, 17), "close", pd.Timestamp("2026-06-17T20:00:00Z")
    )


def test_release_exactly_at_open_is_a_close_entry():
    assert CAL.entry_for("2026-06-10 09:30") == Entry(
        date(2026, 6, 10), "close", pd.Timestamp("2026-06-10T20:00:00Z")
    )


def test_release_exactly_at_close_rolls_to_next_open():
    assert CAL.entry_for("2026-06-10 16:00") == Entry(
        date(2026, 6, 11), "open", pd.Timestamp("2026-06-11T13:30:00Z")
    )


def test_release_after_close_rolls_to_next_open():
    assert CAL.entry_for("2026-06-10 17:45") == Entry(
        date(2026, 6, 11), "open", pd.Timestamp("2026-06-11T13:30:00Z")
    )


def test_weekend_release_rolls_to_monday_open():
    assert CAL.entry_for("2026-06-13 08:30") == Entry(
        date(2026, 6, 15), "open", pd.Timestamp("2026-06-15T13:30:00Z")
    )


def test_holiday_release_rolls_to_next_session_open():
    # 3 July 2026 is a market holiday (Independence Day observed), so it is not a session.
    cal = SessionCalendar.from_dates([date(2026, 7, 2), date(2026, 7, 6)])
    assert cal.entry_for("2026-07-03 08:30") == Entry(
        date(2026, 7, 6), "open", pd.Timestamp("2026-07-06T13:30:00Z")
    )


def test_release_across_dst_gives_dst_correct_entry():
    cal = SessionCalendar.from_dates([date(2026, 3, 6), date(2026, 3, 9)])
    assert cal.entry_for("2026-03-06 08:30") == Entry(
        date(2026, 3, 6), "open", pd.Timestamp("2026-03-06T14:30:00Z")
    )
    assert cal.entry_for("2026-03-09 08:30") == Entry(
        date(2026, 3, 9), "open", pd.Timestamp("2026-03-09T13:30:00Z")
    )


def test_entry_accepts_aware_release_in_another_zone():
    assert CAL.entry_for("2026-06-10T12:30:00Z") == Entry(
        date(2026, 6, 10), "open", pd.Timestamp("2026-06-10T13:30:00Z")
    )


def test_early_close_turns_afternoon_release_into_next_open():
    cal = SessionCalendar.from_dates(
        [date(2026, 6, 8), date(2026, 6, 9), date(2026, 6, 10)],
        early_closes={date(2026, 6, 9): time(13, 0)},
    )
    assert cal.close_at(date(2026, 6, 9)) == pd.Timestamp("2026-06-09T17:00:00Z")
    assert cal.entry_for("2026-06-09 12:59") == Entry(
        date(2026, 6, 9), "close", pd.Timestamp("2026-06-09T17:00:00Z")
    )
    assert cal.entry_for("2026-06-09 14:00") == Entry(
        date(2026, 6, 10), "open", pd.Timestamp("2026-06-10T13:30:00Z")
    )


def test_early_close_for_non_session_is_ignored():
    cal = SessionCalendar.from_dates(JUNE_WEEK, early_closes={date(2026, 7, 3): time(13, 0)})
    assert len(cal) == 8
    assert date(2026, 7, 3) not in cal


def test_release_with_no_later_session_has_no_entry():
    assert CAL.entry_for("2026-06-17 16:30") is None  # after the last close
    assert CAL.entry_for("2026-06-18 08:30") is None  # after the last session


def test_entry_is_frozen_and_kind_is_checked():
    with pytest.raises(dataclasses.FrozenInstanceError):
        OPEN_ENTRY.kind = "close"
    with pytest.raises(ValueError):
        Entry(date(2026, 6, 8), "middle", pd.Timestamp("2026-06-08T13:30:00Z"))


def test_entry_for_unparseable_release_raises():
    with pytest.raises(ValueError):
        CAL.entry_for("not a time")


# --- exit_for ---


def test_entries_built_as_expected():
    assert OPEN_ENTRY.kind == "open" and OPEN_ENTRY.session == date(2026, 6, 8)
    assert CLOSE_ENTRY.kind == "close" and CLOSE_ENTRY.session == date(2026, 6, 8)


@pytest.mark.parametrize(
    "horizon, session",
    [(1, date(2026, 6, 8)), (3, date(2026, 6, 10)), (5, date(2026, 6, 12))],
)
def test_open_entry_exits_at_session_plus_horizon_minus_one(horizon, session):
    expected = (session, pd.Timestamp(f"{session.isoformat()}T20:00:00Z"))
    assert CAL.exit_for(OPEN_ENTRY, horizon) == expected


@pytest.mark.parametrize(
    "horizon, session",
    [(1, date(2026, 6, 9)), (3, date(2026, 6, 11)), (5, date(2026, 6, 15))],
)
def test_close_entry_exits_at_session_plus_horizon(horizon, session):
    expected = (session, pd.Timestamp(f"{session.isoformat()}T20:00:00Z"))
    assert CAL.exit_for(CLOSE_ENTRY, horizon) == expected


def test_exit_beyond_calendar_is_none():
    last_open = CAL.entry_for("2026-06-15 08:30")  # session index 5 of 8
    assert CAL.exit_for(last_open, 3) == (date(2026, 6, 17), pd.Timestamp("2026-06-17T20:00:00Z"))
    assert CAL.exit_for(last_open, 5) is None
    last_close = CAL.entry_for("2026-06-17 14:00")  # the final session
    assert CAL.exit_for(last_close, 1) is None


def test_exit_horizon_below_one_raises():
    with pytest.raises(ValueError):
        CAL.exit_for(OPEN_ENTRY, 0)
    with pytest.raises(ValueError):
        CAL.exit_for(CLOSE_ENTRY, -1)


def test_offset_from_non_session_raises():
    with pytest.raises(ValueError):
        CAL.offset(date(2026, 6, 13), 1)  # Saturday is not a session


def test_offset_negative_count_raises():
    with pytest.raises(ValueError):
        CAL.offset(date(2026, 6, 8), -1)
