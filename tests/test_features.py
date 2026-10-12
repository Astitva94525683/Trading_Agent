"""Tests for macro_agent.features (docs/design.md sections 4 and 6)."""
from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from macro_agent.config import CATEGORIES
from macro_agent.features import build_slots, prepare_events, surprise_index, time_features
from macro_agent.timeutils import SessionCalendar, to_utc


def _weekdays(start: date, end: date) -> list[date]:
    """Weekdays from `start` to `end`, inclusive."""
    days = (start + timedelta(days=i) for i in range((end - start).days + 1))
    return [d for d in days if d.weekday() < 5]


# Six weeks of weekdays, Mon 1 June to Fri 10 July 2026, no holidays. All EDT.
CAL = SessionCalendar.from_dates(_weekdays(date(2026, 6, 1), date(2026, 7, 10)))

# Synthetic CPI-style releases, one week apart, all at 08:30 New York time.
CPI_DAYS = [
    "2026-06-10 08:30",
    "2026-06-17 08:30",
    "2026-06-24 08:30",
    "2026-07-01 08:30",
    "2026-07-08 08:30",
    "2026-07-15 08:30",
]

SLOT_COLUMNS = [
    "slot_id", "entry_session", "entry_kind", "entry_utc", "last_release_utc", "n_events",
    "n_high", "primary_event", "primary_category", "surprise_z", "abs_surprise_z", "has_surprise",
    "cat_fed", "cat_inflation", "cat_labor", "cat_growth", "cat_consumer", "cat_housing",
    "cat_other", "event_ids",
]
RAW_EVENT_COLUMNS = [
    "event_id", "event_name", "category", "impact", "release_utc", "actual", "forecast", "inverted",
]


def _row(event_id, name, category, impact, release, actual=None, forecast=None, inverted=False):
    """One raw calendar event. `release` is read as New York time unless it carries a zone."""
    return {
        "event_id": event_id,
        "event_name": name,
        "category": category,
        "impact": impact,
        "release_utc": release,
        "actual": actual,
        "forecast": forecast,
        "inverted": inverted,
    }


def _assert_close(actual, expected):
    """Element-wise approximate comparison, where `None` in `expected` means NaN."""
    actual = list(actual)
    assert len(actual) == len(expected), f"{actual} != {expected}"
    for got, want in zip(actual, expected):
        if want is None:
            assert math.isnan(got), f"expected NaN in {actual}, got {got!r}"
        else:
            assert got == pytest.approx(want, rel=1e-9, abs=1e-12), f"{actual} != {expected}"


def _slots_for(rows, calendar=CAL, **kwargs):
    """Prepare raw event rows, then build slots on `calendar`."""
    return build_slots(prepare_events(pd.DataFrame(rows)), calendar, **kwargs)


def _scored(rows):
    """Events in prepare_events shape, with weight and surprise_z given directly."""
    return pd.DataFrame([
        {
            "event_id": event_id,
            "event_name": name,
            "category": category,
            "weight": weight,
            "release_utc": to_utc("2026-06-10 08:30"),
            "surprise_z": z,
        }
        for event_id, name, category, weight, z in rows
    ])


# --- prepare_events ----------------------------------------------------------


def test_default_keeps_high_and_medium_impact_and_drops_low():
    events = pd.DataFrame([
        _row("h", "CPI m/m", "inflation", "High", "2026-06-10 08:30", 0.3, 0.2),
        _row("m", "Retail Sales m/m", "growth", "Medium", "2026-06-11 08:30", 0.5, 0.4),
        _row("l", "Factory Orders", "growth", "Low", "2026-06-12 10:00", 1.0, 0.9),
    ])
    out = prepare_events(events)
    assert out["event_id"].tolist() == ["h", "m"]
    assert out["weight"].tolist() == [1.0, 0.5]


@pytest.mark.parametrize(
    "min_weight, kept",
    [(0.25, ["h", "m", "l"]), (0.5, ["h", "m"]), (1.0, ["h"])],
)
def test_min_weight_sets_the_cut_off(min_weight, kept):
    events = pd.DataFrame([
        _row("h", "CPI m/m", "inflation", "High", "2026-06-10 08:30"),
        _row("m", "Retail Sales m/m", "growth", "Medium", "2026-06-11 08:30"),
        _row("l", "Factory Orders", "growth", "Low", "2026-06-12 10:00"),
    ])
    assert prepare_events(events, min_weight=min_weight)["event_id"].tolist() == kept


def test_numeric_impact_ranks_are_weighted_like_labels():
    events = pd.DataFrame([
        _row("h", "CPI m/m", "inflation", 3, "2026-06-10 08:30"),
        _row("m", "Retail Sales m/m", "growth", 2, "2026-06-11 08:30"),
        _row("l", "Factory Orders", "growth", 1, "2026-06-12 10:00"),
    ])
    assert prepare_events(events, min_weight=0.25)["weight"].tolist() == [1.0, 0.5, 0.25]


@pytest.mark.parametrize("flag", [True, 1])
def test_inverted_indicator_flips_the_sign(flag):
    events = pd.DataFrame([
        _row("u", "Unemployment Rate", "labor", "High", "2026-06-05 08:30", 4.4, 4.2,
             inverted=flag),
        _row("c", "CPI m/m", "inflation", "High", "2026-06-10 08:30", 0.4, 0.3, inverted=False),
    ])
    out = prepare_events(events).set_index("event_id")
    assert out.loc["u", "surprise_raw"] == pytest.approx(0.2)
    assert out.loc["u", "surprise"] == pytest.approx(-0.2)  # above forecast is a weak print
    assert out.loc["c", "surprise"] == pytest.approx(0.1)


def test_surprise_z_falls_back_to_sign_then_scales_by_earlier_median_and_clips():
    surprises = [0.1, -0.2, 0.3, 1.0, -0.5, -10.0]
    events = pd.DataFrame([
        _row(f"cpi{i}", "CPI m/m", "inflation", "High", day, s, 0.0)
        for i, (day, s) in enumerate(zip(CPI_DAYS, surprises))
    ])
    z = prepare_events(events)["surprise_z"].tolist()
    # 1, 2: sign. 3: 0.3 / median(0.1, 0.2) = 2. 4: 1.0 / median(0.1, 0.2, 0.3) = 5, clipped to 3.
    # 5: -0.5 / median(0.1, 0.2, 0.3, 1.0) = -2.
    # 6: -10 / median(0.1, 0.2, 0.3, 0.5, 1.0) = -33, clipped to -3.
    _assert_close(z, [1.0, -1.0, 2.0, 3.0, -2.0, -3.0])


def test_zero_median_falls_back_to_sign_and_zero_surprise_gives_zero():
    surprises = [0.0, 0.0, 0.4, 0.0]
    events = pd.DataFrame([
        _row(f"cpi{i}", "CPI m/m", "inflation", "High", day, s, 0.0)
        for i, (day, s) in enumerate(zip(CPI_DAYS, surprises))
    ])
    # Release 3 has priors 0, 0 (median 0), so it falls back to sign(0.4).
    _assert_close(prepare_events(events)["surprise_z"], [0.0, 0.0, 1.0, 0.0])


def test_missing_actual_gives_nan_and_is_not_a_prior():
    actuals = [0.1, None, 0.3, 0.4]
    events = pd.DataFrame([
        _row(f"cpi{i}", "CPI m/m", "inflation", "High", day, a, 0.0)
        for i, (day, a) in enumerate(zip(CPI_DAYS, actuals))
    ])
    # Release 4 has non-NaN priors 0.1 and 0.3, so its median is 0.2 and z = 0.4 / 0.2.
    _assert_close(prepare_events(events)["surprise_z"], [1.0, None, 1.0, 2.0])


def test_later_releases_do_not_change_earlier_surprise_z():
    def z_values(last_actual):
        surprises = [0.1, -0.2, 0.3, 1.0, last_actual]
        events = pd.DataFrame([
            _row(f"cpi{i}", "CPI m/m", "inflation", "High", day, s, 0.0)
            for i, (day, s) in enumerate(zip(CPI_DAYS, surprises))
        ])
        return prepare_events(events)["surprise_z"].tolist()

    early, late = z_values(-0.5), z_values(9.0)
    assert early[:4] == late[:4]
    assert early[4] != late[4]


def test_releases_at_the_same_time_are_not_priors_for_each_other():
    events = pd.DataFrame([
        _row("a", "Retail Sales m/m", "growth", "High", "2026-06-10 08:30", 0.5, 0.0),
        _row("b", "Retail Sales m/m", "growth", "High", "2026-06-10 08:30", -0.25, 0.0),
        _row("c", "Retail Sales m/m", "growth", "High", "2026-06-17 08:30", 0.5, 0.0),
    ])
    # a and b see no strictly earlier release. c's priors are 0.5 and -0.25, median |.| 0.375.
    _assert_close(prepare_events(events)["surprise_z"], [1.0, -1.0, 0.5 / 0.375])


def test_scaling_uses_only_the_same_event_history():
    rows = []
    for i, (day, cpi) in enumerate(zip(CPI_DAYS[:3], [0.1, -0.2, 0.3])):
        rows.append(_row(f"rs{i}", "Retail Sales m/m", "growth", "High", day, 100.0, 0.0))
        rows.append(_row(f"cpi{i}", "CPI m/m", "inflation", "High", day, cpi, 0.0))
    out = prepare_events(pd.DataFrame(rows))
    _assert_close(out.loc[out["event_name"] == "CPI m/m", "surprise_z"], [1.0, -1.0, 2.0])
    _assert_close(out.loc[out["event_name"] == "Retail Sales m/m", "surprise_z"], [1.0, 1.0, 1.0])


def test_output_is_utc_sorted_and_the_input_is_unchanged():
    # The ids sort opposite to the release times, so sorting by event_id alone would fail.
    events = pd.DataFrame([
        _row("a_late", "Retail Sales m/m", "growth", "High", "2026-06-11T08:30:00-04:00", 0.5, 0.4),
        _row("z_early", "CPI m/m", "inflation", "High", "2026-06-10T12:30:00Z", 0.3, 0.2),
    ])
    before = events.copy()
    out = prepare_events(events)
    pd.testing.assert_frame_equal(events, before)
    assert out["event_id"].tolist() == ["z_early", "a_late"]
    assert out["release_utc"].tolist() == [
        pd.Timestamp("2026-06-10T12:30:00Z"),
        pd.Timestamp("2026-06-11T12:30:00Z"),
    ]
    assert str(out["release_utc"].dt.tz) == "UTC"


def test_empty_input_gives_empty_frames_with_the_expected_columns():
    empty = pd.DataFrame(columns=RAW_EVENT_COLUMNS)
    prepared = prepare_events(empty)
    assert prepared.empty
    assert {"weight", "surprise_raw", "surprise", "surprise_z"} <= set(prepared.columns)
    slots = build_slots(prepared, CAL)
    assert slots.empty
    assert list(slots.columns) == SLOT_COLUMNS


# --- build_slots -------------------------------------------------------------


def test_same_morning_releases_share_an_open_slot_and_fomc_gets_a_close_slot():
    rows = [
        _row("cpi", "CPI m/m", "inflation", "High", "2026-06-10 08:30", 0.3, 0.2),
        _row("claims", "Initial Jobless Claims", "labor", "Medium", "2026-06-10 08:30",
             230000, 240000, inverted=True),
        _row("fomc", "FOMC Statement", "fed", "High", "2026-06-10 14:00"),
    ]
    slots = _slots_for(rows)
    assert list(slots.columns) == SLOT_COLUMNS
    assert slots["slot_id"].tolist() == ["2026-06-10_open", "2026-06-10_close"]

    opened, closed = slots.iloc[0], slots.iloc[1]
    assert opened["entry_session"] == date(2026, 6, 10)
    assert opened["entry_kind"] == "open"
    assert opened["entry_utc"] == pd.Timestamp("2026-06-10T13:30:00Z")  # 09:30 EDT
    assert opened["last_release_utc"] == pd.Timestamp("2026-06-10T12:30:00Z")  # 08:30 EDT
    assert opened["n_events"] == 2
    assert opened["n_high"] == 1
    assert opened["event_ids"] == "claims,cpi"  # release order, then event_id on a tie
    assert opened["primary_event"] == "CPI m/m"  # High beats Medium
    assert opened["primary_category"] == "inflation"
    # Both are first releases, so both z values are +1 (the claims surprise is flipped to +10000).
    assert opened["surprise_z"] == pytest.approx(1.0)
    assert bool(opened["has_surprise"])
    assert (opened["cat_inflation"], opened["cat_labor"], opened["cat_fed"]) == (1, 1, 0)

    assert closed["entry_session"] == date(2026, 6, 10)
    assert closed["entry_kind"] == "close"
    assert closed["entry_utc"] == pd.Timestamp("2026-06-10T20:00:00Z")  # 16:00 EDT
    assert closed["n_events"] == 1
    assert closed["primary_event"] == "FOMC Statement"
    assert closed["primary_category"] == "fed"
    assert math.isnan(closed["surprise_z"])
    assert math.isnan(closed["abs_surprise_z"])
    assert not bool(closed["has_surprise"])


def test_saturday_release_enters_at_the_next_monday_open():
    rows = [_row("sat", "Consumer Sentiment", "consumer", "High", "2026-06-06 10:00", 60.0, 58.0)]
    slots = _slots_for(rows)
    assert slots["slot_id"].tolist() == ["2026-06-08_open"]
    assert slots["entry_session"].iloc[0] == date(2026, 6, 8)


@pytest.mark.parametrize(
    "next_session, kept", [(date(2026, 6, 12), True), (date(2026, 6, 13), False)]
)
def test_entries_more_than_max_gap_days_after_release_are_dropped(next_session, kept):
    # Release Mon 8 June after the close. The next session is 4 days later (kept) or 5 (dropped).
    cal = SessionCalendar.from_dates([date(2026, 6, 8), next_session])
    rows = [_row("late", "Retail Sales m/m", "growth", "High", "2026-06-08 17:00", 0.5, 0.4)]
    assert (len(_slots_for(rows, cal)) == 1) is kept
    assert len(_slots_for(rows, cal, max_gap_days=5)) == 1


def test_release_after_the_last_session_has_no_entry_and_is_dropped():
    cal = SessionCalendar.from_dates([date(2026, 6, 8)])
    rows = [_row("end", "Retail Sales m/m", "growth", "High", "2026-06-08 17:00", 0.5, 0.4)]
    assert _slots_for(rows, cal).empty


def test_surprise_z_is_the_impact_weighted_mean_of_the_available_values():
    events = _scored([
        ("h", "CPI m/m", "inflation", 1.0, 1.0),
        ("m", "Jobless Claims", "labor", 0.5, -2.0),
        ("l", "Factory Orders", "growth", 0.25, 3.0),
        ("m2", "ISM", "growth", 0.5, np.nan),
    ])
    slot = build_slots(events, CAL).iloc[0]
    # (1 * 1 + 0.5 * -2 + 0.25 * 3) / (1 + 0.5 + 0.25). The NaN event is in neither sum.
    assert slot["surprise_z"] == pytest.approx(3 / 7)
    assert slot["abs_surprise_z"] == pytest.approx(3 / 7)
    assert bool(slot["has_surprise"])
    assert slot["n_events"] == 4
    assert slot["n_high"] == 1


def test_zero_weighted_mean_means_no_surprise():
    events = _scored([
        ("h", "CPI m/m", "inflation", 1.0, 1.0),
        ("m", "Jobless Claims", "labor", 0.5, -2.0),
    ])
    slot = build_slots(events, CAL).iloc[0]
    assert slot["surprise_z"] == 0.0  # (1 - 1) / 1.5
    assert not bool(slot["has_surprise"])


def test_slot_without_any_scaled_surprise_is_nan():
    events = _scored([("m", "Jobless Claims", "labor", 0.5, np.nan)])
    slot = build_slots(events, CAL).iloc[0]
    assert math.isnan(slot["surprise_z"])
    assert math.isnan(slot["abs_surprise_z"])
    assert not bool(slot["has_surprise"])


PRIMARY_CASES = [
    pytest.param(
        [("Zeta CPI", "inflation", "Medium", "2026-06-10 08:30"),
         ("Industrial Production", "growth", "High", "2026-06-10 08:30")],
        ("Industrial Production", "growth"),
        id="weight-beats-category-order",
    ),
    pytest.param(
        [("Advance Retail Sales", "growth", "High", "2026-06-10 08:30"),
         ("Zeta CPI", "inflation", "High", "2026-06-10 08:30")],
        ("Zeta CPI", "inflation"),
        id="category-order-beats-name",
    ),
    pytest.param(
        [("Aardvark Index", "trade", "High", "2026-06-10 08:30"),
         ("Zeta Retail Sales", "growth", "High", "2026-06-10 08:30")],
        ("Zeta Retail Sales", "growth"),
        id="unlisted-category-ranks-last",
    ),
    pytest.param(
        [("Zeta PPI", "inflation", "High", "2026-06-10 07:00"),
         ("Alpha CPI", "inflation", "High", "2026-06-10 08:30")],
        ("Zeta PPI", "inflation"),
        id="earlier-release-beats-name",
    ),
    pytest.param(
        [("Zeta CPI", "inflation", "High", "2026-06-10 08:30"),
         ("Alpha CPI", "inflation", "High", "2026-06-10 08:30")],
        ("Alpha CPI", "inflation"),
        id="name-breaks-full-tie",
    ),
]


@pytest.mark.parametrize("specs, expected", PRIMARY_CASES)
def test_primary_event_tie_breaks(specs, expected):
    rows = [
        _row(f"e{i}", name, category, impact, release)
        for i, (name, category, impact, release) in enumerate(specs)
    ]
    slot = _slots_for(rows).iloc[0]
    assert (slot["primary_event"], slot["primary_category"]) == expected


def test_category_counts_and_high_impact_count():
    rows = [
        _row("e1", "Initial Jobless Claims", "labor", "Medium", "2026-06-10 07:00",
             230000, 240000, inverted=True),
        _row("e2", "CPI m/m", "inflation", "High", "2026-06-10 08:30", 0.3, 0.2),
        _row("e3", "Retail Sales m/m", "growth", "High", "2026-06-10 08:30", 0.5, 0.4),
        _row("e4", "Trade Balance", "trade", "High", "2026-06-10 08:30", -50.0, -48.0),
    ]
    slot = _slots_for(rows).iloc[0]
    assert slot["event_ids"] == "e1,e2,e3,e4"
    assert slot["n_events"] == 4
    assert slot["n_high"] == 3  # e2, e3 and e4; e1 is Medium
    counts = {name: int(slot[f"cat_{name}"]) for name in CATEGORIES}
    assert counts == {
        "fed": 0, "inflation": 1, "labor": 1, "growth": 1, "consumer": 0, "housing": 0, "other": 0,
    }
    # "trade" is not in CATEGORIES, so no cat_ column counts that event.
    assert sum(counts.values()) == slot["n_events"] - 1
    assert slot["primary_event"] == "CPI m/m"  # weight 1.0, and inflation ranks above growth


# --- time_features -----------------------------------------------------------

HAND_COLUMNS = [
    "entry_session", "entry_kind", "entry_utc", "last_release_utc", "primary_category",
]


def _hand_slots(specs):
    """Slots built by hand: (entry session, kind, entry UTC, last release UTC, primary category)."""
    return pd.DataFrame(
        [
            {
                "entry_session": session,
                "entry_kind": kind,
                "entry_utc": pd.Timestamp(entry),
                "last_release_utc": pd.Timestamp(released),
                "primary_category": category,
            }
            for session, kind, entry, released, category in specs
        ],
        columns=HAND_COLUMNS,
    )


HAND = _hand_slots([
    # 08:30 EST (Tue 10 Feb). The open is 09:30 EST, which is 14:30 UTC.
    (date(2026, 2, 10), "open", "2026-02-10T14:30:00Z", "2026-02-10T13:30:00Z", "inflation"),
    # 08:30 EDT (Wed 10 June). The open is 13:30 UTC.
    (date(2026, 6, 10), "open", "2026-06-10T13:30:00Z", "2026-06-10T12:30:00Z", "inflation"),
    # 14:00 EDT FOMC, a close entry at 16:00 EDT.
    (date(2026, 6, 10), "close", "2026-06-10T20:00:00Z", "2026-06-10T18:00:00Z", "fed"),
    (date(2026, 6, 12), "open", "2026-06-12T13:30:00Z", "2026-06-12T12:30:00Z", "labor"),
    (date(2026, 6, 15), "open", "2026-06-15T13:30:00Z", "2026-06-15T12:30:00Z", "labor"),
    (date(2026, 8, 3), "open", "2026-08-03T13:30:00Z", "2026-08-03T12:30:00Z", "growth"),
])
FOMC = [date(2026, 3, 18), date(2026, 6, 10), date(2026, 7, 29)]


def test_release_hour_is_8_30_in_both_est_and_edt():
    _assert_close(time_features(HAND, FOMC)["release_hour_et"], [8.5, 8.5, 14.0, 8.5, 8.5, 8.5])


def test_calendar_fields_describe_the_entry_session():
    out = time_features(HAND, FOMC)
    assert out["dow"].tolist() == [1, 2, 2, 4, 0, 0]  # Tue, Wed, Wed, Fri, Mon, Mon
    assert out["dom"].tolist() == [10, 10, 10, 12, 15, 3]
    assert out["week_of_month"].tolist() == [2, 2, 2, 2, 3, 1]
    assert out["month"].tolist() == [2, 6, 6, 6, 6, 8]
    assert out["is_open_entry"].tolist() == [1, 1, 0, 1, 1, 1]


def test_week_of_month_changes_after_days_7_14_21_and_28():
    days = [1, 7, 8, 14, 15, 21, 22, 28, 29, 31]
    slots = _hand_slots([
        (date(2026, 7, day), "open", f"2026-07-{day:02d}T13:30:00Z",
         f"2026-07-{day:02d}T12:30:00Z", "growth")
        for day in days
    ])
    out = time_features(slots, [])
    assert out["week_of_month"].tolist() == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]


def test_spacing_uses_the_previous_slot_and_the_previous_slot_of_the_same_category():
    out = time_features(HAND, FOMC)
    _assert_close(
        out["days_since_prev_slot"],
        [None, 120 - 1 / 24, 6.5 / 24, 1 + 17.5 / 24, 3.0, 49.0],
    )
    _assert_close(
        out["days_since_same_category"],
        [None, 120 - 1 / 24, None, None, 3.0, None],
    )


def test_fomc_distances_count_from_the_entry_session():
    out = time_features(HAND, FOMC)
    # A decision on the entry day counts as "next" at 0 days, but not as "last".
    _assert_close(out["days_to_next_fomc"], [36, 0, 0, 47, 44, None])
    _assert_close(out["days_since_last_fomc"], [None, 84, 84, 2, 5, 5])


def test_slots_same_week_counts_slots_in_the_same_iso_week():
    # Week 24 (8-14 June) holds three slots. Every other slot is alone in its ISO week.
    out = time_features(HAND, FOMC)
    assert out["slots_same_week"].tolist() == [1, 3, 3, 3, 1, 1]


def test_iso_weeks_cross_the_new_year_boundary():
    slots = _hand_slots([
        (date(2025, 12, 26), "open", "2025-12-26T14:30:00Z", "2025-12-26T13:30:00Z", "labor"),
        (date(2025, 12, 29), "open", "2025-12-29T14:30:00Z", "2025-12-29T13:30:00Z", "labor"),
        (date(2026, 1, 2), "open", "2026-01-02T14:30:00Z", "2026-01-02T13:30:00Z", "labor"),
    ])
    # 26 Dec 2025 is in ISO week 52 of 2025. 29 Dec 2025 and 2 Jan 2026 are both in ISO 2026-W01.
    assert time_features(slots, [])["slots_same_week"].tolist() == [1, 2, 2]


def test_row_order_does_not_change_the_features_and_the_output_keeps_input_order():
    forward = time_features(HAND, FOMC).reset_index(drop=True)
    reversed_back = time_features(HAND.iloc[::-1], FOMC).iloc[::-1].reset_index(drop=True)
    pd.testing.assert_frame_equal(reversed_back, forward)


def test_empty_slots_give_empty_frames():
    out = time_features(_hand_slots([]), FOMC)
    assert out.empty
    assert "days_to_next_fomc" in out.columns


# --- surprise_index ----------------------------------------------------------

SI_COLUMNS = ["entry_utc", "last_release_utc", "surprise_z", "primary_category"]


def _si_frame(specs):
    """Slots for surprise_index: (time UTC, surprise_z, primary category).

    Entry and release share the same time, so by default every age is a whole number of days.
    """
    return pd.DataFrame(
        [
            {
                "entry_utc": pd.Timestamp(t),
                "last_release_utc": pd.Timestamp(t),
                "surprise_z": z,
                "primary_category": category,
            }
            for t, z, category in specs
        ],
        columns=SI_COLUMNS,
    )


SI3 = _si_frame([
    ("2026-06-01T12:00:00Z", 1.0, "inflation"),  # A
    ("2026-07-01T12:00:00Z", -2.0, "labor"),  # B, 30 days after A
    ("2026-07-31T12:00:00Z", 0.5, "inflation"),  # C, 60 days after A and 30 days after B
])


def test_surprise_index_hand_computed_values_for_three_slots():
    out = surprise_index(SI3)
    # A has no earlier slot, so it is NaN everywhere.
    # B: only A, at weight 0.5 ** (30 / 30) = 0.5. So si_all = si_inflation = 1.0.
    # C: A at weight 0.25 (z 1) and B at weight 0.5 (z -2).
    #    si_all = (0.25 * 1 + 0.5 * -2) / 0.75 = -1.
    #    si_inflation = A alone = 1. si_labor = B alone = -2.
    _assert_close(out["si_all"], [None, 1.0, -1.0])
    _assert_close(out["si_inflation"], [None, 1.0, 1.0])
    _assert_close(out["si_labor"], [None, None, -2.0])
    _assert_close(out["si_growth"], [None, None, None])
    _assert_close(out["si_fed"], [None, None, None])


def test_age_runs_from_the_earlier_slot_release_not_its_entry():
    slots = _si_frame([
        ("2026-06-01T12:00:00Z", 1.0, "inflation"),  # A
        ("2026-07-01T12:00:00Z", -2.0, "labor"),  # B, with its release 15 days before its entry
        ("2026-07-31T12:00:00Z", 0.5, "inflation"),  # C
    ])
    slots.loc[1, "last_release_utc"] = pd.Timestamp("2026-06-16T12:00:00Z")
    out = surprise_index(slots)
    # For C, A is 60 days old (weight 0.25) and B's release is 45 days old (weight 2 ** -1.5).
    # si_all = (0.25 - 2 ** -1.5 * 2) / (0.25 + 2 ** -1.5) = 3 * sqrt(2) - 5.
    _assert_close(out["si_all"], [None, 1.0, 3 * math.sqrt(2) - 5])
    _assert_close(out["si_labor"], [None, None, -2.0])


def test_halflife_sets_how_fast_old_slots_fade():
    out = surprise_index(SI3, halflife_days=1e9)  # effectively equal weights
    assert out["si_all"].iloc[2] == pytest.approx(-0.5, rel=1e-6)  # (1 + -2) / 2


def test_slots_without_a_surprise_are_skipped():
    slots = _si_frame([
        ("2026-06-01T12:00:00Z", 1.0, "inflation"),  # A
        ("2026-07-01T12:00:00Z", -2.0, "labor"),  # B
        ("2026-07-15T12:00:00Z", np.nan, "growth"),  # D, no surprise
        ("2026-07-31T12:00:00Z", 0.5, "inflation"),  # C
    ])
    out = surprise_index(slots)
    # D sees A at 44 days and B at 14 days. Their weight ratio is 2, so si_all = (1 - 4) / 3.
    # C's values match the three-slot case, because D is excluded from them.
    _assert_close(out["si_all"], [None, 1.0, -1.0, -1.0])
    _assert_close(out["si_inflation"], [None, 1.0, 1.0, 1.0])
    _assert_close(out["si_labor"], [None, None, -2.0, -2.0])
    _assert_close(out["si_growth"], [None, None, None, None])


def test_category_list_controls_which_columns_are_added():
    out = surprise_index(SI3, categories=("labor",))
    assert list(out.columns) == SI_COLUMNS + ["si_all", "si_labor"]
    _assert_close(out["si_labor"], [None, None, -2.0])


def test_row_order_does_not_change_the_values():
    slots = _si_frame([
        ("2026-06-01T12:00:00Z", 1.0, "inflation"),
        ("2026-07-01T12:00:00Z", -2.0, "labor"),
        ("2026-07-15T12:00:00Z", np.nan, "growth"),
        ("2026-07-31T12:00:00Z", 0.5, "inflation"),
    ])
    forward = surprise_index(slots).reset_index(drop=True)
    reversed_back = surprise_index(slots.iloc[::-1]).iloc[::-1].reset_index(drop=True)
    pd.testing.assert_frame_equal(reversed_back, forward)


@pytest.mark.parametrize("halflife", [0.0, -5.0])
def test_halflife_must_be_positive(halflife):
    with pytest.raises(ValueError):
        surprise_index(SI3, halflife_days=halflife)


def test_empty_slots_give_empty_frame_with_index_columns():
    out = surprise_index(_si_frame([]))
    assert out.empty
    assert {"si_all", "si_inflation", "si_labor", "si_growth", "si_fed"} <= set(out.columns)
