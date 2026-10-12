from __future__ import annotations

import json
import math
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from macro_agent.ingest.calendar import (
    DEFAULT_ACTOR,
    EVENT_COLUMNS,
    SOURCE,
    calendar_input,
    fetch_calendar,
    normalize_calendar,
    run_actor,
)
from macro_agent.ingest.http import IngestError
from macro_agent.ingest.raw import load_json
from macro_agent.parsing import make_event_id

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "forexfactory_items.json"
RUNS_URL = "https://api.apify.com/v2/acts/xtracto~forexfactory-calendar/runs"
TOKEN = "apify-test-token"
INGESTED = "2026-10-12T10:00:00Z"
BASE_ITEM = {
    "id": 150001, "title": "CPI m/m", "currency": "USD", "impact": "high",
    "datetimeISO": "2026-09-11T12:30:00Z", "dateline": 1789129800,
    "timeLabel": "7:30pm", "actual": "0.4%",
}


class _Response:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body
        self.text = str(body)

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


class _Session:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _Clock:
    """Fake monotonic clock: sleeping advances it."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def _run(status, run_id="run1", dataset="ds1"):
    return {"data": {"id": run_id, "status": status, "defaultDatasetId": dataset}}


def _fixture_items():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _rows():
    return normalize_calendar(_fixture_items(), ingested_utc=INGESTED).set_index("event_id")


def test_calendar_input_asks_for_one_range_and_leaves_the_proxy_to_the_actor():
    assert calendar_input(date(2026, 9, 10), date(2026, 9, 17)) == {
        "dateRange": "range",
        "rangeFrom": "2026-09-10",
        "rangeTo": "2026-09-17",
        "currencies": ["USD"],
        "minImpact": "medium",
        "upcomingOnly": False,
        "maxItems": 0,
    }


def test_fixture_drops_holidays_and_untimed_items_and_keeps_the_later_duplicate():
    frame = normalize_calendar(_fixture_items(), ingested_utc=INGESTED)
    assert list(frame.columns) == list(EVENT_COLUMNS)
    assert list(frame["event_id"]) == [
        "ff149224", "ff150001", "ff150003", "ff150004", "ff150002", "ff150006",
    ]
    assert frame.index.tolist() == list(range(6))


def test_cpi_row_has_utc_and_new_york_times_and_parsed_values():
    row = _rows().loc["ff150001"]
    assert row["release_utc"] == "2026-09-11T12:30:00Z"
    assert row["release_et"] == "2026-09-11T08:30:00-04:00"
    assert row["actual"] == 0.4
    assert row["forecast"] == 0.3
    assert row["unit"] == "%"
    assert row["category"] == "inflation"
    assert row["impact"] == "high"
    assert row["inverted"] == 0
    assert row["source"] == SOURCE
    assert row["ingested_utc"] == INGESTED


def test_later_claims_duplicate_is_kept_and_marked_inverted():
    row = _rows().loc["ff150002"]
    assert row["actual"] == 238000.0
    assert row["inverted"] == 1
    assert row["category"] == "labor"
    assert row["unit"] == "K"


def test_fed_rate_is_the_new_york_afternoon_and_categorised_as_fed():
    row = _rows().loc["ff150003"]
    assert row["release_et"] == "2026-09-16T14:00:00-04:00"
    assert row["category"] == "fed"


def test_press_conference_without_numbers_has_nan_and_no_raw_text():
    row = _rows().loc["ff150004"]
    assert math.isnan(row["actual"])
    assert row["actual_raw"] is None
    assert row["unit"] is None


def test_release_after_daylight_saving_ends_uses_eastern_standard_time():
    row = _rows().loc["ff150006"]
    assert row["release_et"] == "2026-11-06T08:30:00-05:00"


def test_other_currencies_are_kept():
    row = _rows().loc["ff149224"]
    assert row["currency"] == "CAD"
    assert row["release_et"] == "2026-06-30T08:30:00-04:00"


def test_dateline_is_used_when_datetime_iso_is_missing():
    item = {**BASE_ITEM, "datetimeISO": ""}
    frame = normalize_calendar([item], ingested_utc=INGESTED)
    assert frame.loc[0, "release_utc"] == "2026-09-11T12:30:00Z"


@pytest.mark.parametrize("change", [
    {"impact": "Holiday"},
    {"title": "   "},
    {"datetimeISO": None, "dateline": None},
    {"timeLabel": "All Day"},
    {"timeLabel": "Tentative"},
])
def test_holidays_untitled_undated_and_untimed_items_are_skipped(change):
    assert len(normalize_calendar([BASE_ITEM], ingested_utc=INGESTED)) == 1
    assert normalize_calendar([{**BASE_ITEM, **change}], ingested_utc=INGESTED).empty


def test_no_items_give_an_empty_frame_with_the_event_columns():
    frame = normalize_calendar([], ingested_utc=INGESTED)
    assert frame.empty
    assert list(frame.columns) == list(EVENT_COLUMNS)


def test_item_without_an_id_gets_a_generated_event_id():
    item = {
        "title": "ISM Manufacturing PMI", "currency": "USD", "impact": "high",
        "datetimeISO": "2026-10-01T14:00:00Z", "timeLabel": "10:00am", "actual": "52.4",
    }
    frame = normalize_calendar([item], ingested_utc=INGESTED)
    assert frame.loc[0, "event_id"] == make_event_id(
        SOURCE, "USD", "ISM Manufacturing PMI", "2026-10-01T14:00:00Z"
    )


def test_run_actor_polls_until_success_then_reads_the_dataset():
    session = _Session([
        _Response(201, _run("READY")),
        _Response(200, _run("RUNNING")),
        _Response(200, _run("SUCCEEDED")),
        _Response(200, [{"id": 1}]),
    ])
    sleeps = []
    items = run_actor(DEFAULT_ACTOR, {"maxItems": 0}, TOKEN, poll_seconds=2.0,
                      session=session, sleep=sleeps.append)
    assert items == [{"id": 1}]
    assert [(method, url) for method, url, _ in session.calls] == [
        ("POST", RUNS_URL),
        ("GET", "https://api.apify.com/v2/actor-runs/run1"),
        ("GET", "https://api.apify.com/v2/actor-runs/run1"),
        ("GET", "https://api.apify.com/v2/datasets/ds1/items"),
    ]
    assert sleeps == [2.0, 2.0]
    for _, url, kwargs in session.calls:
        assert kwargs["headers"] == {"Authorization": f"Bearer {TOKEN}"}
        assert TOKEN not in url
        assert TOKEN not in str(kwargs.get("params"))
    assert session.calls[0][2]["json"] == {"maxItems": 0}
    assert session.calls[-1][2]["params"] == {"format": "json", "clean": "true"}


@pytest.mark.parametrize("status", ["FAILED", "ABORTED", "TIMED-OUT"])
def test_run_actor_raises_naming_the_run_when_it_ends_badly(status):
    session = _Session([_Response(201, _run("RUNNING")), _Response(200, _run(status))])
    with pytest.raises(IngestError, match=f"run1.*{status}"):
        run_actor(DEFAULT_ACTOR, {}, TOKEN, session=session, sleep=lambda s: None)


def test_run_actor_gives_up_once_the_timeout_has_passed():
    clock = _Clock()
    session = _Session([_Response(201, _run("RUNNING"))] + [_Response(200, _run("RUNNING"))] * 5)
    with pytest.raises(IngestError, match="run1"):
        run_actor(DEFAULT_ACTOR, {}, TOKEN, poll_seconds=5.0, timeout_seconds=12.0,
                  session=session, sleep=clock.sleep, clock=clock)
    assert clock.now == 15.0


def test_run_actor_rejects_a_dataset_that_is_not_a_list():
    session = _Session([_Response(201, _run("SUCCEEDED")), _Response(200, {"error": "nope"})])
    with pytest.raises(IngestError, match="list"):
        run_actor(DEFAULT_ACTOR, {}, TOKEN, session=session, sleep=lambda s: None)


def test_online_fetch_saves_the_raw_items_and_offline_reloads_an_equal_frame(tmp_path):
    items = _fixture_items()
    session = _Session([
        _Response(201, _run("READY")),
        _Response(200, _run("SUCCEEDED")),
        _Response(200, items),
    ])
    online = fetch_calendar(
        date(2026, 9, 10), date(2026, 9, 17), TOKEN, tmp_path,
        session=session, sleep=lambda s: None,
        now=datetime(2026, 10, 12, 10, 0, tzinfo=timezone.utc),
    )
    saved = load_json(tmp_path, "calendar", "forexfactory_2026-09-10_2026-09-17")
    assert saved["actor"] == DEFAULT_ACTOR
    assert saved["input"] == calendar_input(date(2026, 9, 10), date(2026, 9, 17))
    assert session.calls[0][2]["json"] == saved["input"]
    assert saved["fetched_utc"] == INGESTED
    assert saved["items"] == items
    offline = fetch_calendar(date(2026, 9, 10), date(2026, 9, 17), None, tmp_path, offline=True)
    assert len(offline) == 6
    pd.testing.assert_frame_equal(offline, online)


def test_offline_fetch_without_a_saved_snapshot_raises(tmp_path):
    with pytest.raises(IngestError, match="no calendar snapshot"):
        fetch_calendar(date(2026, 9, 10), date(2026, 9, 17), None, tmp_path, offline=True)


@pytest.mark.parametrize("token", [None, ""])
def test_online_fetch_without_a_token_raises_before_any_request(tmp_path, token):
    session = _Session([])
    with pytest.raises(IngestError, match="APIFY_TOKEN"):
        fetch_calendar(date(2026, 9, 10), date(2026, 9, 17), token, tmp_path, session=session)
    assert session.calls == []


def test_online_fetch_sends_the_run_limits_and_records_the_run(tmp_path):
    finished = _run("SUCCEEDED")
    finished["data"]["usageTotalUsd"] = 0.01
    finished["data"]["chargedEventCounts"] = {"apify-default-dataset-item": 6}
    session = _Session([_Response(201, _run("READY")), _Response(200, finished),
                        _Response(200, _fixture_items())])
    fetch_calendar(
        date(2026, 9, 10), date(2026, 9, 17), TOKEN, tmp_path,
        session=session, sleep=lambda s: None,
        max_charge_usd=0.1, memory_mb=1024, timeout_secs=600,
    )
    limits = {"maxTotalChargeUsd": 0.1, "memory": 1024, "timeout": 600}
    assert session.calls[0][2]["params"] == limits
    saved = load_json(tmp_path, "calendar", "forexfactory_2026-09-10_2026-09-17")
    assert saved["run_options"] == limits
    assert saved["run"]["usageTotalUsd"] == 0.01
    assert saved["run"]["chargedEventCounts"] == {"apify-default-dataset-item": 6}
