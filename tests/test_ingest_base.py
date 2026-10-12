from __future__ import annotations

import pytest
import requests

from macro_agent.ingest.http import IngestError, request_json
from macro_agent.ingest.raw import latest_name, list_names, load_json, save_json


def test_snapshots_round_trip_and_list_in_name_order(tmp_path):
    save_json(tmp_path, "exa", "fed_2026-06-01_2026-06-08", {"results": [1]})
    save_json(tmp_path, "exa", "fed_2026-05-25_2026-06-01", {"results": []})
    save_json(tmp_path, "exa", "labor_2026-06-01_2026-06-08", {"results": [2]})
    assert load_json(tmp_path, "exa", "fed_2026-06-01_2026-06-08") == {"results": [1]}
    assert load_json(tmp_path, "exa", "missing") is None
    assert list_names(tmp_path, "exa", "fed_") == [
        "fed_2026-05-25_2026-06-01", "fed_2026-06-01_2026-06-08",
    ]
    assert latest_name(tmp_path, "exa", "fed_") == "fed_2026-06-01_2026-06-08"
    assert latest_name(tmp_path, "tavily") is None


@pytest.mark.parametrize("name", ["../x", "a/b", "", ".hidden"])
def test_snapshot_names_cannot_escape_the_raw_folder(tmp_path, name):
    with pytest.raises(ValueError):
        save_json(tmp_path, "exa", name, {})


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


def test_request_json_retries_429_and_connection_errors_then_succeeds():
    session = _Session([
        _Response(429, "slow down"),
        requests.ConnectionError("reset"),
        _Response(200, {"ok": True}),
    ])
    waits = []
    body = request_json("GET", "https://x.test/a", session=session, sleep=waits.append)
    assert body == {"ok": True}
    assert waits == [2.0, 4.0]
    assert len(session.calls) == 3


def test_request_json_raises_on_client_error_without_leaking_query():
    session = _Session([_Response(401, "bad key")])
    with pytest.raises(IngestError) as info:
        request_json("GET", "https://x.test/a?token=SECRET", session=session, sleep=lambda s: None)
    assert "401" in str(info.value)
    assert "SECRET" not in str(info.value)


def test_request_json_gives_up_after_retries():
    session = _Session([_Response(503, "down")] * 3)
    with pytest.raises(IngestError):
        request_json("GET", "https://x.test/a", session=session, retries=2, sleep=lambda s: None)
    assert len(session.calls) == 3


def test_request_json_rejects_a_body_that_is_not_json():
    session = _Session([_Response(200, ValueError("no json"))])
    with pytest.raises(IngestError):
        request_json("GET", "https://x.test/a", session=session, sleep=lambda s: None)
