"""Tests for Exa and Tavily news ingestion. Fake sessions return the real fixtures; no network."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from macro_agent.ingest.http import IngestError
from macro_agent.ingest.news import (
    EXA_URL,
    NEWS_COLUMNS,
    TAVILY_URL,
    article_id,
    exa_search,
    fetch_exa_history,
    fetch_tavily_recent,
    normalize_exa,
    normalize_tavily,
    tavily_search,
    week_windows,
)
from macro_agent.ingest.raw import load_json, save_json

FIXTURES = Path(__file__).parent / "fixtures"
EXA = json.loads((FIXTURES / "exa_search.json").read_text(encoding="utf-8"))
TAVILY = json.loads((FIXTURES / "tavily_search.json").read_text(encoding="utf-8"))


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


def _no_sleep(seconds: float) -> None:
    """Retries never wait in tests."""


def test_article_id_is_16_hex_characters_of_the_sha1_of_the_stripped_url():
    url = "https://x.test/a"
    assert article_id(f"  {url}  ") == article_id(url)
    assert article_id(url) == hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    assert re.fullmatch(r"[0-9a-f]{16}", article_id(url))


def test_normalize_exa_maps_each_result_to_a_news_row():
    df = normalize_exa(EXA, "fed", "Federal Reserve", ingested_utc="2026-06-10T00:00:00Z")
    assert list(df.columns) == list(NEWS_COLUMNS)
    assert len(df) == 3
    assert set(df["provider"]) == {"exa"}
    assert set(df["theme"]) == {"fed"}
    assert set(df["query"]) == {"Federal Reserve"}
    assert df["url"].tolist() == [item["url"] for item in EXA["results"]]
    assert df["article_id"].tolist() == [article_id(item["url"]) for item in EXA["results"]]
    assert df["title"].tolist() == [item["title"] for item in EXA["results"]]
    assert df["text"].tolist() == [item["text"] for item in EXA["results"]]
    assert df["published_utc"].tolist() == [
        "2026-06-01T23:59:59Z", "2026-06-05T23:59:59Z", "2026-06-02T23:59:59Z",
    ]
    assert set(df["ingested_utc"]) == {"2026-06-10T00:00:00Z"}
    assert df["sentiment"].isna().all()
    assert df["macro_tone"].isna().all()


def test_normalize_tavily_maps_each_result_to_a_news_row():
    df = normalize_tavily(TAVILY, "fed", "Federal Reserve", ingested_utc="2026-10-12T00:00:00Z")
    assert list(df.columns) == list(NEWS_COLUMNS)
    assert len(df) == 2
    assert set(df["provider"]) == {"tavily"}
    assert df["published_utc"].tolist() == ["2026-10-11T02:00:00Z", "2026-10-07T19:00:00Z"]
    assert df["text"].tolist() == [item["content"] for item in TAVILY["results"]]
    assert df["article_id"].tolist() == [article_id(item["url"]) for item in TAVILY["results"]]


def test_default_ingested_time_is_utc_in_the_news_format():
    df = normalize_exa(EXA, "fed", "q")
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", df.loc[0, "ingested_utc"])


def test_results_without_a_url_are_skipped_and_a_missing_date_stays_empty():
    response = {"results": [
        {"title": "no url"},
        {"url": "   ", "title": "blank url"},
        {"url": "https://x.test/a", "title": "kept"},
    ]}
    df = normalize_exa(response, "fed", "q", ingested_utc="2026-06-10T00:00:00Z")
    assert df["url"].tolist() == ["https://x.test/a"]
    assert pd.isna(df.loc[0, "published_utc"])


def test_empty_results_give_an_empty_frame_with_news_columns():
    for df in (
        normalize_exa({"results": []}, "fed", "q"),
        normalize_tavily({"results": []}, "fed", "q"),
    ):
        assert df.empty
        assert list(df.columns) == list(NEWS_COLUMNS)


def test_week_windows_clip_the_last_week_and_are_empty_when_end_is_not_after_start():
    assert week_windows(date(2026, 6, 1), date(2026, 6, 17)) == [
        (date(2026, 6, 1), date(2026, 6, 8)),
        (date(2026, 6, 8), date(2026, 6, 15)),
        (date(2026, 6, 15), date(2026, 6, 17)),
    ]
    assert week_windows(date(2026, 6, 1), date(2026, 6, 1)) == []
    assert week_windows(date(2026, 6, 2), date(2026, 6, 1)) == []


def test_exa_search_sends_the_key_only_in_the_header_with_the_documented_body():
    session = _Session([_Response(200, EXA)])
    body = exa_search(
        "Federal Reserve", date(2026, 6, 1), date(2026, 6, 8), "SECRET-KEY",
        session=session, sleep=_no_sleep,
    )
    assert body == EXA
    method, url, kwargs = session.calls[0]
    assert (method, url) == ("POST", EXA_URL)
    assert kwargs["headers"] == {"x-api-key": "SECRET-KEY"}
    assert kwargs["json"] == {
        "query": "Federal Reserve",
        "type": "auto",
        "category": "news",
        "numResults": 10,
        "startPublishedDate": "2026-06-01T00:00:00.000Z",
        "endPublishedDate": "2026-06-08T00:00:00.000Z",
        "contents": {"text": {"maxCharacters": 1000}},
    }
    assert "SECRET-KEY" not in url
    assert "SECRET-KEY" not in json.dumps(kwargs["json"])


def test_tavily_search_sends_the_key_only_in_the_header_with_the_documented_body():
    session = _Session([_Response(200, TAVILY)])
    tavily_search("Federal Reserve", "SECRET-KEY", session=session, sleep=_no_sleep)
    method, url, kwargs = session.calls[0]
    assert (method, url) == ("POST", TAVILY_URL)
    assert kwargs["headers"] == {"Authorization": "Bearer SECRET-KEY"}
    assert kwargs["json"] == {
        "query": "Federal Reserve",
        "topic": "news",
        "time_range": "week",
        "max_results": 10,
        "include_raw_content": False,
    }
    assert "SECRET-KEY" not in url
    assert "SECRET-KEY" not in json.dumps(kwargs["json"])


def test_fetch_exa_history_fetches_every_week_then_reuses_finished_ones(tmp_path):
    themes = {"fed": "Federal Reserve interest rates"}
    first = _Session([_Response(200, EXA)] * 2)
    fetch_exa_history(
        themes, date(2026, 6, 1), date(2026, 6, 15), "key", tmp_path,
        today=date(2026, 6, 10), session=first, sleep=_no_sleep,
    )
    assert len(first.calls) == 2
    assert sorted(p.stem for p in (tmp_path / "exa").glob("*.json")) == [
        "fed_2026-06-01_2026-06-08", "fed_2026-06-08_2026-06-15",
    ]

    # The week ending 06-08 is finished and reused; the week ending 06-15 is still open.
    second = _Session([_Response(200, EXA)])
    df = fetch_exa_history(
        themes, date(2026, 6, 1), date(2026, 6, 15), "key", tmp_path,
        today=date(2026, 6, 10), session=second, sleep=_no_sleep,
    )
    assert len(second.calls) == 1
    assert second.calls[0][2]["json"]["startPublishedDate"] == "2026-06-08T00:00:00.000Z"
    assert len(df) == 3
    assert df["published_utc"].tolist() == [
        "2026-06-01T23:59:59Z", "2026-06-02T23:59:59Z", "2026-06-05T23:59:59Z",
    ]


def test_fetch_exa_history_offline_uses_saved_weeks_and_skips_missing_ones(tmp_path):
    themes = {"fed": "q"}
    online = fetch_exa_history(
        themes, date(2026, 6, 1), date(2026, 6, 8), "key", tmp_path,
        today=date(2026, 6, 20), session=_Session([_Response(200, EXA)]), sleep=_no_sleep,
    )
    session = _Session([])
    offline = fetch_exa_history(
        themes, date(2026, 6, 1), date(2026, 6, 15), None, tmp_path,
        today=date(2026, 6, 20), offline=True, session=session, sleep=_no_sleep,
    )
    assert session.calls == []
    pd.testing.assert_frame_equal(offline, online)


def test_fetch_exa_history_drops_urls_repeated_across_themes(tmp_path):
    session = _Session([_Response(200, EXA), _Response(200, EXA)])
    df = fetch_exa_history(
        {"fed": "fed query", "labor": "labor query"}, date(2026, 6, 1), date(2026, 6, 8),
        "key", tmp_path, today=date(2026, 6, 20), session=session, sleep=_no_sleep,
    )
    assert len(session.calls) == 2
    assert len(df) == 3
    assert df["article_id"].is_unique
    assert set(df["theme"]) == {"fed"}


def test_fetch_tavily_recent_saves_online_and_reloads_offline(tmp_path):
    online = fetch_tavily_recent(
        {"fed": "q"}, "key", tmp_path, today=date(2026, 10, 12),
        session=_Session([_Response(200, TAVILY)]), sleep=_no_sleep,
    )
    assert len(online) == 2
    assert load_json(tmp_path, "tavily", "fed_2026-10-12")["theme"] == "fed"

    session = _Session([])
    offline = fetch_tavily_recent(
        {"fed": "q"}, None, tmp_path, today=date(2026, 10, 13),
        offline=True, session=session, sleep=_no_sleep,
    )
    assert session.calls == []
    pd.testing.assert_frame_equal(offline, online)


def test_fetch_tavily_recent_offline_does_not_load_a_theme_that_only_shares_a_prefix(tmp_path):
    save_json(tmp_path, "tavily", "inflation_2026-10-10", {
        "theme": "inflation", "query": "q", "fetched_utc": "2026-10-10T12:00:00Z",
        "response": {"results": [TAVILY["results"][0]]},
    })
    save_json(tmp_path, "tavily", "inflation_expectations_2026-10-12", {
        "theme": "inflation_expectations", "query": "q", "fetched_utc": "2026-10-12T12:00:00Z",
        "response": TAVILY,
    })
    df = fetch_tavily_recent(
        {"inflation": "q"}, None, tmp_path, today=date(2026, 10, 13),
        offline=True, session=_Session([]), sleep=_no_sleep,
    )
    assert len(df) == 1
    assert set(df["theme"]) == {"inflation"}


def test_fetch_tavily_recent_offline_skips_a_theme_without_a_snapshot(tmp_path):
    df = fetch_tavily_recent(
        {"labor": "q"}, None, tmp_path, today=date(2026, 10, 13),
        offline=True, session=_Session([]), sleep=_no_sleep,
    )
    assert df.empty
    assert list(df.columns) == list(NEWS_COLUMNS)


def test_rows_sort_by_publication_time_with_missing_dates_last(tmp_path):
    response = {"results": [
        {"url": "https://x.test/undated", "title": "undated", "content": "a"},
        {"url": "https://x.test/late", "title": "late", "content": "b",
         "published_date": "Sun, 11 Oct 2026 02:00:00 GMT"},
        {"url": "https://x.test/early", "title": "early", "content": "c",
         "published_date": "Wed, 07 Oct 2026 19:00:00 GMT"},
    ]}
    save_json(tmp_path, "tavily", "fed_2026-10-12", {
        "theme": "fed", "query": "q", "fetched_utc": "2026-10-12T00:00:00Z", "response": response,
    })
    df = fetch_tavily_recent(
        {"fed": "q"}, None, tmp_path, today=date(2026, 10, 12),
        offline=True, session=_Session([]), sleep=_no_sleep,
    )
    assert df["title"].tolist() == ["early", "late", "undated"]


def test_exa_fetch_that_needs_the_network_raises_without_a_key(tmp_path):
    session = _Session([])
    with pytest.raises(IngestError, match="EXA_API_KEY is not set"):
        fetch_exa_history(
            {"fed": "q"}, date(2026, 6, 1), date(2026, 6, 8), None, tmp_path,
            today=date(2026, 6, 20), session=session, sleep=_no_sleep,
        )
    assert session.calls == []


def test_tavily_fetch_raises_without_a_key(tmp_path):
    session = _Session([])
    with pytest.raises(IngestError, match="TAVILY_API_KEY is not set"):
        fetch_tavily_recent(
            {"fed": "q"}, None, tmp_path, today=date(2026, 6, 20),
            session=session, sleep=_no_sleep,
        )
    assert session.calls == []
