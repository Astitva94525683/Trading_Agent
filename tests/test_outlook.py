"""Tests for macro_agent.outlook on the synthetic database from test_pipeline, with no network."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macro_agent import db, outlook, pipeline
from macro_agent.config import ET
from macro_agent.ingest.news import NEWS_COLUMNS
from test_pipeline import AS_OF, _article, _events, _news, _prices, _sessions, _settings

# 16:00 New York on the last session, after its close and before the 2026-08-05 CPI release.
AS_OF_UTC = pd.Timestamp("2026-07-31T20:00:00Z")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Settings and the dataset built from the synthetic database as of AS_OF."""
    tmp_path = tmp_path_factory.mktemp("outlook")
    settings = _settings(tmp_path)
    conn = db.connect(tmp_path / "t.db")
    sessions = _sessions()
    db.upsert(conn, "prices", _prices(sessions), key=["symbol", "date"])
    db.upsert(conn, "events", _events(sessions), key="event_id")
    db.upsert(conn, "news", _news(sessions), key="article_id")
    return settings, pipeline.build_dataset(conn, settings, AS_OF)


def test_latest_releases_are_newest_first_and_skip_future_events(built):
    _, ds = built
    table = outlook.latest_releases(ds.events, AS_OF_UTC, n=50)
    assert list(table.columns) == ["date", "time", "event", "actual", "forecast", "surprise_z"]
    assert table.iloc[0]["date"] == "2026-07-29"  # the last Wednesday release before as-of
    assert (table["date"] <= "2026-07-31").all()  # the 2026-08-05 CPI release is excluded
    stamps = pd.to_datetime(table["date"] + " " + table["time"]).dt.tz_localize(ET)
    assert stamps.is_monotonic_decreasing
    assert len(outlook.latest_releases(ds.events, AS_OF_UTC, n=3)) == 3


def test_upcoming_events_skip_past_releases(built):
    _, ds = built
    table = outlook.upcoming_events(ds.events, AS_OF_UTC)
    assert list(table.columns) == ["date", "time", "event", "impact", "forecast", "previous"]
    assert table.iloc[0][["date", "time"]].tolist() == ["2026-08-05", "08:30"]
    assert (table["date"] > "2026-07-31").all()
    later = pd.Timestamp("2026-08-06T00:00:00Z")
    assert outlook.upcoming_events(ds.events, later).empty


def test_sentiment_trend_has_7_and_30_day_bullets_and_at_most_weeks_rows(built):
    _, ds = built
    table, bullets = outlook.sentiment_trend(ds.news, AS_OF_UTC, weeks=8)
    assert list(table.columns) == ["week", "n", "macro_tone", "sentiment"]
    assert len(table) == 8  # the synthetic news spans more than eight weeks
    assert list(table["week"]) == sorted(table["week"])
    assert len(outlook.sentiment_trend(ds.news, AS_OF_UTC, weeks=3)[0]) == 3
    assert bullets[0].startswith("Macro tone, last 7 days:") and "(30 days:" in bullets[0]
    assert bullets[1].startswith("VADER sentiment, last 7 days:") and "(30 days:" in bullets[1]
    assert any(b.startswith("Most negative theme this week:") for b in bullets)
    # Only articles published by as_of count toward the table.
    published = pd.to_datetime(ds.news["published_utc"], utc=True)
    total = outlook.sentiment_trend(ds.news, AS_OF_UTC, weeks=100)[0]["n"].sum()
    assert total == (published <= AS_OF_UTC).sum()


def test_headlines_prefer_tavily_and_fall_back_to_all_providers():
    news = pd.DataFrame([
        _article("e1", "exa", "fed", "2026-07-30T12:00:00Z", "Exa story"),
        _article("t1", "tavily", "labor", "2026-07-29T12:00:00Z", "Tavily older"),
        _article("t2", "tavily", "growth", "2026-07-30T13:00:00Z", "Tavily newer"),
        _article("t3", "tavily", "growth", "2026-08-02T12:00:00Z", "Tavily after as-of"),
    ], columns=list(NEWS_COLUMNS))
    table = outlook.headlines(news, AS_OF_UTC)
    assert list(table.columns) == ["title", "url", "published", "theme"]
    assert table["title"].tolist() == ["Tavily newer", "Tavily older"]
    assert table.iloc[0]["published"] == "2026-07-30 13:00 UTC"
    fallback = outlook.headlines(news[news["provider"].eq("exa")], AS_OF_UTC)
    assert fallback["title"].tolist() == ["Exa story"]


def test_playbook_scores_beat_and_miss_for_the_next_key_release(built):
    settings, ds = built
    table, note = outlook.playbook(
        ds, settings, AS_OF_UTC, regressor="ridge", classifier="logit"
    )
    assert list(table.columns) == list(outlook.PLAYBOOK_COLUMNS)
    assert set(table["scenario"]) == {"beat", "miss"}
    assert np.isfinite(table["pred"].to_numpy(dtype=float)).all()
    traded = table[table["strategy"].ne("Stay flat")]
    assert traded["pred"].gt(0).all()
    assert traded["win_prob"].between(0, 1).all()
    assert traded.groupby("scenario").size().le(3).all()
    flat = table[table["strategy"].eq("Stay flat")]
    assert flat["strategy_id"].eq("").all() and flat["win_prob"].isna().all()
    assert note.startswith("Next key release: CPI m/m on 2026-08-05 at 08:30 ET;")
    assert "Sessions after 2026-07-31 assume weekdays" in note


def test_playbook_without_upcoming_key_release_is_empty(built):
    settings, ds = built
    table, note = outlook.playbook(ds, settings, pd.Timestamp("2026-08-06T00:00:00Z"))
    assert table.empty
    assert list(table.columns) == list(outlook.PLAYBOOK_COLUMNS)
    assert note == "No high-impact release is scheduled in the calendar window."
