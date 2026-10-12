"""Pipeline stage 1 on fetcher stubs and a synthetic database, with no network."""
from __future__ import annotations

import dataclasses
from datetime import date

import numpy as np
import pandas as pd
import pytest

from macro_agent import db, pipeline
from macro_agent.config import ASSETS, ET, FOMC_DECISION_DATES, NEWS_THEMES, VIX_SYMBOL, Settings
from macro_agent.features import feature_columns
from macro_agent.ingest.calendar import EVENT_COLUMNS
from macro_agent.ingest.news import NEWS_COLUMNS
from macro_agent.ingest.prices import PRICE_COLUMNS
from macro_agent.parsing import categorize_event, is_inverted
from macro_agent.sentiment import score_news
from macro_agent.timeutils import to_utc

AS_OF = date(2026, 7, 31)
SYMBOLS = (*ASSETS, VIX_SYMBOL)
VIX_ONLY_DAY = date(2026, 5, 23)  # a Saturday


def _settings(tmp_path):
    return dataclasses.replace(Settings(), data_dir=tmp_path, min_train_slots=10)


def _event(event_id, currency, name, day, hhmm, *, actual=0.3, forecast=0.2, impact="High"):
    release = to_utc(f"{day} {hhmm}")
    return {
        "event_id": event_id,
        "source": "test",
        "currency": currency,
        "event_name": name,
        "category": categorize_event(name),
        "impact": impact,
        "release_utc": release.strftime(db.TS_FORMAT),
        "release_et": release.tz_convert(ET).isoformat(),
        "actual": actual,
        "forecast": forecast,
        "previous": 0.1,
        "unit": "%",
        "actual_raw": f"{actual}%",
        "forecast_raw": f"{forecast}%",
        "previous_raw": "0.1%",
        "inverted": int(is_inverted(name)),
        "ingested_utc": "2026-07-31T12:00:00Z",
    }


def _article(article_id, provider, theme, published, title):
    return {
        "article_id": article_id,
        "provider": provider,
        "theme": theme,
        "query": NEWS_THEMES[theme],
        "title": title,
        "url": f"https://example.com/{article_id}",
        "published_utc": published,
        "text": "",
        "sentiment": None,
        "macro_tone": None,
        "ingested_utc": "2026-07-31T12:00:00Z",
    }


def _calendar_stub():
    rows = [
        _event("usd-cpi", "USD", "CPI m/m", "2026-07-29", "08:30"),
        _event("eur-cpi", "EUR", "CPI m/m", "2026-07-29", "08:30"),
    ]
    return pd.DataFrame(rows, columns=list(EVENT_COLUMNS))


def _news_stub(provider):
    # Both news providers return the same article; ingest keeps the first copy (Exa).
    row = _article("a1", provider, "fed", "2026-07-30T12:00:00Z",
                   "Fed signals rate cuts as inflation cools")
    return pd.DataFrame([row], columns=list(NEWS_COLUMNS))


def _price_stub():
    rows = [
        {"symbol": symbol, "date": day, "open": 100.0, "high": 101.0, "low": 99.0,
         "close": 100.5, "volume": 1.0}
        for symbol in ("SPY", "TLT")
        for day in ("2026-07-29", "2026-07-30")
    ]
    return pd.DataFrame(rows, columns=list(PRICE_COLUMNS))


def test_ingest_upserts_filtered_rows_and_scores_news(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    conn = db.connect(tmp_path / "t.db")
    seen = {}

    def calendar(start, end, token, raw_dir, **kwargs):
        seen["window"] = (start, end)
        return _calendar_stub()

    monkeypatch.setattr(pipeline, "fetch_calendar", calendar)
    monkeypatch.setattr(pipeline, "fetch_exa_history", lambda *a, **k: _news_stub("exa"))
    monkeypatch.setattr(pipeline, "fetch_tavily_recent", lambda *a, **k: _news_stub("tavily"))
    monkeypatch.setattr(pipeline, "fetch_prices", lambda *a, **k: _price_stub())

    counts = pipeline.ingest(conn, settings, AS_OF)

    assert seen["window"] == pipeline.window(settings, AS_OF)
    assert counts == {"events": 1, "news": 1, "prices": 4}
    assert db.read_df(conn, "SELECT currency FROM events")["currency"].tolist() == ["USD"]
    assert len(db.read_df(conn, "SELECT * FROM prices")) == 4
    news = db.read_df(conn, "SELECT provider, macro_tone FROM news")
    assert news["provider"].tolist() == ["exa"]
    assert news["macro_tone"].notna().all()


def _sessions():
    return list(pd.bdate_range("2026-03-02", "2026-07-31").date)


def _prices(sessions, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    for symbol in SYMBOLS:
        close = 100.0
        for day in sessions:
            open_ = close * (1 + rng.normal(0, 0.003))
            close = open_ * (1 + rng.normal(0, 0.008))
            rows.append({
                "symbol": symbol, "date": day, "open": open_,
                "high": max(open_, close), "low": min(open_, close), "close": close,
                "volume": 1.0,
            })
    return pd.DataFrame(rows, columns=list(PRICE_COLUMNS))


def _events(sessions, seed=11):
    """CPI at 08:30 and Fed funds decisions at 14:00 on alternate Wednesdays, plus a CPI
    release after the last session."""
    rng = np.random.default_rng(seed)
    rows = []
    wednesdays = [day for day in sessions if day.weekday() == 2]
    for i, day in enumerate(wednesdays):
        noise = rng.normal(0, 0.1)
        if i % 2 == 0:
            rows.append(_event(f"cpi-{day}", "USD", "CPI m/m", str(day), "08:30",
                               actual=0.2 + noise))
        else:
            rows.append(_event(f"ffr-{day}", "USD", "Federal Funds Rate", str(day), "14:00",
                               actual=0.2 + noise))
    rows.append(_event("cpi-upcoming", "USD", "CPI m/m", "2026-08-05", "08:30"))
    return pd.DataFrame(rows, columns=list(EVENT_COLUMNS))


def _news(sessions):
    """One article per session at 07:00 New York time, cycling through four themes."""
    texts = [
        ("Fed signals rate cuts as inflation cooled", "fed"),
        ("Hotter-than-expected CPI keeps the Fed hawkish", "inflation"),
        ("Payrolls surged and the labor market stayed tight", "labor"),
        ("Recession fears grow as manufacturing contracted", "growth"),
    ]
    rows = []
    for i, day in enumerate(sessions):
        title, theme = texts[i % len(texts)]
        published = to_utc(f"{day} 07:00").strftime(db.TS_FORMAT)
        rows.append(_article(f"news-{i}", "exa", theme, published, title))
    scored = score_news(pd.DataFrame(rows, columns=list(NEWS_COLUMNS)), use_vader=False)
    return scored[list(NEWS_COLUMNS)]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """A synthetic database for the window ending at AS_OF, and the dataset built from it."""
    tmp_path = tmp_path_factory.mktemp("built")
    settings = _settings(tmp_path)
    conn = db.connect(tmp_path / "t.db")
    sessions = _sessions()
    db.upsert(conn, "prices", _prices(sessions), key=["symbol", "date"])
    # Yahoo prints ^VIX on exchange holidays: a VIX-only day must not become a session.
    vix_only = pd.DataFrame([{
        "symbol": VIX_SYMBOL, "date": VIX_ONLY_DAY, "open": 15.0, "high": 15.0, "low": 15.0,
        "close": 15.0, "volume": 0.0,
    }], columns=list(PRICE_COLUMNS))
    db.upsert(conn, "prices", vix_only, key=["symbol", "date"])
    db.upsert(conn, "events", _events(sessions), key="event_id")
    db.upsert(conn, "news", _news(sessions), key="article_id")
    return conn, pipeline.build_dataset(conn, settings, AS_OF)


def test_build_dataset_builds_panel_and_derived_tables(built):
    conn, dataset = built
    assert not dataset.panel.empty
    assert {"strategy_id", "net_ret"} <= set(dataset.panel.columns)
    assert "news_n_7d" in feature_columns(dataset.panel)
    tables = set(db.read_df(conn, "SELECT name FROM sqlite_master WHERE type = 'table'")["name"])
    assert {"slots", "labels", "features"} <= tables
    stored = db.read_df(conn, "SELECT COUNT(*) AS n FROM features")["n"].iloc[0]
    assert stored == len(dataset.panel)


def test_news_count_uses_only_articles_published_before_entry(built):
    _, dataset = built
    slot = dataset.slots.iloc[len(dataset.slots) // 2]
    entry = pd.Timestamp(slot["entry_utc"])
    published = pd.to_datetime(dataset.news["published_utc"], utc=True)
    direct = int(((published >= entry - pd.Timedelta(days=7)) & (published < entry)).sum())
    assert direct > 0
    assert slot["news_n_7d"] == direct


def test_events_after_last_session_make_no_slot(built):
    _, dataset = built
    cutoff = to_utc("2026-08-01")
    releases = pd.to_datetime(dataset.events["release_utc"], utc=True)
    assert (releases >= cutoff).any()  # the upcoming CPI is in the window
    assert (pd.to_datetime(dataset.slots["entry_utc"], utc=True) < cutoff).all()


def test_fomc_dates_include_schedule_and_fed_rate_events(built):
    _, dataset = built
    assert set(FOMC_DECISION_DATES) <= set(dataset.fomc_dates)
    releases = pd.to_datetime(dataset.events["release_utc"], utc=True)
    is_fed_rate = dataset.events["event_name"].eq("Federal Funds Rate")
    fed_days = {ts.tz_convert(ET).date() for ts in releases[is_fed_rate]}
    assert fed_days and fed_days <= set(dataset.fomc_dates)


def test_sessions_come_from_the_anchor_asset_only(built):
    _, dataset = built
    assert VIX_ONLY_DAY not in dataset.calendar.dates
    assert VIX_ONLY_DAY not in set(dataset.closes.index)
    assert dataset.closes.notna().all().all()
