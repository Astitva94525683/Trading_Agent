"""Pipeline stage 1: ingest the raw sources into SQLite, then build the model panel.

`ingest` fills the events, news and prices tables for a window around an as-of date.
`build_dataset` reads them back, builds slots, labels and market features, writes the derived
tables (slots, labels, features) and returns everything in a `Dataset`. See docs/design.md
section 2 (pipeline) and section 9 (database).
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from macro_agent import db
from macro_agent.config import (
    ET, FOMC_DECISION_DATES, NEWS_THEMES, VIX_SYMBOL, Settings, api_key, apify_token,
)
from macro_agent.features import (
    build_panel,
    build_slots,
    market_features,
    news_features,
    prepare_events,
    surprise_index,
    time_features,
)
from macro_agent.ingest.calendar import EVENT_COLUMNS, fetch_calendar
from macro_agent.ingest.news import NEWS_COLUMNS, fetch_exa_history, fetch_tavily_recent
from macro_agent.ingest.prices import PRICE_COLUMNS, fetch_prices
from macro_agent.sentiment import score_news
from macro_agent.strategies import compute_labels, pivot_prices, strategy_grid
from macro_agent.timeutils import SessionCalendar


@dataclass
class Dataset:
    """Tables and features built by `build_dataset` for one as-of date.

    `events` holds the prepared events in the window, past and upcoming. `news` and `prices`
    are the stored table rows read for the build.
    """

    as_of: date
    events: pd.DataFrame
    slots: pd.DataFrame
    labels: pd.DataFrame
    market: pd.DataFrame
    panel: pd.DataFrame
    news: pd.DataFrame
    prices: pd.DataFrame
    opens: pd.DataFrame
    closes: pd.DataFrame
    calendar: SessionCalendar
    fomc_dates: list[date]


def window(settings: Settings, as_of: date) -> tuple[date, date]:
    """The first and last calendar dates of the data window around `as_of`."""
    return (
        as_of - timedelta(days=settings.lookback_days),
        as_of + timedelta(days=settings.lookahead_days),
    )


def ingest(
    conn: sqlite3.Connection, settings: Settings, as_of: date, offline: bool = False
) -> dict[str, int]:
    """Fetch events, news and prices for the window around `as_of` and upsert them.

    Offline, the fetchers read saved snapshots. Returns the rows sent to each table.
    """
    start, end = window(settings, as_of)

    events = fetch_calendar(
        start, end, apify_token(), settings.raw_dir,
        actor=settings.calendar_actor, offline=offline,
        max_charge_usd=settings.apify_max_charge_usd, memory_mb=settings.apify_memory_mb,
        timeout_secs=settings.apify_timeout_secs,
    )
    events = events.loc[events["currency"].isin(settings.currencies), list(EVENT_COLUMNS)]

    news = pd.concat(
        [
            fetch_exa_history(
                NEWS_THEMES, start, as_of + timedelta(days=1), api_key("EXA_API_KEY"),
                settings.raw_dir, today=as_of, offline=offline,
                num_results=settings.news_results,
            ),
            fetch_tavily_recent(
                NEWS_THEMES, api_key("TAVILY_API_KEY"), settings.raw_dir,
                today=as_of, offline=offline, max_results=settings.news_results,
            ),
        ],
        ignore_index=True,
    )
    news = news.drop_duplicates("article_id", keep="first")
    news = score_news(news, use_vader=True)[list(NEWS_COLUMNS)]

    prices = fetch_prices(
        (*settings.assets, VIX_SYMBOL),
        start - timedelta(days=settings.price_warmup_days),
        as_of,
        settings.raw_dir,
        today=as_of,
        offline=offline,
    )

    return {
        "events": db.upsert(conn, "events", events, key="event_id"),
        "news": db.upsert(conn, "news", news, key="article_id"),
        "prices": db.upsert(conn, "prices", prices[list(PRICE_COLUMNS)], key=["symbol", "date"]),
    }


def build_dataset(conn: sqlite3.Connection, settings: Settings, as_of: date) -> Dataset:
    """Build slots, labels and the model panel from the stored tables for `as_of`.

    Writes the derived tables slots, labels and features (the panel) to the database.
    Raises ValueError when the prices table has no rows for the window.
    """
    start, end = window(settings, as_of)
    marks = ",".join("?" * len(settings.currencies))
    events = db.read_df(
        conn,
        "SELECT * FROM events WHERE release_utc >= ? AND release_utc < ? "
        f"AND currency IN ({marks})",
        [f"{start}T00:00:00Z", f"{end + timedelta(days=1)}T00:00:00Z", *settings.currencies],
    )
    events = prepare_events(events, min_weight=settings.min_impact_weight)

    price_start = start - timedelta(days=settings.price_warmup_days)
    prices = db.read_df(
        conn,
        "SELECT * FROM prices WHERE date >= ? AND date <= ? ORDER BY symbol, date",
        [str(price_start), str(as_of)],
    )
    if prices.empty:
        raise ValueError(f"no prices from {price_start} to {as_of}; run ingest first")
    # Sessions are the anchor asset's (SPY's) trading days. Yahoo also prints ^VIX values on
    # exchange holidays and a partial bar for today, so other symbols' extra dates are dropped.
    anchor = prices["symbol"] == settings.assets[0]
    session_days = set(prices.loc[anchor & prices["close"].notna(), "date"])
    prices = prices[prices["date"].isin(session_days)].reset_index(drop=True)
    opens, closes = pivot_prices(prices)
    sessions = [pd.Timestamp(day).date() for day in closes.index]
    calendar = SessionCalendar.from_dates(sessions)

    is_fed_rate = events["category"].eq("fed") & events["event_name"].str.contains(
        "Federal Funds Rate", regex=False
    )
    fed_releases = events.loc[is_fed_rate, "release_utc"]
    fomc_dates = sorted(
        set(FOMC_DECISION_DATES) | {ts.tz_convert(ET).date() for ts in fed_releases}
    )

    slots = build_slots(events, calendar)
    slots = surprise_index(time_features(slots, fomc_dates))
    news = db.read_df(conn, "SELECT * FROM news")
    slots = news_features(slots, news)

    labels = compute_labels(
        slots, opens, closes, calendar,
        strategy_grid(settings.assets, settings.rules, settings.horizons),
        cost_bps=settings.cost_bps,
    )
    market = market_features(slots, opens, closes, settings.assets, VIX_SYMBOL)
    panel = build_panel(slots, labels, market)

    db.write_table(conn, "slots", slots)
    db.write_table(conn, "labels", labels)
    db.write_table(conn, "features", panel)

    return Dataset(
        as_of=as_of,
        events=events,
        slots=slots,
        labels=labels,
        market=market,
        panel=panel,
        news=news,
        prices=prices,
        opens=opens,
        closes=closes,
        calendar=calendar,
        fomc_dates=fomc_dates,
    )
