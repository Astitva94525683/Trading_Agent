"""Tests for macro_agent/db.py: schema, upserts, views and derived tables."""

from __future__ import annotations

import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from macro_agent.db import connect, read_df, upsert, write_table

ET = ZoneInfo("America/New_York")


@pytest.fixture
def conn():
    database = connect()
    yield database
    database.close()


def _count(conn, table: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def _event(event_id: str, release_utc: str, release_et: str, name: str = "CPI", **extra) -> dict:
    row = {
        "event_id": event_id,
        "source": "test",
        "event_name": name,
        "release_utc": release_utc,
        "release_et": release_et,
    }
    row.update(extra)
    return row


def _article(article_id: str, published_utc: str, sentiment: float, macro_tone: float) -> dict:
    return {
        "article_id": article_id,
        "provider": "tavily",
        "published_utc": published_utc,
        "sentiment": sentiment,
        "macro_tone": macro_tone,
    }


def test_schema_creates_tables_views_and_indexes(conn):
    kinds = dict(conn.execute("SELECT name, type FROM sqlite_master"))
    assert kinds["events"] == kinds["news"] == kinds["prices"] == "table"
    assert kinds["v_event_news_7d"] == kinds["v_event_market"] == "view"
    assert [r[2] for r in conn.execute("PRAGMA index_info('ix_events_release_utc')")] == ["release_utc"]
    assert [r[2] for r in conn.execute("PRAGMA index_info('ix_news_published_utc')")] == ["published_utc"]


def test_connect_creates_parent_folders_and_keeps_data(tmp_path):
    path = tmp_path / "data" / "nested" / "macro.db"
    conn = connect(path)
    upsert(conn, "prices", pd.DataFrame({"symbol": ["SPY"], "date": [datetime.date(2026, 6, 10)], "close": [500.0]}), ["symbol", "date"])
    conn.close()

    conn = connect(str(path))  # re-applying the schema must leave existing rows alone
    assert path.exists()
    assert _count(conn, "prices") == 1
    conn.close()


def test_upsert_is_idempotent_and_updates_changed_values(conn):
    df = pd.DataFrame(
        [
            _event("a", "2026-06-10T12:30:00Z", "2026-06-10T08:30:00-04:00", actual=3.1),
            _event("b", "2026-06-11T12:30:00Z", "2026-06-11T08:30:00-04:00", name="PPI"),
        ]
    )
    assert upsert(conn, "events", df, ["event_id"]) == 2
    assert upsert(conn, "events", df, ["event_id"]) == 2
    assert _count(conn, "events") == 2

    upsert(conn, "events", df.assign(actual=[3.2, np.nan]), ["event_id"])
    got = read_df(conn, "SELECT event_id, actual FROM events ORDER BY event_id")
    assert got["actual"].iloc[0] == pytest.approx(3.2)
    assert pd.isna(got["actual"].iloc[1])
    assert _count(conn, "events") == 2


def test_nan_nat_and_none_become_null(conn):
    news = pd.DataFrame(
        {
            "article_id": ["n1"],
            "provider": ["tavily"],
            "title": [None],
            "published_utc": pd.to_datetime([None]),
            "sentiment": [np.nan],
            "macro_tone": [np.float64("nan")],
        }
    )
    upsert(conn, "news", news, ["article_id"])
    row = conn.execute("SELECT title, published_utc, sentiment, macro_tone FROM news").fetchone()
    assert row == (None, None, None, None)


def test_timestamps_are_stored_in_utc_and_dates_as_iso_dates(conn):
    news = pd.DataFrame(
        {
            "article_id": ["summer", "winter", "already_utc"],
            "provider": ["tavily"] * 3,
            "published_utc": [
                pd.Timestamp("2026-06-10 08:30", tz=ET),  # EDT, UTC-4
                datetime.datetime(2026, 1, 15, 9, 30, tzinfo=ET),  # EST, UTC-5
                datetime.datetime(2026, 6, 10, 12, 30, tzinfo=datetime.timezone.utc),
            ],
        }
    )
    upsert(conn, "news", news, ["article_id"])
    got = dict(conn.execute("SELECT article_id, published_utc FROM news"))
    assert got == {
        "summer": "2026-06-10T12:30:00Z",
        "winter": "2026-01-15T14:30:00Z",
        "already_utc": "2026-06-10T12:30:00Z",
    }

    prices = pd.DataFrame({"symbol": ["SPY"], "date": [datetime.date(2026, 6, 10)], "close": [500.25]})
    upsert(conn, "prices", prices, ["symbol", "date"])
    assert conn.execute("SELECT date FROM prices").fetchone() == ("2026-06-10",)


def test_numpy_scalars_are_stored_as_python_values(conn):
    events = pd.DataFrame(
        {
            "event_id": ["e1"],
            "source": ["test"],
            "event_name": ["CPI"],
            "release_utc": ["2026-06-10T12:30:00Z"],
            "actual": pd.Series([np.float64(3.2)], dtype=object),
            "inverted": pd.Series([np.int64(1)], dtype=object),
        }
    )
    upsert(conn, "events", events, ["event_id"])
    row = conn.execute("SELECT typeof(actual), actual, typeof(inverted), inverted FROM events").fetchone()
    assert row == ("real", 3.2, "integer", 1)


def test_naive_timestamp_is_rejected(conn):
    news = pd.DataFrame({"article_id": ["n1"], "provider": ["exa"], "published_utc": [pd.Timestamp("2026-06-10 12:30")]})
    with pytest.raises(ValueError, match="timezone-aware"):
        upsert(conn, "news", news, ["article_id"])
    assert _count(conn, "news") == 0


def test_empty_frame_returns_zero(conn):
    empty = pd.DataFrame(columns=["event_id", "source", "event_name", "release_utc"])
    assert upsert(conn, "events", empty, ["event_id"]) == 0
    assert _count(conn, "events") == 0


def test_null_key_values_are_rejected(conn):
    prices = pd.DataFrame({"symbol": ["SPY"], "date": [None], "close": [500.0]})
    with pytest.raises(ValueError, match="NULL"):
        upsert(conn, "prices", prices, ["symbol", "date"])


def test_news_window_covers_the_seven_days_before_release_only(conn):
    upsert(
        conn,
        "events",
        pd.DataFrame(
            [
                _event("cpi", "2026-06-10T12:30:00Z", "2026-06-10T08:30:00-04:00"),
                _event("quiet", "2026-06-20T12:30:00Z", "2026-06-20T08:30:00-04:00"),
            ]
        ),
        ["event_id"],
    )
    upsert(
        conn,
        "news",
        pd.DataFrame(
            [
                _article("at_release", "2026-06-10T12:30:00Z", 0.9, 0.9),
                _article("eight_days_before", "2026-06-02T12:30:00Z", 0.9, 0.9),
                _article("exactly_seven_days", "2026-06-03T12:30:00Z", 0.2, 0.1),
                _article("inside", "2026-06-09T09:00:00Z", -0.4, -0.2),
                _article("after_release", "2026-06-10T13:00:00Z", 0.9, 0.9),
            ]
        ),
        ["article_id"],
    )
    got = read_df(conn, "SELECT * FROM v_event_news_7d ORDER BY event_id").set_index("event_id")

    assert got.loc["cpi", "news_n_7d"] == 2
    assert got.loc["cpi", "news_sent_7d"] == pytest.approx((0.2 - 0.4) / 2)
    assert got.loc["cpi", "news_tone_7d"] == pytest.approx((0.1 - 0.2) / 2)
    assert got.loc["quiet", "news_n_7d"] == 0
    assert pd.isna(got.loc["quiet", "news_sent_7d"])
    assert pd.isna(got.loc["quiet", "news_tone_7d"])


def test_market_view_uses_friday_and_monday_around_a_weekend_release(conn):
    # 2026-06-13 is a Saturday.
    upsert(
        conn,
        "events",
        pd.DataFrame([_event("weekend", "2026-06-13T14:00:00Z", "2026-06-13T10:00:00-04:00")]),
        ["event_id"],
    )
    prices = pd.DataFrame(
        [
            {"symbol": "SPY", "date": datetime.date(2026, 6, 11), "close": 500.0},  # Thu
            {"symbol": "SPY", "date": datetime.date(2026, 6, 12), "close": 505.0},  # Fri
            {"symbol": "SPY", "date": datetime.date(2026, 6, 15), "close": 510.0},  # Mon
            {"symbol": "SPY", "date": datetime.date(2026, 6, 16), "close": 512.0},  # Tue
            {"symbol": "TLT", "date": datetime.date(2026, 6, 12), "close": 90.0},
            {"symbol": "TLT", "date": datetime.date(2026, 6, 15), "close": 91.8},
        ]
    )
    upsert(conn, "prices", prices, ["symbol", "date"])

    got = read_df(conn, "SELECT * FROM v_event_market WHERE event_id = 'weekend'").set_index("symbol")
    assert set(got.index) == {"SPY", "TLT"}
    spy = got.loc["SPY"]
    assert spy["prev_date"] == "2026-06-12" and spy["prev_close"] == 505.0
    assert spy["next_date"] == "2026-06-15" and spy["next_close"] == 510.0
    assert spy["react_ret"] == pytest.approx(510 / 505 - 1)
    tlt = got.loc["TLT"]
    assert tlt["prev_date"] == "2026-06-12" and tlt["prev_close"] == 90.0
    assert tlt["next_date"] == "2026-06-15" and tlt["next_close"] == 91.8
    assert tlt["react_ret"] == pytest.approx(91.8 / 90 - 1)


def test_market_view_next_leg_is_same_day_for_premarket_release(conn):
    # 2026-06-10 is a Wednesday; an 08:30 ET release sits on that session.
    upsert(
        conn,
        "events",
        pd.DataFrame(
            [
                _event("cpi", "2026-06-10T12:30:00Z", "2026-06-10T08:30:00-04:00"),
                _event("early", "2026-06-01T12:30:00Z", "2026-06-01T08:30:00-04:00"),
            ]
        ),
        ["event_id"],
    )
    upsert(
        conn,
        "prices",
        pd.DataFrame(
            [
                {"symbol": "UUP", "date": datetime.date(2026, 6, 9), "close": 27.0},
                {"symbol": "UUP", "date": datetime.date(2026, 6, 10), "close": 27.54},
            ]
        ),
        ["symbol", "date"],
    )
    got = read_df(conn, "SELECT * FROM v_event_market").set_index("event_id")
    cpi = got.loc["cpi"]
    assert cpi["prev_date"] == "2026-06-09" and cpi["next_date"] == "2026-06-10"
    assert cpi["react_ret"] == pytest.approx(0.02)
    # Release before the first price row: no previous close, so no reaction.
    early = got.loc["early"]
    assert pd.isna(early["prev_close"]) and early["next_date"] == "2026-06-09"
    assert pd.isna(early["react_ret"])


def test_write_table_round_trips_and_replaces(conn):
    slots = pd.DataFrame(
        {
            "slot_id": [1, 2],
            "entry_utc": [pd.Timestamp("2026-06-10 12:30", tz="UTC"), pd.NaT],
            "entry_date": [datetime.date(2026, 6, 10), None],
            "surprise_z": [1.5, np.nan],
            "label": ["cpi", None],
            "won": [True, False],
        }
    )
    write_table(conn, "slots", slots)
    rows = conn.execute(
        "SELECT slot_id, entry_utc, entry_date, surprise_z, label, won FROM slots ORDER BY slot_id"
    ).fetchall()
    assert rows == [
        (1, "2026-06-10T12:30:00Z", "2026-06-10", 1.5, "cpi", 1),
        (2, None, None, None, None, 0),
    ]
    assert read_df(conn, "SELECT label FROM slots WHERE slot_id = ?", (1,)).iloc[0, 0] == "cpi"

    write_table(conn, "slots", slots.iloc[:1])  # replace, not append
    assert _count(conn, "slots") == 1


def test_read_df_binds_parameters(conn):
    upsert(
        conn,
        "prices",
        pd.DataFrame({"symbol": ["SPY", "TLT"], "date": ["2026-06-10", "2026-06-10"], "close": [500.0, 90.0]}),
        ["symbol", "date"],
    )
    got = read_df(conn, "SELECT symbol, close FROM prices WHERE symbol = ?", ("TLT",))
    assert got.to_dict("records") == [{"symbol": "TLT", "close": 90.0}]


@pytest.mark.parametrize("bad", ["events; DROP TABLE events", "1events", "ev ents", "events\n", "", 'ev"ents', None])
def test_bad_table_names_raise(bad, conn):
    df = pd.DataFrame([_event("a", "2026-06-10T12:30:00Z", "2026-06-10T08:30:00-04:00")])
    with pytest.raises(ValueError):
        upsert(conn, bad, df, ["event_id"])
    with pytest.raises(ValueError):
        write_table(conn, bad, df)


def test_bad_column_and_key_names_raise(conn):
    df = pd.DataFrame([_event("a", "2026-06-10T12:30:00Z", "2026-06-10T08:30:00-04:00")])
    with pytest.raises(ValueError):
        upsert(conn, "events", df.rename(columns={"source": "sou rce"}), ["event_id"])
    with pytest.raises(ValueError):
        write_table(conn, "derived", df.rename(columns={"source": "sou rce"}))
    with pytest.raises(ValueError):
        write_table(conn, "derived", pd.DataFrame({0: [1]}))
    with pytest.raises(ValueError):
        upsert(conn, "events", df, ["event id"])
    with pytest.raises(ValueError):
        upsert(conn, "events", df, ["event_id; DROP TABLE events"])
    with pytest.raises(ValueError):
        upsert(conn, "events", df, ["news_id"])  # valid name, but not a column of df
    assert _count(conn, "events") == 0
    assert conn.execute("SELECT name FROM sqlite_master WHERE name = 'events'").fetchone() is not None
