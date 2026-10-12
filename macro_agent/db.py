"""SQLite storage for the macro agent (design: docs/design.md section 9).

Raw tables (`events`, `news`, `prices`) are written with idempotent upserts.
Derived tables are replaced with `write_table`. Timestamps are stored as UTC
ISO strings (`YYYY-MM-DDTHH:MM:SSZ`) and dates as `YYYY-MM-DD`.
"""

from __future__ import annotations

import datetime
import math
import re
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id     TEXT PRIMARY KEY,
    source       TEXT NOT NULL,
    currency     TEXT,
    event_name   TEXT NOT NULL,
    category     TEXT,
    impact       TEXT,
    release_utc  TEXT NOT NULL,
    release_et   TEXT,
    actual       REAL,
    forecast     REAL,
    previous     REAL,
    unit         TEXT,
    actual_raw   TEXT,
    forecast_raw TEXT,
    previous_raw TEXT,
    inverted     INTEGER NOT NULL DEFAULT 0,
    ingested_utc TEXT
);
CREATE INDEX IF NOT EXISTS ix_events_release_utc ON events (release_utc);

CREATE TABLE IF NOT EXISTS news (
    article_id    TEXT PRIMARY KEY,
    provider      TEXT NOT NULL,
    theme         TEXT,
    query         TEXT,
    title         TEXT,
    url           TEXT,
    published_utc TEXT,
    text          TEXT,
    sentiment     REAL,
    macro_tone    REAL,
    ingested_utc  TEXT
);
CREATE INDEX IF NOT EXISTS ix_news_published_utc ON news (published_utc);

CREATE TABLE IF NOT EXISTS prices (
    symbol TEXT NOT NULL,
    date   TEXT NOT NULL,
    open   REAL,
    high   REAL,
    low    REAL,
    close  REAL,
    volume REAL,
    PRIMARY KEY (symbol, date)
);

-- News published in [release - 7 days, release). Events with no news get count 0 and NULL means.
CREATE VIEW IF NOT EXISTS v_event_news_7d AS
SELECT e.event_id,
       e.release_utc,
       COUNT(n.article_id) AS news_n_7d,
       AVG(n.sentiment)    AS news_sent_7d,
       AVG(n.macro_tone)   AS news_tone_7d
FROM events AS e
LEFT JOIN news AS n
       ON n.published_utc >= strftime('%Y-%m-%dT%H:%M:%SZ', e.release_utc, '-7 days')
      AND n.published_utc <  e.release_utc
GROUP BY e.event_id, e.release_utc;

-- Per (event, symbol): last close before the release's ET date and first close on or after it.
CREATE VIEW IF NOT EXISTS v_event_market AS
WITH ev AS (
    SELECT event_id, event_name, release_utc, substr(release_et, 1, 10) AS d_et
    FROM events
),
pairs AS (
    SELECT ev.event_id, ev.event_name, ev.release_utc, ev.d_et, s.symbol
    FROM ev
    CROSS JOIN (SELECT DISTINCT symbol FROM prices) AS s
),
legs AS (
    SELECT p.event_id, p.event_name, p.release_utc, p.symbol,
           (SELECT q.date  FROM prices AS q
             WHERE q.symbol = p.symbol AND q.date <  p.d_et
             ORDER BY q.date DESC LIMIT 1) AS prev_date,
           (SELECT q.close FROM prices AS q
             WHERE q.symbol = p.symbol AND q.date <  p.d_et
             ORDER BY q.date DESC LIMIT 1) AS prev_close,
           (SELECT q.date  FROM prices AS q
             WHERE q.symbol = p.symbol AND q.date >= p.d_et
             ORDER BY q.date ASC LIMIT 1) AS next_date,
           (SELECT q.close FROM prices AS q
             WHERE q.symbol = p.symbol AND q.date >= p.d_et
             ORDER BY q.date ASC LIMIT 1) AS next_close
    FROM pairs AS p
)
SELECT event_id, event_name, release_utc, symbol,
       prev_date, prev_close, next_date, next_close,
       next_close / prev_close - 1.0 AS react_ret
FROM legs;
"""


def connect(path: str | Path = ":memory:") -> sqlite3.Connection:
    """Open a SQLite database (creating parent folders for file paths) and apply the schema."""
    target = str(path)
    if target != ":memory:":
        Path(target).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target)
    conn.executescript(SCHEMA)
    return conn


def upsert(conn: sqlite3.Connection, table: str, df: pd.DataFrame, key: Sequence[str]) -> int:
    """Insert the rows of `df` into `table`, updating rows that match on `key`.

    Commits and returns the number of rows sent (0 for an empty frame). NaN, NaT and
    None become NULL; timezone-aware timestamps are stored in UTC; `datetime.date`
    values are stored as 'YYYY-MM-DD'. Naive timestamps are rejected.
    """
    keys = [key] if isinstance(key, str) else list(key)
    _check_name(table)
    cols = _check_columns(df)
    for k in keys:
        _check_name(k)
    if not keys:
        raise ValueError("upsert needs at least one key column")
    if df.empty:
        return 0
    missing = [k for k in keys if k not in cols]
    if missing:
        raise ValueError(f"key columns missing from df: {missing}")
    if df[keys].isna().to_numpy().any():
        raise ValueError(f"key columns {keys} must not contain NULL/NaN values")

    rest = [c for c in cols if c not in keys]
    if rest:
        action = "DO UPDATE SET " + ", ".join(f"{_q(c)} = excluded.{_q(c)}" for c in rest)
    else:
        action = "DO NOTHING"
    sql = (
        f"INSERT INTO {_q(table)} ({', '.join(_q(c) for c in cols)}) "
        f"VALUES ({', '.join('?' for _ in cols)}) "
        f"ON CONFLICT ({', '.join(_q(k) for k in keys)}) {action}"
    )
    rows = _rows(df, cols)
    with conn:
        conn.executemany(sql, rows)
    return len(rows)


def read_df(conn: sqlite3.Connection, sql: str, params: Sequence[Any] = ()) -> pd.DataFrame:
    """Run a query (with `?` placeholders bound to `params`) and return a DataFrame."""
    return pd.read_sql_query(sql, conn, params=tuple(params))


def write_table(conn: sqlite3.Connection, name: str, df: pd.DataFrame) -> None:
    """Replace table `name` with the rows of `df`, for derived tables such as slots or features.

    Values are converted as in `upsert`. Column types are inferred by pandas.
    """
    _check_name(name)
    cols = _check_columns(df)
    out = pd.DataFrame({c: pd.Series(_column_values(df[c]), dtype=object) for c in cols})
    out.to_sql(name, conn, if_exists="replace", index=False)


def _check_name(name: object) -> str:
    """Return `name` if it is a plain SQL identifier, otherwise raise ValueError."""
    if not isinstance(name, str) or _IDENTIFIER.fullmatch(name) is None:
        raise ValueError(f"invalid SQL identifier: {name!r}")
    return name


def _check_columns(df: pd.DataFrame) -> list[str]:
    """Validate every column name of `df` and return them as a list."""
    cols = [_check_name(c) for c in df.columns]
    if len(set(cols)) != len(cols):
        raise ValueError(f"duplicate column names: {cols}")
    return cols


def _q(name: str) -> str:
    """Quote an identifier that has already passed `_check_name`."""
    return f'"{name}"'


def _rows(df: pd.DataFrame, cols: Sequence[str]) -> list[tuple[Any, ...]]:
    """Convert every cell of `df` to a storable value, one tuple per row."""
    return list(zip(*(_column_values(df[c]) for c in cols)))


def _column_values(series: pd.Series) -> list[Any]:
    """Convert every cell of one column."""
    return [_to_sql_value(v) for v in series.tolist()]


def _to_sql_value(value: Any) -> Any:
    """Convert one cell to a value that sqlite3 can store."""
    if value is None or value is pd.NaT or value is pd.NA:
        return None
    if isinstance(value, np.datetime64):
        if np.isnat(value):
            return None
        value = pd.Timestamp(value)
    elif isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, datetime.datetime):  # before date: datetime is a date subclass
        if value.tzinfo is None:
            raise ValueError(f"naive timestamp {value!r}: make it timezone-aware before storing")
        return value.astimezone(datetime.timezone.utc).strftime(TS_FORMAT)
    if isinstance(value, datetime.date):
        return value.isoformat()
    return value
