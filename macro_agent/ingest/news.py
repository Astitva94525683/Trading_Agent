"""News search from Exa and Tavily, normalized to rows of the news table.

Exa is queried one week at a time and each week is saved as a raw snapshot, so a finished
week is never fetched again. Tavily only covers the last week, so each day's search is saved
and can be reloaded offline.
"""
from __future__ import annotations

import hashlib
import re
import time
from collections.abc import Callable, Mapping
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from macro_agent.ingest.http import IngestError, request_json
from macro_agent.ingest.raw import list_names, load_json, save_json
from macro_agent.parsing import conservative_published_utc

EXA_URL = "https://api.exa.ai/search"
TAVILY_URL = "https://api.tavily.com/search"
NEWS_COLUMNS = (
    "article_id", "provider", "theme", "query", "title", "url", "published_utc",
    "text", "sentiment", "macro_tone", "ingested_utc",
)
_STAMP = "%Y-%m-%dT%H:%M:%SZ"
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def article_id(url: str) -> str:
    """A short stable id: the first 16 hex characters of the SHA-1 of the stripped URL."""
    return hashlib.sha1(url.strip().encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def exa_search(
    query: str,
    start: date,
    end: date,
    api_key: str,
    *,
    num_results: int = 10,
    max_chars: int = 1000,
    session: requests.Session | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Search Exa news published in [start, end). The key is sent in a header only."""
    return request_json(
        "POST",
        EXA_URL,
        headers={"x-api-key": api_key},
        json={
            "query": query,
            "type": "auto",
            "category": "news",
            "numResults": num_results,
            "startPublishedDate": f"{start.isoformat()}T00:00:00.000Z",
            "endPublishedDate": f"{end.isoformat()}T00:00:00.000Z",
            "contents": {"text": {"maxCharacters": max_chars}},
        },
        session=session,
        sleep=sleep,
    )


def tavily_search(
    query: str,
    api_key: str,
    *,
    max_results: int = 10,
    session: requests.Session | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Search Tavily news from the last week. The key is sent in a header only."""
    return request_json(
        "POST",
        TAVILY_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "query": query,
            "topic": "news",
            "time_range": "week",
            "max_results": max_results,
            "include_raw_content": False,
        },
        session=session,
        sleep=sleep,
    )


def normalize_exa(
    response: Mapping[str, Any], theme: str, query: str, ingested_utc: str | None = None
) -> pd.DataFrame:
    """News-table rows from an Exa response, one per result that has a URL."""
    return _normalize("exa", response, theme, query, ingested_utc, "publishedDate", "text")


def normalize_tavily(
    response: Mapping[str, Any], theme: str, query: str, ingested_utc: str | None = None
) -> pd.DataFrame:
    """News-table rows from a Tavily response, one per result that has a URL."""
    return _normalize("tavily", response, theme, query, ingested_utc, "published_date", "content")


def week_windows(start: date, end: date) -> list[tuple[date, date]]:
    """Half-open 7-day windows from start; the last one is clipped to end. Empty if end <= start."""
    windows: list[tuple[date, date]] = []
    day = start
    while day < end:
        step = min(day + timedelta(days=7), end)
        windows.append((day, step))
        day = step
    return windows


def fetch_exa_history(
    themes: Mapping[str, str],
    start: date,
    end: date,
    api_key: str | None,
    raw_dir: str | Path,
    *,
    today: date,
    offline: bool = False,
    num_results: int = 10,
    session: requests.Session | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """Exa news for each theme over [start, end), with one snapshot per theme and week.

    A week that ends on or before today is reused from its snapshot. Offline, saved snapshots
    are used as they are and weeks without one are skipped.
    """
    frames: list[pd.DataFrame] = []
    for theme, query in themes.items():
        for win_start, win_end in week_windows(start, end):
            name = f"{theme}_{win_start.isoformat()}_{win_end.isoformat()}"
            snapshot = load_json(raw_dir, "exa", name)
            reusable = snapshot is not None and (offline or win_end <= today)
            if not reusable:
                if offline:
                    continue
                if not api_key:
                    raise IngestError("EXA_API_KEY is not set")
                snapshot = {
                    "theme": theme,
                    "query": query,
                    "start": win_start.isoformat(),
                    "end": win_end.isoformat(),
                    "fetched_utc": _utc_now(),
                    "response": exa_search(
                        query, win_start, win_end, api_key,
                        num_results=num_results, session=session, sleep=sleep,
                    ),
                }
                save_json(raw_dir, "exa", name, snapshot)
            frames.append(normalize_exa(
                snapshot["response"], snapshot["theme"], snapshot["query"], snapshot["fetched_utc"],
            ))
    return _combine(frames)


def fetch_tavily_recent(
    themes: Mapping[str, str],
    api_key: str | None,
    raw_dir: str | Path,
    *,
    today: date,
    offline: bool = False,
    max_results: int = 10,
    session: requests.Session | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """Tavily news from the last week for each theme, saved once per day.

    Offline, the newest saved snapshot of each theme is used, and a theme with none is skipped.
    """
    frames: list[pd.DataFrame] = []
    for theme, query in themes.items():
        if offline:
            name = _latest_tavily_name(raw_dir, theme)
            snapshot = load_json(raw_dir, "tavily", name) if name else None
            if snapshot is None:
                continue
        else:
            if not api_key:
                raise IngestError("TAVILY_API_KEY is not set")
            snapshot = {
                "theme": theme,
                "query": query,
                "fetched_utc": _utc_now(),
                "response": tavily_search(
                    query, api_key, max_results=max_results, session=session, sleep=sleep,
                ),
            }
            save_json(raw_dir, "tavily", f"{theme}_{today.isoformat()}", snapshot)
        frames.append(normalize_tavily(
            snapshot["response"], snapshot["theme"], snapshot["query"], snapshot["fetched_utc"],
        ))
    return _combine(frames)


def _normalize(
    provider: str,
    response: Mapping[str, Any],
    theme: str,
    query: str,
    ingested_utc: str | None,
    date_key: str,
    text_key: str,
) -> pd.DataFrame:
    """Rows for one response. A result without a URL is dropped; a missing date stays empty."""
    ingested = ingested_utc or _utc_now()
    records: list[dict[str, Any]] = []
    for item in response.get("results") or []:
        url = (item.get("url") or "").strip()
        if not url:
            continue
        raw_date = item.get(date_key)
        published = conservative_published_utc(raw_date) if raw_date else None
        stamp = None if published is None or pd.isna(published) else published.strftime(_STAMP)
        records.append({
            "article_id": article_id(url),
            "provider": provider,
            "theme": theme,
            "query": query,
            "title": item.get("title") or "",
            "url": url,
            "published_utc": stamp,
            "text": item.get(text_key) or "",
            "sentiment": np.nan,
            "macro_tone": np.nan,
            "ingested_utc": ingested,
        })
    return pd.DataFrame(records, columns=list(NEWS_COLUMNS))


def _latest_tavily_name(raw_dir: str | Path, theme: str) -> str | None:
    """The newest saved Tavily snapshot name for theme, or None.

    Only names of the form theme_YYYY-MM-DD count, so theme "fed" never loads a snapshot
    saved for a longer theme such as "fed_extra".
    """
    prefix = f"{theme}_"
    dated = [
        name for name in list_names(raw_dir, "tavily", prefix)
        if _DATE.fullmatch(name[len(prefix):])
    ]
    return dated[-1] if dated else None


def _combine(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """One table: repeated article ids keep the first row; sorted by date, missing dates last."""
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=list(NEWS_COLUMNS))
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset="article_id", keep="first")
    combined = combined.sort_values(by=["published_utc", "article_id"], na_position="last")
    return combined.reset_index(drop=True)


def _utc_now() -> str:
    """The current time in UTC, in the news-table format."""
    return datetime.now(timezone.utc).strftime(_STAMP)
