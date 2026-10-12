"""ForexFactory economic calendar, pulled through an Apify actor and normalized to one row per event."""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from macro_agent.ingest.http import IngestError, request_json
from macro_agent.ingest.raw import latest_name, load_json, save_json
from macro_agent.parsing import categorize_event, is_inverted, make_event_id, parse_value

SOURCE = "forexfactory"
APIFY_BASE = "https://api.apify.com/v2"
DEFAULT_ACTOR = "xtracto/forexfactory-calendar"
EVENT_COLUMNS = (
    "event_id", "source", "currency", "event_name", "category", "impact",
    "release_utc", "release_et", "actual", "forecast", "previous", "unit",
    "actual_raw", "forecast_raw", "previous_raw", "inverted", "ingested_utc",
)
_VALUE_FIELDS = ("actual", "forecast", "previous")
_FAILED_STATES = frozenset({"FAILED", "ABORTED", "TIMED-OUT"})
_RUN_FIELDS = ("id", "status", "defaultDatasetId")
_UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def calendar_input(
    start: date,
    end: date,
    currencies: Sequence[str] = ("USD",),
    min_impact: str = "medium",
) -> dict[str, Any]:
    """Actor input for the events from start to end in the given currencies."""
    # proxyConfiguration is left out on purpose: the actor's default (Apify residential
    # proxy) is what returns the actual values.
    return {
        "dateRange": "range",
        "rangeFrom": start.isoformat(),
        "rangeTo": end.isoformat(),
        "currencies": list(currencies),
        "minImpact": min_impact,
        "upcomingOnly": False,
        "maxItems": 0,
    }


def run_actor(
    actor: str,
    run_input: dict[str, Any],
    token: str,
    *,
    poll_seconds: float = 5.0,
    timeout_seconds: float = 900.0,
    session: requests.Session | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    run_options: Mapping[str, Any] | None = None,
    run_log: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Run an Apify actor until it succeeds and return the items of its default dataset.

    run_options are sent as query parameters of the run request (e.g. maxTotalChargeUsd,
    memory, timeout). When run_log is given, the final run object is appended to it, so the
    caller can record the run's id, usage and charges.
    """
    headers = {"Authorization": f"Bearer {token}"}
    started = clock()
    run = _run_data(request_json(
        "POST",
        f"{APIFY_BASE}/acts/{actor.replace('/', '~')}/runs",
        headers=headers, params=dict(run_options or {}), json=run_input,
        session=session, sleep=sleep,
    ))
    while run["status"] != "SUCCEEDED":
        if run["status"] in _FAILED_STATES:
            raise IngestError(f"Apify run {run['id']} ended with status {run['status']}")
        if clock() - started > timeout_seconds:
            raise IngestError(
                f"Apify run {run['id']} still {run['status']} after {timeout_seconds:g}s"
            )
        sleep(poll_seconds)
        run = _run_data(request_json(
            "GET", f"{APIFY_BASE}/actor-runs/{run['id']}",
            headers=headers, session=session, sleep=sleep,
        ))
    items = request_json(
        "GET", f"{APIFY_BASE}/datasets/{run['defaultDatasetId']}/items",
        headers=headers, params={"format": "json", "clean": "true"}, session=session, sleep=sleep,
    )
    if not isinstance(items, list):
        raise IngestError(f"Apify dataset {run['defaultDatasetId']} did not return a list")
    if run_log is not None:
        run_log.append(run)
    return items


def normalize_calendar(
    items: Sequence[Mapping[str, Any]], ingested_utc: str | None = None
) -> pd.DataFrame:
    """One row per event with EVENT_COLUMNS, sorted by release time.

    Holidays, untitled items, items without a release time and items whose time label has no
    digit ("All Day", "Tentative") are skipped. A repeated event id keeps its last occurrence.
    """
    ingested = ingested_utc if ingested_utc is not None else _utc_stamp(datetime.now(timezone.utc))
    rows: dict[str, dict[str, Any]] = {}
    for item in items:
        row = _normalize_item(item, ingested)
        if row is not None:
            rows[row["event_id"]] = row
    # object dtype keeps None in the text columns (pandas 3 would store NaN there)
    frame = pd.DataFrame(list(rows.values()), columns=list(EVENT_COLUMNS), dtype=object)
    frame = frame.astype({
        "actual": "float64", "forecast": "float64", "previous": "float64", "inverted": "int64",
    })
    return frame.sort_values(["release_utc", "event_id"]).reset_index(drop=True)


def fetch_calendar(
    start: date,
    end: date,
    token: str | None,
    raw_dir: str | Path,
    *,
    actor: str = DEFAULT_ACTOR,
    offline: bool = False,
    session: requests.Session | None = None,
    sleep: Callable[[float], None] = time.sleep,
    now: datetime | None = None,
    max_charge_usd: float | None = None,
    memory_mb: int | None = None,
    timeout_secs: int | None = None,
) -> pd.DataFrame:
    """Calendar events from start to end.

    Online, runs the actor and saves the raw items under raw_dir/calendar first, with a summary
    of the run (id, usage, charged events). max_charge_usd caps the actor's pay-per-event
    charges (Apify's maxTotalChargeUsd); memory_mb and timeout_secs bound the run. Offline,
    reloads the newest saved snapshot, so a run can be repeated without network access.
    """
    if offline:
        name = latest_name(raw_dir, "calendar", "forexfactory_")
        if name is None:
            raise IngestError(f"no calendar snapshot in {raw_dir}; run once online first")
        payload = load_json(raw_dir, "calendar", name)
    else:
        if not token:
            raise IngestError("APIFY_TOKEN (or APIFY_API_KEY) is not set")
        run_input = calendar_input(start, end)
        options = {
            name: value
            for name, value in (
                ("maxTotalChargeUsd", max_charge_usd), ("memory", memory_mb),
                ("timeout", timeout_secs),
            )
            if value is not None
        }
        runs: list[dict[str, Any]] = []
        items = run_actor(
            actor, run_input, token, session=session, sleep=sleep,
            run_options=options, run_log=runs,
        )
        payload = {
            "actor": actor,
            "input": run_input,
            "run_options": options,
            "run": {key: runs[-1].get(key) for key in _RUN_SUMMARY},
            "fetched_utc": _utc_stamp(now if now is not None else datetime.now(timezone.utc)),
            "items": items,
        }
        save_json(raw_dir, "calendar", f"forexfactory_{start}_{end}", payload)
    return normalize_calendar(payload["items"], ingested_utc=payload["fetched_utc"])


def _normalize_item(item: Mapping[str, Any], ingested: str) -> dict[str, Any] | None:
    """One EVENT_COLUMNS row for a scraped item, or None when the item is skipped."""
    impact = _text(item.get("impact")).lower()
    title = _text(item.get("title")).strip()
    if impact == "holiday" or not title:
        return None
    if _missing(item.get("datetimeISO")) and _missing(item.get("dateline")):
        return None
    label = item.get("timeLabel")
    if isinstance(label, str) and label.strip() and not any(c.isdigit() for c in label):
        return None
    release = _release_time(item.get("datetimeISO"), item.get("dateline"))
    release_utc = release.strftime(_UTC_FORMAT)
    currency = _text(item.get("currency"))
    if _missing(item.get("id")):
        event_id = make_event_id(SOURCE, currency, title, release_utc)
    else:
        event_id = f"ff{item['id']}"
    parsed = {field: parse_value(item.get(field)) for field in _VALUE_FIELDS}
    row: dict[str, Any] = {
        "event_id": event_id,
        "source": SOURCE,
        "currency": currency,
        "event_name": title,
        "category": categorize_event(title),
        "impact": impact,
        "release_utc": release_utc,
        "release_et": release.tz_convert("America/New_York").isoformat(),
        "unit": next((unit for _, unit in parsed.values() if unit is not None), None),
        "inverted": int(is_inverted(title)),
        "ingested_utc": ingested,
    }
    for field in _VALUE_FIELDS:
        number = parsed[field][0]
        raw = item.get(field)
        row[field] = math.nan if number is None else float(number)
        row[f"{field}_raw"] = None if _missing(raw) else str(raw)
    return row


def _release_time(iso: Any, dateline: Any) -> pd.Timestamp:
    """The release instant in UTC: datetimeISO when present, else the Unix-seconds dateline."""
    if not _missing(iso):
        stamp = pd.Timestamp(iso)
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("UTC")
        return stamp.tz_convert("UTC")
    return pd.Timestamp(int(dateline), unit="s", tz="UTC")


# Run fields kept in the snapshot, to audit what each calendar fetch cost.
_RUN_SUMMARY = (
    "id", "status", "startedAt", "finishedAt", "usageTotalUsd", "chargedEventCounts",
    "pricingInfo", "options",
)


def _run_data(body: Any) -> dict[str, Any]:
    """The run object of an Apify response, checked for the fields run_actor reads."""
    run = body.get("data") if isinstance(body, dict) else None
    if not isinstance(run, dict) or any(field not in run for field in _RUN_FIELDS):
        raise IngestError("Apify response has no run data")
    return run


def _utc_stamp(moment: datetime) -> str:
    """An aware datetime as YYYY-MM-DDTHH:MM:SSZ in UTC."""
    return moment.astimezone(timezone.utc).strftime(_UTC_FORMAT)


def _text(value: Any) -> str:
    """The value as text, with None as the empty string."""
    return "" if value is None else str(value)


def _missing(value: Any) -> bool:
    """True for None or an empty string."""
    return value is None or value == ""
