"""JSON over HTTPS with retries. API keys go in headers, never in URLs or error messages."""
from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

import requests

RETRY_STATUS = frozenset({429, 500, 502, 503, 504})


class IngestError(RuntimeError):
    """A data source failed or returned something unusable."""


def request_json(
    method: str,
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    params: Mapping[str, Any] | None = None,
    json: Any = None,
    timeout: float = 60.0,
    retries: int = 3,
    backoff: float = 2.0,
    session: requests.Session | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Send a request and return the decoded JSON body.

    Connection errors and statuses 429/5xx are retried up to `retries` times, waiting
    backoff * 2**attempt seconds. Other 4xx statuses, a final failure, or a body that is
    not JSON raise IngestError. Messages name the URL without its query string.
    """
    http = session or requests
    where = f"{method} {url.split('?', 1)[0]}"
    for attempt in range(retries + 1):
        last = attempt == retries
        try:
            response = http.request(
                method, url, headers=dict(headers or {}), params=params, json=json, timeout=timeout
            )
        except requests.RequestException as exc:
            if last:
                raise IngestError(f"{where} failed: {type(exc).__name__}") from exc
            sleep(backoff * 2**attempt)
            continue
        if response.status_code in RETRY_STATUS and not last:
            sleep(backoff * 2**attempt)
            continue
        if response.status_code >= 400:
            raise IngestError(f"{where} -> HTTP {response.status_code}: {response.text[:200]}")
        try:
            return response.json()
        except ValueError as exc:
            raise IngestError(f"{where} returned a body that is not JSON") from exc
    raise AssertionError("unreachable")
