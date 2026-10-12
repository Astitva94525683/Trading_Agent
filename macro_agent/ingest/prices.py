"""Yahoo Finance daily prices for the ``prices`` table.

Each download is saved as a raw snapshot under ``prices/yahoo_<date>.json`` so a
run can be repeated offline.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from macro_agent.ingest.http import IngestError
from macro_agent.ingest.raw import latest_name, load_json, save_json

PRICE_COLUMNS = ("symbol", "date", "open", "high", "low", "close", "volume")

_DTYPES: dict[str, type | str] = {
    "symbol": str,
    "date": str,
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume": "float64",
}

DownloadFn = Callable[[Sequence[str], date, date], pd.DataFrame]


def download_prices(symbols: Sequence[str], start: date, end: date) -> pd.DataFrame:
    """Download daily bars for ``symbols`` from Yahoo, with ``end`` included.

    Returns yfinance's wide frame, with (ticker, field) columns.
    """
    import yfinance as yf

    return yf.download(
        list(symbols),
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),  # yfinance's end is exclusive
        # auto_adjust=True adjusts open/high/low/close for splits and dividends,
        # so returns include dividends (this matters for TLT).
        auto_adjust=True,
        actions=False,
        group_by="ticker",
        progress=False,
        threads=False,
    )


def normalize_yahoo(wide: pd.DataFrame, symbols: Sequence[str]) -> pd.DataFrame:
    """Reshape a (ticker, field) yfinance frame into one row per (symbol, date).

    Rows whose close is not finite and positive are dropped, NaN volume becomes
    0.0, and symbols missing from ``wide`` are skipped. A None frame (yfinance can
    return one) counts as empty.
    """
    if wide is None or wide.empty:
        return _typed(pd.DataFrame(columns=list(PRICE_COLUMNS)))
    tickers = set(wide.columns.get_level_values(0))
    parts: list[pd.DataFrame] = []
    for symbol in dict.fromkeys(symbols):  # each requested symbol once, in order
        if symbol not in tickers:
            continue
        bars = wide[symbol]
        close = bars["Close"].to_numpy(dtype="float64", na_value=np.nan)
        keep = np.isfinite(close) & (close > 0)
        if not keep.any():
            continue
        arrays = {
            "date": bars.index.strftime("%Y-%m-%d").to_numpy(),
            "open": bars["Open"].to_numpy(dtype="float64", na_value=np.nan),
            "high": bars["High"].to_numpy(dtype="float64", na_value=np.nan),
            "low": bars["Low"].to_numpy(dtype="float64", na_value=np.nan),
            "close": close,
            "volume": bars["Volume"].fillna(0.0).to_numpy(dtype="float64", na_value=np.nan),
        }
        part = pd.DataFrame({name: values[keep] for name, values in arrays.items()})
        part.insert(0, "symbol", symbol)
        parts.append(part)
    if not parts:
        return _typed(pd.DataFrame(columns=list(PRICE_COLUMNS)))
    out = pd.concat(parts, ignore_index=True)
    return _typed(out.sort_values(["symbol", "date"]))


def fetch_prices(
    symbols: Sequence[str],
    start: date,
    end: date,
    raw_dir: str | Path,
    *,
    today: date,
    offline: bool = False,
    download: DownloadFn = download_prices,
) -> pd.DataFrame:
    """Daily prices for ``symbols``, downloaded from Yahoo or read from a snapshot.

    Online, the download is saved as ``prices/yahoo_<today>.json``. Offline, the
    newest snapshot is returned as it was saved, whatever symbols and dates were
    asked for.
    """
    if offline:
        name = latest_name(raw_dir, "prices", "yahoo_")
        if name is None:
            raise IngestError(f"no price snapshot in {raw_dir}; run online first")
        payload = load_json(raw_dir, "prices", name)
        return _typed(pd.DataFrame(payload["rows"], columns=list(PRICE_COLUMNS)))

    frame = normalize_yahoo(download(symbols, start, end), symbols)
    if frame.empty:
        raise IngestError("Yahoo returned no prices")
    found = set(frame["symbol"])
    for symbol in dict.fromkeys(symbols):
        if symbol not in found:
            warnings.warn(f"Yahoo returned no prices for {symbol}", stacklevel=2)
    save_json(
        raw_dir,
        "prices",
        f"yahoo_{today}",
        {
            "symbols": list(symbols),
            "start": str(start),
            "end": str(end),
            "fetched_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "rows": frame.to_dict("records"),
        },
    )
    return frame


def _typed(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep the ``prices`` columns in table order, with the table's dtypes."""
    return frame.loc[:, list(PRICE_COLUMNS)].astype(_DTYPES).reset_index(drop=True)
