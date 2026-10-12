"""Tests for Yahoo price ingestion. Nothing here touches the network."""

from __future__ import annotations

import re
from datetime import date

import numpy as np
import pandas as pd
import pytest

from macro_agent.ingest.http import IngestError
from macro_agent.ingest.prices import (
    PRICE_COLUMNS,
    DownloadFn,
    download_prices,
    fetch_prices,
    normalize_yahoo,
)
from macro_agent.ingest.raw import latest_name, load_json

START = date(2026, 1, 2)
END = date(2026, 1, 6)
TODAY = date(2026, 1, 7)
DATES = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"])
FIELDS = ["Open", "High", "Low", "Close", "Volume"]
NUMERIC = PRICE_COLUMNS[2:]


def _wide(symbols: list[str]) -> pd.DataFrame:
    """A yfinance-style frame: (ticker, field) columns and one row per date."""
    columns = pd.MultiIndex.from_product([symbols, FIELDS])
    values = np.arange(1.0, len(DATES) * len(columns) + 1.0).reshape(len(DATES), len(columns))
    return pd.DataFrame(values, index=DATES, columns=columns)


def _fake(frame: pd.DataFrame) -> DownloadFn:
    """A download stand-in that ignores its arguments and returns ``frame``."""

    def download(symbols, start, end):
        return frame

    return download


def _all_nan_close() -> pd.DataFrame:
    wide = _wide(["SPY", "TLT"])
    for symbol in ("SPY", "TLT"):
        wide[(symbol, "Close")] = np.nan
    return wide


def test_normalize_gives_one_typed_row_per_symbol_and_date():
    wide = _wide(["TLT", "SPY"])
    out = normalize_yahoo(wide, ["TLT", "SPY"])

    assert list(out.columns) == list(PRICE_COLUMNS)
    assert out["symbol"].tolist() == ["SPY"] * 3 + ["TLT"] * 3  # sorted by symbol
    assert out["date"].tolist() == ["2026-01-02", "2026-01-05", "2026-01-06"] * 2
    assert pd.api.types.is_string_dtype(out["symbol"])
    assert pd.api.types.is_string_dtype(out["date"])
    for column in NUMERIC:
        assert out[column].dtype == np.float64
    tlt = out[out["symbol"] == "TLT"]
    for field, column in zip(FIELDS, NUMERIC):
        np.testing.assert_array_equal(tlt[column].to_numpy(), wide[("TLT", field)].to_numpy())


@pytest.mark.parametrize("bad_close", [np.nan, 0.0])
def test_rows_without_a_finite_positive_close_are_dropped(bad_close):
    wide = _wide(["SPY"])
    wide.loc[DATES[1], ("SPY", "Close")] = bad_close
    out = normalize_yahoo(wide, ["SPY"])
    assert out["date"].tolist() == ["2026-01-02", "2026-01-06"]


def test_nan_volume_becomes_zero():
    wide = _wide(["SPY"])
    wide.loc[DATES[0], ("SPY", "Volume")] = np.nan
    out = normalize_yahoo(wide, ["SPY"])
    assert out["volume"].iloc[0] == 0.0
    assert not out["volume"].isna().any()


def test_symbol_missing_from_wide_is_skipped():
    out = normalize_yahoo(_wide(["SPY"]), ["SPY", "GLD"])
    assert set(out["symbol"]) == {"SPY"}


def test_empty_input_gives_typed_empty_frame():
    out = normalize_yahoo(pd.DataFrame(), ["SPY"])
    assert list(out.columns) == list(PRICE_COLUMNS)
    assert out.empty
    assert out["close"].dtype == np.float64


def test_missing_symbols_warn_once_each_and_are_skipped(tmp_path):
    with pytest.warns(UserWarning) as record:
        out = fetch_prices(
            ["SPY", "GLD", "XLE"],
            START,
            END,
            tmp_path,
            today=TODAY,
            download=_fake(_wide(["SPY"])),
        )
    messages = [str(w.message) for w in record if "no prices" in str(w.message)]
    assert messages == ["Yahoo returned no prices for GLD", "Yahoo returned no prices for XLE"]
    assert set(out["symbol"]) == {"SPY"}


def test_online_snapshot_reloads_offline_as_an_equal_frame(tmp_path):
    symbols = ["TLT", "SPY"]
    online = fetch_prices(
        symbols, START, END, tmp_path, today=TODAY, download=_fake(_wide(symbols))
    )

    assert latest_name(tmp_path, "prices", "yahoo_") == f"yahoo_{TODAY}"
    snapshot = load_json(tmp_path, "prices", f"yahoo_{TODAY}")
    assert snapshot["symbols"] == symbols
    assert snapshot["start"] == "2026-01-02"
    assert snapshot["end"] == "2026-01-06"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", snapshot["fetched_utc"])
    assert len(snapshot["rows"]) == len(online) == 6

    offline = fetch_prices(symbols, START, END, tmp_path, today=TODAY, offline=True)
    pd.testing.assert_frame_equal(offline, online)


def test_offline_without_snapshot_raises(tmp_path):
    with pytest.raises(IngestError, match="no price snapshot"):
        fetch_prices(["SPY"], START, END, tmp_path, today=TODAY, offline=True)


@pytest.mark.parametrize("empty", [pd.DataFrame(), _all_nan_close()], ids=["no-data", "nan-close"])
def test_empty_download_raises_and_saves_nothing(tmp_path, empty):
    with pytest.raises(IngestError, match="no prices"):
        fetch_prices(["SPY", "TLT"], START, END, tmp_path, today=TODAY, download=_fake(empty))
    assert latest_name(tmp_path, "prices", "yahoo_") is None


def test_download_prices_requests_adjusted_bars_through_end(monkeypatch):
    seen: dict = {}

    def fake_download(tickers, **kwargs):
        seen["tickers"] = tickers
        seen.update(kwargs)
        return _wide(list(tickers))

    monkeypatch.setattr("yfinance.download", fake_download)
    download_prices(("TLT", "SPY"), START, END)

    assert seen["tickers"] == ["TLT", "SPY"]
    assert seen["start"] == "2026-01-02"
    assert seen["end"] == "2026-01-07"  # yfinance's end is exclusive, so END + 1 day
    assert seen["auto_adjust"] is True
    assert seen["group_by"] == "ticker"
