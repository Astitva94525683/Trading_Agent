"""Offline end-to-end check of the command-line run (`python -m macro_agent run --offline`).

Synthetic raw snapshots are written in the formats the fetchers read, so no network is used. The
command runs once for as_of 2026-07-31 and the tests check its exit code, the reports, the
figures and the predictions table. See docs/design.md sections 2 (pipeline) and 10 (outputs).
"""
from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from macro_agent.__main__ import main
from macro_agent.config import (
    ASSETS,
    ET,
    FOMC_DECISION_DATES,
    NEWS_THEMES,
    VIX_SYMBOL,
    Settings,
)
from macro_agent.ingest.calendar import calendar_input
from macro_agent.ingest.news import week_windows

AS_OF = date(2026, 7, 31)
DEFAULTS = Settings()
WINDOW_START = AS_OF - timedelta(days=DEFAULTS.lookback_days)
WINDOW_END = AS_OF + timedelta(days=DEFAULTS.lookahead_days)
# as_of is not today, so the cut-off is 23:59:59 New York time on AS_OF.
CUTOFF = datetime(2026, 7, 31, 23, 59, 59, tzinfo=ET)
FETCHED = "2026-07-31T23:30:00Z"
CPI_DAYS = (
    date(2026, 2, 11), date(2026, 3, 11), date(2026, 4, 10), date(2026, 5, 12),
    date(2026, 6, 10), date(2026, 7, 14), date(2026, 8, 12),
)
CONSENSUS = {
    "Unemployment Claims": ("230K", "228K"),
    "CPI m/m": ("0.3%", "0.2%"),
    "Federal Funds Rate": ("3.75%", "3.75%"),
    "Non-Farm Employment Change": ("120K", "110K"),
}
FIGURES = (
    "sentiment_heatmap.png",
    "surprise_heatmap.png",
    "sentiment_timeline.png",
    "strategy_matrix.png",
    "equity_drawdown.png",
)
HEADLINES = {
    "fed": "Fed holds rates steady and signals patience",
    "inflation": "Consumer prices rise more than expected",
    "labor": "Job growth slows as unemployment edges up",
    "growth": "Retail sales beat forecasts as growth holds up",
}
TEXTS = {
    "fed": "The Federal Reserve held its policy rate steady and signaled patience while "
    "inflation stays above target.",
    "inflation": "Consumer prices rose faster than expected, keeping inflation pressure on the "
    "Federal Reserve and bond yields.",
    "labor": "Employers added fewer jobs than forecast and the unemployment rate edged higher, "
    "a sign of a cooling labor market.",
    "growth": "Retail sales and industrial output beat forecasts, pointing to resilient growth "
    "despite recession worries.",
}


def _new_york(day: date, hour: int, minute: int) -> datetime:
    """A New York wall-clock time as an aware UTC datetime (08:30 ET in summer is 12:30Z)."""
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=ET).astimezone(timezone.utc)


def _schedule() -> list[tuple[datetime, str, str]]:
    """(release time in UTC, title, impact) for the window's events, in time order."""
    rows: list[tuple[datetime, str, str]] = []
    thursday = WINDOW_START + timedelta(days=(3 - WINDOW_START.weekday()) % 7)
    while thursday <= WINDOW_END:
        rows.append((_new_york(thursday, 8, 30), "Unemployment Claims", "medium"))
        thursday += timedelta(days=7)
    rows += [(_new_york(day, 8, 30), "CPI m/m", "high") for day in CPI_DAYS]
    rows += [
        (_new_york(day, 14, 0), "Federal Funds Rate", "high")
        for day in FOMC_DECISION_DATES
        if WINDOW_START <= day <= WINDOW_END
    ]
    for month in range(WINDOW_START.month, WINDOW_END.month + 1):
        first_friday = date(WINDOW_START.year, month, 1)
        first_friday += timedelta(days=(4 - first_friday.weekday()) % 7)
        rows.append((_new_york(first_friday, 8, 30), "Non-Farm Employment Change", "high"))
    return sorted(rows)


def _actual(title: str, number: int) -> str:
    """A synthetic actual value in the calendar's text format."""
    if title == "Unemployment Claims":
        return f"{225 + (number * 7) % 15}K"
    if title == "CPI m/m":
        return f"0.{2 + number % 3}%"
    if title == "Federal Funds Rate":
        return "3.75%"
    return f"{100 + (number * 37) % 60}K"


def _calendar_items() -> list[dict[str, Any]]:
    """Calendar items in the actor's output format. Actuals are null after the cut-off."""
    items: list[dict[str, Any]] = []
    for number, (when, title, impact) in enumerate(_schedule(), start=1):
        forecast, previous = CONSENSUS[title]
        items.append({
            "id": 100000 + number,
            "title": title,
            "prefixedName": f"US {title}",
            "currency": "USD",
            "country": "US",
            "dateline": int(when.timestamp()),
            "datetimeISO": when.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "timeLabel": when.astimezone(ET).strftime("%I:%M%p").lstrip("0").lower(),
            "impact": impact,
            "impactTitle": f"{impact.capitalize()} Impact Expected",
            "actual": _actual(title, number) if when <= CUTOFF else None,
            "forecast": forecast,
            "previous": previous,
            "revision": "",
            "leaked": False,
        })
    return items


def _price_rows() -> list[list[Any]]:
    """Seeded random-walk daily bars for the assets and VIX on business days."""
    days = pd.bdate_range(WINDOW_START - timedelta(days=DEFAULTS.price_warmup_days), AS_OF)
    rng = np.random.default_rng(20260731)
    rows: list[list[Any]] = []
    for symbol in (*ASSETS, VIX_SYMBOL):
        is_vix = symbol == VIX_SYMBOL
        close = 18.0 if is_vix else 100.0
        for day in days:
            open_ = close * float(np.exp(rng.normal(0, 0.003)))
            close = open_ * float(np.exp(rng.normal(0, 0.04 if is_vix else 0.012)))
            rows.append([
                symbol,
                day.strftime("%Y-%m-%d"),
                round(open_, 4),
                round(max(open_, close) * 1.004, 4),
                round(min(open_, close) * 0.996, 4),
                round(close, 4),
                int(rng.integers(1_000_000, 9_000_000)),
            ])
    return rows


def _write(path: Path, payload: dict[str, Any]) -> None:
    """Save a JSON snapshot, creating its folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def _write_calendar(raw: Path) -> None:
    """ForexFactory calendar snapshot for the window."""
    _write(raw / "calendar" / f"forexfactory_{WINDOW_START}_{WINDOW_END}.json", {
        "actor": DEFAULTS.calendar_actor,
        "input": calendar_input(WINDOW_START, WINDOW_END),
        "fetched_utc": FETCHED,
        "items": _calendar_items(),
    })


def _write_exa(raw: Path) -> None:
    """Exa snapshots: the last four weekly windows of each theme, two articles per window."""
    for theme, query in NEWS_THEMES.items():
        for win_start, win_end in week_windows(WINDOW_START, AS_OF + timedelta(days=1))[-4:]:
            results = [
                {
                    "url": f"https://news.example.com/{theme}/{win_start}/{part}",
                    "title": f"{HEADLINES[theme]} (week of {win_start})",
                    "publishedDate": f"{win_start + timedelta(days=part + 1)}T14:00:00.000Z",
                    "text": TEXTS[theme],
                }
                for part in range(2)
            ]
            _write(raw / "exa" / f"{theme}_{win_start}_{win_end}.json", {
                "theme": theme,
                "query": query,
                "start": win_start.isoformat(),
                "end": win_end.isoformat(),
                "fetched_utc": FETCHED,
                "response": {"results": results},
            })


def _write_tavily(raw: Path) -> None:
    """Tavily snapshots: one per theme on the as-of date, two recent articles each."""
    for theme, query in NEWS_THEMES.items():
        results = [
            {
                "url": f"https://news.example.com/tavily/{theme}/{day}",
                "title": HEADLINES[theme],
                "published_date": f"2026-07-{day}T10:00:00Z",
                "content": TEXTS[theme],
            }
            for day in (27, 29)
        ]
        _write(raw / "tavily" / f"{theme}_{AS_OF}.json", {
            "theme": theme,
            "query": query,
            "fetched_utc": FETCHED,
            "response": {"results": results},
        })


def _write_prices(raw: Path) -> None:
    """Yahoo price snapshot from 60 days before the window to the as-of date."""
    _write(raw / "prices" / f"yahoo_{AS_OF}.json", {
        "symbols": [*ASSETS, VIX_SYMBOL],
        "start": str(WINDOW_START - timedelta(days=DEFAULTS.price_warmup_days)),
        "end": str(AS_OF),
        "fetched_utc": FETCHED,
        "rows": _price_rows(),
    })


def _run_args(data_dir: Path, reports_dir: Path) -> list[str]:
    """Command-line arguments for the offline run on AS_OF."""
    return [
        "run", "--offline",
        "--as-of", AS_OF.isoformat(),
        "--data-dir", str(data_dir),
        "--reports-dir", str(reports_dir),
        "--min-train-slots", "20",
        "--step", "5",
    ]


@pytest.fixture(scope="module")
def cli_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Run the offline command once on the synthetic snapshots and return its results."""
    root = tmp_path_factory.mktemp("cli")
    data_dir = root / "data"
    reports_dir = root / "reports"
    raw = data_dir / "raw"
    _write_calendar(raw)
    _write_exa(raw)
    _write_tavily(raw)
    _write_prices(raw)
    code = main(_run_args(data_dir, reports_dir))
    return {"code": code, "data_dir": data_dir, "reports_dir": reports_dir}


def test_offline_run_returns_zero(cli_run: dict[str, Any]) -> None:
    assert cli_run["code"] == 0


def test_both_reports_are_written(cli_run: dict[str, Any]) -> None:
    reports = cli_run["reports_dir"]
    assert (reports / "evaluation_report.md").is_file()
    assert (reports / "macro_outlook.md").is_file()


def test_all_five_figures_are_written(cli_run: dict[str, Any]) -> None:
    for name in FIGURES:
        assert (cli_run["reports_dir"] / "figures" / name).is_file(), name


def test_evaluation_report_has_strategy_table_and_release_times(cli_run: dict[str, Any]) -> None:
    text = (cli_run["reports_dir"] / "evaluation_report.md").read_text(encoding="utf-8")
    for expected in ("## Strategy performance", "Gradient boosting", "SPY buy and hold", "08:30"):
        assert expected in text, expected


def test_outlook_has_playbook_and_headlines(cli_run: dict[str, Any]) -> None:
    text = (cli_run["reports_dir"] / "macro_outlook.md").read_text(encoding="utf-8")
    for expected in ("## Conditional playbook", "## Recent headlines"):
        assert expected in text, expected


def test_predictions_table_is_written(cli_run: dict[str, Any]) -> None:
    conn = sqlite3.connect(cli_run["data_dir"] / "macro.db")
    try:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    finally:
        conn.close()
    assert "predictions" in {name for (name,) in rows}


def test_empty_data_dir_returns_two(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(_run_args(tmp_path / "data", tmp_path / "reports"))
    assert code == 2
    assert capsys.readouterr().err.startswith("error: ")


def test_evidence_note_flags_a_weak_model() -> None:
    import pandas as pd

    from macro_agent.__main__ import _evidence_note

    predictive = pd.DataFrame({"model": ["ridge", "hgb"], "rank_ic": [0.02, 0.15],
                               "rank_ic_t": [0.4, 3.1]})
    assert "weak evidence" in _evidence_note(predictive, "ridge")
    assert "supported out of sample" in _evidence_note(predictive, "hgb")
    assert _evidence_note(predictive, "baseline") == ""


def test_module_runs_as_a_script(tmp_path) -> None:
    import subprocess
    import sys

    done = subprocess.run(
        [sys.executable, "-m", "macro_agent", "run", "--offline", "--data-dir", str(tmp_path)],
        capture_output=True, text=True, timeout=120,
    )
    assert done.returncode == 2
    assert done.stderr.startswith("error:")
