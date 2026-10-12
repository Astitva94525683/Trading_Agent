"""Tests for macro_agent.charts (docs/design.md section 10), run on synthetic inputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from macro_agent.charts import (
    equity_curves,
    sentiment_heatmap,
    sentiment_timeline,
    strategy_matrix,
    surprise_heatmap,
)
from macro_agent.config import ASSETS, CATEGORIES, RULES

PNG_MAGIC = b"\x89PNG"
THEMES = ("all", "fed", "inflation", "labor", "growth", "consumer")
WEEKLY_COLUMNS = ["period_end", "theme", "n", "sentiment", "macro_tone"]
EVENT_COLUMNS = ["category", "release_utc", "surprise_z", "weight"]


def assert_png(path: Path) -> None:
    """Check that the file exists and is a PNG large enough to hold a real figure."""
    assert path.is_file()
    data = path.read_bytes()
    assert data.startswith(PNG_MAGIC)
    assert len(data) > 1000


def daily_returns(seed: int, days: int = 120) -> pd.Series:
    """Synthetic daily returns on business days, indexed by date."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2026-05-01", periods=days)
    return pd.Series(rng.normal(0.0003, 0.01, days), index=index)


@pytest.fixture
def weekly() -> pd.DataFrame:
    """Eight weeks of sentiment by theme, with an 'all' row per week and one NaN theme."""
    rng = np.random.default_rng(7)
    rows = [
        {
            "period_end": week.date(),
            "theme": theme,
            "n": int(rng.integers(1, 9)),
            "sentiment": float(rng.uniform(-1.0, 1.0)),
            "macro_tone": float(rng.normal(0.0, 0.5)),
        }
        for week in pd.date_range("2026-05-08", periods=8, freq="W-FRI")
        for theme in THEMES
    ]
    frame = pd.DataFrame(rows, columns=WEEKLY_COLUMNS)
    frame.loc[frame["theme"] == "labor", "macro_tone"] = np.nan
    return frame


@pytest.fixture
def events() -> pd.DataFrame:
    """Thirty synthetic releases over about five months; every third one has weight 0.5."""
    rng = np.random.default_rng(11)
    count = 30
    return pd.DataFrame(
        {
            "category": [CATEGORIES[i % len(CATEGORIES)] for i in range(count)],
            "release_utc": pd.date_range("2026-05-01 12:30", periods=count, freq="5D", tz="UTC"),
            "surprise_z": rng.normal(0.0, 1.0, count),
            "weight": [0.5 if i % 3 == 0 else 1.0 for i in range(count)],
        },
        columns=EVENT_COLUMNS,
    )


@pytest.fixture
def strategy_values() -> pd.DataFrame:
    """A synthetic per-trade return for every asset, rule and horizon combination."""
    rng = np.random.default_rng(3)
    rows = [
        {"asset": asset, "rule": rule, "horizon": horizon, "value": float(rng.normal(0.0, 0.002))}
        for asset in ASSETS
        for rule in RULES
        for horizon in (1, 5, 20)
    ]
    return pd.DataFrame(rows)


def test_sentiment_heatmap_writes_png(weekly, tmp_path):
    path = sentiment_heatmap(weekly, tmp_path / "reports" / "tone.png")
    assert path == tmp_path / "reports" / "tone.png"
    assert_png(path)


def test_sentiment_heatmap_other_value(weekly, tmp_path):
    assert_png(sentiment_heatmap(weekly, tmp_path / "sentiment.png", value="sentiment"))


def test_sentiment_heatmap_empty_writes_no_data(tmp_path):
    empty = pd.DataFrame(columns=WEEKLY_COLUMNS)
    assert_png(sentiment_heatmap(empty, tmp_path / "empty.png"))


def test_sentiment_heatmap_all_row_only_writes_no_data(weekly, tmp_path):
    all_only = weekly.loc[weekly["theme"] == "all"]
    assert_png(sentiment_heatmap(all_only, tmp_path / "all_only.png"))


def test_surprise_heatmap_writes_png(events, tmp_path):
    assert_png(surprise_heatmap(events, tmp_path / "surprise.png"))


def test_surprise_heatmap_accepts_iso_strings(events, tmp_path):
    as_text = events.assign(release_utc=events["release_utc"].astype(str))
    assert_png(surprise_heatmap(as_text, tmp_path / "iso.png"))


def test_surprise_heatmap_many_cells_skips_annotations(tmp_path):
    """Two years of releases gives more than 120 cells, so the figure is drawn without numbers."""
    rng = np.random.default_rng(5)
    count = 200
    long_run = pd.DataFrame(
        {
            "category": [CATEGORIES[i % len(CATEGORIES)] for i in range(count)],
            "release_utc": pd.date_range("2025-01-03 12:30", periods=count, freq="4D", tz="UTC"),
            "surprise_z": rng.normal(0.0, 1.0, count),
        }
    )
    assert_png(surprise_heatmap(long_run, tmp_path / "long.png"))


def test_surprise_heatmap_empty_writes_no_data(tmp_path):
    empty = pd.DataFrame(columns=["category", "release_utc", "surprise_z"])
    assert_png(surprise_heatmap(empty, tmp_path / "empty.png"))


def test_sentiment_timeline_writes_png(weekly, events, tmp_path):
    assert_png(sentiment_timeline(weekly, events, tmp_path / "timeline.png"))


def test_sentiment_timeline_single_line_without_weight_column(weekly, events, tmp_path):
    """All-NaN sentiment leaves one line, and a frame without weight keeps every release."""
    path = sentiment_timeline(
        weekly.assign(sentiment=np.nan), events.drop(columns="weight"), tmp_path / "single.png"
    )
    assert_png(path)


def test_sentiment_timeline_empty_writes_no_data(tmp_path):
    weekly_empty = pd.DataFrame(columns=WEEKLY_COLUMNS)
    events_empty = pd.DataFrame(columns=EVENT_COLUMNS)
    assert_png(sentiment_timeline(weekly_empty, events_empty, tmp_path / "empty.png"))


def test_strategy_matrix_writes_png(strategy_values, tmp_path):
    assert_png(strategy_matrix(strategy_values, tmp_path / "matrix.png"))


def test_strategy_matrix_handles_nan(strategy_values, tmp_path):
    values = strategy_values.copy()
    values.loc[0, "value"] = np.nan
    values = values.iloc[:-1]  # one missing asset-rule-horizon combination, drawn as n/a
    assert_png(strategy_matrix(values, tmp_path / "nan.png"))


def test_strategy_matrix_empty_writes_no_data(tmp_path):
    empty = pd.DataFrame(columns=["asset", "rule", "horizon", "value"])
    assert_png(strategy_matrix(empty, tmp_path / "empty.png"))


def test_equity_curves_writes_png_for_four_series(tmp_path):
    curves = {
        "long": daily_returns(1),
        "short": daily_returns(2),
        "long_on_beat": daily_returns(3),
        "short_on_beat": daily_returns(4),
    }
    assert_png(equity_curves(curves, tmp_path / "nested" / "equity.png"))


def test_equity_curves_rejects_five_series(tmp_path):
    curves = {f"s{i}": daily_returns(i) for i in range(5)}
    with pytest.raises(ValueError, match="at most 4"):
        equity_curves(curves, tmp_path / "five.png")
    assert not (tmp_path / "five.png").exists()


def test_equity_curves_empty_writes_no_data(tmp_path):
    assert_png(equity_curves({}, tmp_path / "empty.png"))
