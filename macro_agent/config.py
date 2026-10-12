"""Project settings and shared constants.

Design decisions behind these values are explained in docs/design.md.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
ET = ZoneInfo("America/New_York")

# One liquid ETF per macro asset class. These are the tradable assets.
ASSETS = {
    "SPY": "US equities (S&P 500)",
    "TLT": "US Treasuries, 20y+",
    "GLD": "Gold",
    "UUP": "US dollar index",
}
# Price series used only as model features.
VIX_SYMBOL = "^VIX"

# Direction rules. A "beat" is a release that came in stronger or hotter than
# forecast, oriented so that higher unemployment or jobless claims count as a miss.
RULES = ("long", "short", "long_on_beat", "short_on_beat")
# Holding periods, in trading sessions held. An open entry counts its own session.
HORIZONS = (1, 3, 5)
COST_BPS_PER_SIDE = 2.0

# FOMC rate decisions (the second day of each scheduled meeting), from the Federal Reserve's
# published calendar (federalreserve.gov/monetarypolicy/fomccalendars.htm). The schedule is
# public well in advance, so using future dates as features is not look-ahead.
FOMC_DECISION_DATES = tuple(date.fromisoformat(d) for d in (
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
    "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
    "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
    "2027-01-27", "2027-03-17", "2027-04-28", "2027-06-09",
    "2027-07-28", "2027-09-15", "2027-10-27", "2027-12-08",
))

# Event categories in priority order (used to pick a slot's primary category).
CATEGORIES = ("fed", "inflation", "labor", "growth", "consumer", "housing", "other")

# News themes and the search query used for each (Exa history, Tavily recent).
NEWS_THEMES = {
    "fed": "Federal Reserve FOMC interest rate decision Powell",
    "inflation": "US inflation CPI PCE consumer prices",
    "labor": "US jobs report nonfarm payrolls unemployment labor market",
    "growth": "US economy GDP growth recession retail sales ISM",
}


@dataclass(frozen=True)
class Settings:
    lookback_days: int = 180
    lookahead_days: int = 30
    currencies: tuple[str, ...] = ("USD",)
    min_impact_weight: float = 0.5  # keep high (1.0) and medium (0.5) impact events
    assets: tuple[str, ...] = tuple(ASSETS)
    rules: tuple[str, ...] = RULES
    horizons: tuple[int, ...] = HORIZONS
    cost_bps: float = COST_BPS_PER_SIDE
    min_train_slots: int = 30
    news_results: int = 10  # results per Exa or Tavily query
    price_warmup_days: int = 60  # price history before the window, for market features
    data_dir: Path = ROOT / "data"
    reports_dir: Path = ROOT / "reports"
    # Apify Store actor for the economic calendar; confirmed against real output in Phase B.
    calendar_actor: str = "xtracto/forexfactory-calendar"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "macro.db"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"


def api_key(name: str) -> str | None:
    """Read an API key (APIFY_TOKEN, TAVILY_API_KEY, EXA_API_KEY) from the
    environment, falling back to a git-ignored .env file in the repo root."""
    value = os.environ.get(name)
    if value:
        return value
    try:
        from dotenv import dotenv_values
    except ImportError:
        return None
    return dotenv_values(ROOT / ".env").get(name) or None
