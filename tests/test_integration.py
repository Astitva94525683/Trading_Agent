"""End-to-end check of the offline core on synthetic data.

Events → slots → features → labels → panel → purged walk-forward → trade selection →
calendar-time backtest → metrics, with every module wired the way the pipeline uses them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from macro_agent import db
from macro_agent.backtest import portfolio_returns, select_trades
from macro_agent.config import ASSETS, VIX_SYMBOL
from macro_agent.features import (
    build_panel,
    build_slots,
    feature_columns,
    market_features,
    news_features,
    prepare_events,
    surprise_index,
    time_features,
)
from macro_agent.metrics import summarize
from macro_agent.parsing import categorize_event, is_inverted
from macro_agent.sentiment import score_news
from macro_agent.strategies import compute_labels, pivot_prices, strategy_grid
from macro_agent.timeutils import SessionCalendar, to_utc
from macro_agent.walkforward import MeanBaseline, walk_forward

SYMBOLS = (*ASSETS, VIX_SYMBOL)


def _prices(sessions, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    for symbol in SYMBOLS:
        close = 100.0
        for day in sessions:
            open_ = close * (1 + rng.normal(0, 0.003))
            close = open_ * (1 + rng.normal(0, 0.008))
            rows.append({"symbol": symbol, "date": day, "open": open_, "close": close})
    return pd.DataFrame(rows)


def _events(sessions, seed=11):
    rng = np.random.default_rng(seed)
    rows = []
    for i, day in enumerate(sessions[25:-6]):
        if day.weekday() == 3:  # Thursday 08:30 jobless claims
            name, hhmm, impact = "Unemployment Claims", "08:30", "Medium"
        elif day.weekday() == 2 and i % 2 == 0:  # alternate Wednesdays: CPI at 08:30
            name, hhmm, impact = "CPI m/m", "08:30", "High"
        elif day.weekday() == 2:  # other Wednesdays: FOMC at 14:00
            name, hhmm, impact = "Federal Funds Rate", "14:00", "High"
        else:
            continue
        forecast = 0.2 if name != "Unemployment Claims" else 220_000.0
        actual = forecast + rng.normal(0, 0.1 if name != "Unemployment Claims" else 8_000.0)
        rows.append({
            "event_id": f"{name}-{day}",
            "event_name": name,
            "category": categorize_event(name),
            "impact": impact,
            "release_utc": to_utc(f"{day} {hhmm}"),
            "actual": actual,
            "forecast": forecast,
            "inverted": is_inverted(name),
        })
    return pd.DataFrame(rows)


def _news(sessions):
    texts = [
        ("Fed signals rate cuts as inflation cooled", "fed"),
        ("Hotter-than-expected CPI keeps the Fed hawkish", "inflation"),
        ("Payrolls surged and the labor market stayed tight", "labor"),
        ("Recession fears grow as manufacturing contracted", "growth"),
    ]
    rows = []
    for i, day in enumerate(sessions):
        title, theme = texts[i % len(texts)]
        rows.append({
            "published_utc": to_utc(f"{day} 07:00"),
            "title": title,
            "text": "",
            "theme": theme,
        })
    return score_news(pd.DataFrame(rows), use_vader=False)


def test_offline_pipeline_runs_end_to_end_without_look_ahead():
    sessions = list(pd.bdate_range("2026-03-02", "2026-07-31").date)
    calendar = SessionCalendar.from_dates(sessions)
    prices = _prices(sessions)
    opens, closes = pivot_prices(prices)

    events = prepare_events(_events(sessions))
    slots = build_slots(events, calendar)
    fomc = [ts.tz_convert("America/New_York").date()
            for ts in events.loc[events["category"] == "fed", "release_utc"]]
    slots = surprise_index(time_features(slots, fomc))
    slots = news_features(slots, _news(sessions))

    labels = compute_labels(slots, opens, closes, calendar, strategy_grid(), cost_bps=2.0)
    market = market_features(slots, opens, closes, tuple(ASSETS), VIX_SYMBOL)
    panel = build_panel(slots, labels, market)
    assert len(panel) == len(labels) > 0
    assert set(feature_columns(panel)).isdisjoint({"exit_price", "gross_ret", "net_ret", "win"})

    preds = walk_forward(panel, ["strategy_id"], "net_ret", MeanBaseline, min_train_slots=10)
    assert not preds.empty
    # Every prediction was trained only on trades that had exited before its slot's entry.
    entries = panel.set_index("slot_id")["entry_utc"].groupby(level=0).min()
    assert (preds["train_cutoff"] <= preds["slot_id"].map(entries)).all()

    scored = preds.join(panel[["asset", "sign", "entry_kind", "entry_session",
                               "exit_session", "net_ret", "entry_utc"]])
    trades = select_trades(scored, top_k=1, min_pred=0.0)
    assert trades.groupby("slot_id").size().max() == 1

    book = portfolio_returns(trades, opens, closes, cost_bps=2.0)
    stats = summarize(book["ret"], trade_returns=trades["net_ret"], n_active=book["n_active"])
    assert stats["n_trades"] == len(trades)
    assert stats["max_drawdown"] <= 0.0
    assert np.isfinite(stats["total_return"])

    # The database round-trips the raw tables and the derived panel.
    conn = db.connect()
    db.upsert(conn, "prices", prices.assign(high=prices["close"], low=prices["open"], volume=0.0),
              key=["symbol", "date"])
    db.write_table(conn, "panel", panel.drop(columns=["entry_session", "exit_session"]))
    assert db.read_df(conn, "SELECT COUNT(*) AS n FROM panel")["n"].iloc[0] == len(panel)
