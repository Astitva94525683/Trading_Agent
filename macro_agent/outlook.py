"""Macro outlook for the report: recent and upcoming releases, the news sentiment trend,
headlines, and a conditional playbook for the next key release. See docs/design.md section 10.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from macro_agent.config import ET, VIX_SYMBOL, Settings
from macro_agent.features import (
    build_panel,
    build_slots,
    market_features,
    news_features,
    surprise_index,
    time_features,
)
from macro_agent.model import make_classifier, make_regressor, model_inputs
from macro_agent.pipeline import Dataset
from macro_agent.sentiment import weekly_sentiment
from macro_agent.strategies import position_sign, strategy_grid
from macro_agent.timeutils import SessionCalendar

PLAYBOOK_COLUMNS = ("scenario", "strategy", "strategy_id", "pred", "win_prob")
KEY_CATEGORIES = ("inflation", "labor", "fed", "growth")
SCENARIOS = (("beat", 1.0), ("miss", -1.0))
TOP_STRATEGIES = 3
UPCOMING_MIN_WEIGHT = 0.5
NO_KEY_RELEASE = "No high-impact release is scheduled in the calendar window."


def _utc(values: pd.Series) -> pd.Series:
    """Timestamps as timezone-aware UTC."""
    return pd.to_datetime(values, utc=True)


def _select(frame: pd.DataFrame, names: dict[str, str]) -> pd.DataFrame:
    """Keep the columns named in `names` (source to output name), in that order."""
    return frame.loc[:, list(names)].rename(columns=names).reset_index(drop=True)


def _new_york(stamps: pd.Series) -> tuple[pd.Series, pd.Series]:
    """New York date ("YYYY-MM-DD") and time ("HH:MM") of UTC timestamps."""
    local = stamps.dt.tz_convert(ET)
    return local.dt.strftime("%Y-%m-%d"), local.dt.strftime("%H:%M")


def _text(value: float) -> str:
    """A mean to three decimals, or "n/a" when it is undefined."""
    return f"{value:.3f}" if pd.notna(value) else "n/a"


def latest_releases(events: pd.DataFrame, as_of_utc: pd.Timestamp, n: int = 10) -> pd.DataFrame:
    """The newest n high-impact releases with an actual, released by as_of_utc, newest first."""
    release = _utc(events["release_utc"])
    keep = release.le(as_of_utc) & events["weight"].eq(1.0) & events["actual"].notna()
    picked = events.loc[keep].assign(release=release[keep])
    picked = picked.sort_values("release", ascending=False).head(n)
    day, clock = _new_york(picked["release"])
    picked = picked.assign(date=day, time=clock)
    return _select(picked, {
        "date": "date",
        "time": "time",
        "event_name": "event",
        "actual_raw": "actual",
        "forecast_raw": "forecast",
        "surprise_z": "surprise_z",
    })


def upcoming_events(events: pd.DataFrame, as_of_utc: pd.Timestamp, n: int = 10) -> pd.DataFrame:
    """The first n releases after as_of_utc with a weight of at least 0.5, in time order."""
    release = _utc(events["release_utc"])
    keep = release.gt(as_of_utc) & events["weight"].ge(UPCOMING_MIN_WEIGHT)
    picked = events.loc[keep].assign(release=release[keep])
    picked = picked.sort_values("release").head(n)
    day, clock = _new_york(picked["release"])
    picked = picked.assign(date=day, time=clock)
    return _select(picked, {
        "date": "date",
        "time": "time",
        "event_name": "event",
        "impact": "impact",
        "forecast_raw": "forecast",
        "previous_raw": "previous",
    })


def _trend_line(label: str, column: str, recent: pd.DataFrame, month: pd.DataFrame) -> str:
    """One bullet: the mean of `column` over the last 7 and 30 days, with article counts."""
    mean7 = recent[column].astype("float64").mean()
    mean30 = month[column].astype("float64").mean()
    return (
        f"{label}, last 7 days: {_text(mean7)} from {len(recent)} articles "
        f"(30 days: {_text(mean30)} from {len(month)} articles)."
    )


def sentiment_trend(
    news: pd.DataFrame, as_of_utc: pd.Timestamp, weeks: int = 8
) -> tuple[pd.DataFrame, list[str]]:
    """Weekly sentiment for the last `weeks` weeks, and bullets on the last 7 and 30 days.

    Uses the articles published by as_of_utc. The bullets use the articles in the 7 and 30
    days before as_of_utc. The theme bullet needs at least two themes in the last 7 days.
    """
    stamps = _utc(news["published_utc"])
    known = news.loc[stamps.le(as_of_utc)]
    weekly = weekly_sentiment(known)
    overall = weekly.loc[weekly["theme"].eq("all")].tail(weeks)
    table = _select(overall, {
        "period_end": "week",
        "n": "n",
        "macro_tone": "macro_tone",
        "sentiment": "sentiment",
    })

    recent = news.loc[stamps.ge(as_of_utc - pd.Timedelta(days=7)) & stamps.lt(as_of_utc)]
    month = news.loc[stamps.ge(as_of_utc - pd.Timedelta(days=30)) & stamps.lt(as_of_utc)]
    bullets = [
        _trend_line("Macro tone", "macro_tone", recent, month),
        _trend_line("VADER sentiment", "sentiment", recent, month),
    ]
    tone = recent.assign(tone=recent["macro_tone"].astype("float64"))
    themes = tone.groupby("theme")["tone"].mean().dropna()
    if len(themes) >= 2:
        worst, best = themes.idxmin(), themes.idxmax()
        bullets.append(
            f"Most negative theme this week: {worst} ({themes[worst]:.3f}); "
            f"most positive: {best} ({themes[best]:.3f})."
        )
    return table, bullets


def headlines(news: pd.DataFrame, as_of_utc: pd.Timestamp, n: int = 8) -> pd.DataFrame:
    """The newest n headlines published by as_of_utc: Tavily results, or every provider if none."""
    stamps = _utc(news["published_utc"])
    known = news.assign(stamp=stamps).loc[stamps.le(as_of_utc)]
    tavily = known.loc[known["provider"].eq("tavily")]
    pool = tavily if not tavily.empty else known
    pool = pool.sort_values("stamp", ascending=False).head(n)
    published = pool["stamp"].dt.strftime("%Y-%m-%d %H:%M UTC")
    return _select(pool.assign(published=published), {
        "title": "title",
        "url": "url",
        "published": "published",
        "theme": "theme",
    })


def playbook(
    ds: Dataset,
    settings: Settings,
    as_of_utc: pd.Timestamp,
    regressor: str = "hgb",
    classifier: str = "hgb_clf",
) -> tuple[pd.DataFrame, str]:
    """The best predicted trades for the next key release, if it beats or misses the forecast.

    The regressor (net_ret) and classifier (win) are fitted on the panel trades that exited
    before as_of_utc. The key release's slot is then scored for every strategy under a beat
    (surprise +1) and a miss (-1). Each scenario keeps up to three trades with a positive
    prediction, or a "Stay flat" row. Returns the rows and a note on the setup.
    """
    release = _utc(ds.events["release_utc"])
    is_key = (
        release.gt(as_of_utc)
        & ds.events["weight"].eq(1.0)
        & ds.events["category"].isin(KEY_CATEGORIES)
    )
    if not is_key.any():
        return pd.DataFrame(columns=list(PLAYBOOK_COLUMNS)), NO_KEY_RELEASE
    key = ds.events.loc[is_key].assign(release=release[is_key])
    key = key.sort_values("release").iloc[0]
    key_id = str(key["event_id"])
    key_local = key["release"].tz_convert(ET)

    last_session = ds.calendar.dates[-1]
    weekdays = pd.bdate_range(
        last_session + timedelta(days=1), key_local.date() + timedelta(days=14)
    )
    calendar = SessionCalendar.from_dates([*ds.calendar.dates, *(d.date() for d in weekdays)])

    slots = build_slots(ds.events.loc[release.le(key["release"])], calendar)
    feats = news_features(surprise_index(time_features(slots, ds.fomc_dates)), ds.news)
    holds_key = feats["event_ids"].fillna("").str.split(",").map(lambda ids: key_id in ids)
    hyp_row = feats.loc[holds_key].iloc[:1]
    if hyp_row.empty:
        raise ValueError(f"the key release {key_id} has no slot in the extended calendar")
    hyp = hyp_row.iloc[0]

    numeric, categorical = model_inputs(ds.panel)
    cols = [*numeric, *categorical, "strategy_id"]
    exited = _utc(ds.panel["exit_utc"]).lt(as_of_utc)
    train = ds.panel.loc[exited & ds.panel["net_ret"].notna()]
    reg = make_regressor(regressor, numeric, categorical)
    reg.fit(train[cols], train["net_ret"])
    clf = make_classifier(classifier, numeric, categorical)
    clf.fit(train[cols], train["win"])

    grid = strategy_grid(settings.assets, settings.rules, settings.horizons)
    records: list[dict[str, object]] = []
    for scenario, surprise in SCENARIOS:
        row = hyp_row.assign(surprise_z=surprise, abs_surprise_z=1.0, has_surprise=True)
        market = market_features(row, ds.opens, ds.closes, settings.assets, VIX_SYMBOL)
        labels = pd.DataFrame([
            {
                "slot_id": hyp["slot_id"],
                "strategy_id": strategy.id,
                "asset": strategy.asset,
                "rule": strategy.rule,
                "horizon": strategy.horizon,
                "sign": position_sign(strategy.rule, surprise),
                "entry_kind": hyp["entry_kind"],
                "entry_session": hyp["entry_session"],
                "entry_utc": hyp["entry_utc"],
            }
            for strategy in grid
        ])
        rows = build_panel(row, labels, market)
        for name in cols:
            if name not in rows.columns:
                rows[name] = np.nan
        rows = rows.assign(
            pred=reg.predict(rows[cols]),
            win_prob=clf.predict_proba(rows[cols])[:, 1],
        )
        # A sign of 0 means no position under this scenario, so it cannot be recommended.
        traded = rows.loc[rows["sign"].ne(0)]
        best = (
            traded.loc[traded["pred"].gt(0)]
            .sort_values("pred", ascending=False)
            .head(TOP_STRATEGIES)
        )
        if best.empty:
            records.append({
                "scenario": scenario,
                "strategy": "Stay flat",
                "strategy_id": "",
                "pred": traded["pred"].max(),
                "win_prob": np.nan,
            })
        for item in best.to_dict("records"):
            side = "Long" if item["sign"] > 0 else "Short"
            label = f"{side} {item['asset']} for {item['horizon']} session(s) ({item['rule']})"
            records.append({
                "scenario": scenario,
                "strategy": label,
                "strategy_id": item["strategy_id"],
                "pred": item["pred"],
                "win_prob": item["win_prob"],
            })

    frame = pd.DataFrame(records, columns=list(PLAYBOOK_COLUMNS))
    note = (
        f"Next key release: {key['event_name']} on {key_local.strftime('%Y-%m-%d')} "
        f"at {key_local.strftime('%H:%M')} ET; the trade would enter at the "
        f"{hyp['entry_kind']} of {hyp['entry_session']}. Beat sets the release's surprise to +1 "
        "(stronger or hotter than forecast) and miss to -1. Predicted net return from the "
        f"{regressor} model and win probability from the {classifier} model, both fitted on all "
        f"{len(train)} realised trades. Sessions after {last_session} assume weekdays without "
        "exchange holidays."
    )
    return frame, note
