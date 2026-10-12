"""Command-line run: `python -m macro_agent run [--offline]`.

Ingests the raw sources for a window around the as-of date, builds the dataset, runs the purged
walk-forward evaluation, writes the figures and the two Markdown reports (evaluation and macro
outlook) and prints a short summary. See docs/design.md section 2 (pipeline) and section 10
(outputs).
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import replace
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from macro_agent import db, pipeline
from macro_agent.charts import (
    equity_curves,
    sentiment_heatmap,
    sentiment_timeline,
    strategy_matrix,
    surprise_heatmap,
)
from macro_agent.config import ET, Settings
from macro_agent.evaluate import (
    BUY_HOLD,
    LONG_SPY,
    MODEL_LABELS,
    Evaluation,
    evaluate,
    top_strategies,
)
from macro_agent.ingest.http import IngestError
from macro_agent.model import CLASSIFIERS, REGRESSORS
from macro_agent.outlook import (
    headlines,
    latest_releases,
    playbook,
    sentiment_trend,
    upcoming_events,
)
from macro_agent.report import render_evaluation, render_outlook, write_text
from macro_agent.sentiment import weekly_sentiment

EVALUATION_REPORT = "evaluation_report.md"
OUTLOOK_REPORT = "macro_outlook.md"
FIGURE_DIR = "figures"
FIGURE_FILES = (
    "sentiment_heatmap.png",
    "surprise_heatmap.png",
    "sentiment_timeline.png",
    "strategy_matrix.png",
    "equity_drawdown.png",
)
FIGURE_CAPTIONS = (
    "Weekly macro tone by news theme",
    "Mean release surprise by category and month",
    "Weekly news sentiment and high-impact release times",
    "Out-of-sample mean net return per trade, by asset and rule",
    "Out-of-sample equity curves and drawdowns",
)
# Event names whose release times are listed in the coverage section (the 08:30 and 14:00 checks).
ANCHOR_EVENTS = ("CPI", "Federal Funds Rate", "Non-Farm Employment Change")
# Model keys (in MODEL_LABELS) that can be the playbook's regressor.
PLAYBOOK_REGRESSORS = ("ridge", "hgb")
PLAYBOOK_CLASSIFIER = "hgb_clf"
SELECTION = (
    "each slot trades the strategy with the highest predicted net return when it is above zero, "
    "otherwise stays flat"
)
PURGING = "a fold trains only on trades that exited before the test slot's entry"
LIMITATIONS = (
    "Small sample: about 180 days gives roughly 100 slots, so out-of-sample results carry wide "
    "confidence intervals.",
    "Daily bars: entry waits for the next open or close, so the immediate reaction to a release "
    "is only partly captured.",
    "Forecasts are the calendar site's consensus, not a survey of record.",
    "Sentiment from the macro tone lexicon and VADER is coarse.",
    "Costs are a flat 2 bps per side; slippage around releases can be larger.",
    "Early market closes are not modelled; a release after a 13:00 ET early close is treated "
    "as in-session.",
)


def run(
    settings: Settings,
    as_of: date,
    *,
    offline: bool = False,
    step: int = 1,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Run the pipeline for `as_of`, write the figures and reports, and return a summary.

    The result holds the ingest row counts, the two report paths, the five figure paths and the
    strategy summary table. Raises IngestError or ValueError when the data cannot be used.
    """
    as_of_utc = _as_of_utc(as_of, now_utc)
    conn = db.connect(settings.db_path)
    try:
        counts = pipeline.ingest(conn, settings, as_of, offline=offline)
        ds = pipeline.build_dataset(conn, settings, as_of)
        ev = evaluate(ds.panel, ds.opens, ds.closes, settings, step=step)
        db.write_table(conn, "predictions", ev.preds)
    finally:
        conn.close()
    figures = _write_figures(settings, ds, ev, as_of_utc)
    evaluation = _write_evaluation(settings, ds, ev, as_of, as_of_utc, step)
    outlook = _write_outlook(settings, ds, ev, as_of, as_of_utc)
    return {
        "counts": counts,
        "reports": [evaluation, outlook],
        "figures": figures,
        "summary": ev.summary,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point. Returns 0 on success and 2 for input or data errors."""
    args = _parser().parse_args(argv)
    settings = _settings(args)
    as_of = args.as_of or datetime.now(ET).date()
    try:
        result = run(settings, as_of, offline=args.offline, step=args.step)
    except (IngestError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _print_summary(result)
    return 0


def _parser() -> argparse.ArgumentParser:
    """The argument parser: a single `run` subcommand."""
    defaults = Settings()
    parser = argparse.ArgumentParser(
        prog="python -m macro_agent", description="Macro news and market trading agent."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run_cmd = commands.add_parser("run", help="ingest, evaluate and write the reports")
    run_cmd.add_argument(
        "--offline", action="store_true", help="read saved raw snapshots instead of the network"
    )
    run_cmd.add_argument(
        "--as-of", type=_day, metavar="YYYY-MM-DD",
        help="as-of date (default: today in New York)",
    )
    run_cmd.add_argument(
        "--data-dir", type=Path, help=f"data folder (default: {defaults.data_dir})"
    )
    run_cmd.add_argument(
        "--reports-dir", type=Path, help=f"report folder (default: {defaults.reports_dir})"
    )
    run_cmd.add_argument(
        "--min-train-slots", type=int,
        help=f"training slots before the first fold (default: {defaults.min_train_slots})",
    )
    run_cmd.add_argument("--step", type=int, default=1, help="walk-forward step in slots")
    return parser


def _day(text: str) -> date:
    """An argparse type for YYYY-MM-DD dates."""
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {text!r}") from exc


def _settings(args: argparse.Namespace) -> Settings:
    """Default settings with the folders and training size given on the command line."""
    changes: dict[str, Any] = {}
    if args.data_dir is not None:
        changes["data_dir"] = args.data_dir
    if args.reports_dir is not None:
        changes["reports_dir"] = args.reports_dir
    if args.min_train_slots is not None:
        changes["min_train_slots"] = args.min_train_slots
    return replace(Settings(), **changes)


def _print_summary(result: dict[str, Any]) -> None:
    """Print the ingest row counts, the report paths and the strategy and Sharpe table."""
    counts = ", ".join(f"{table} {rows}" for table, rows in result["counts"].items())
    print(f"Ingested rows: {counts}")
    evaluation, outlook = result["reports"]
    print(f"Evaluation report: {evaluation}")
    print(f"Macro outlook: {outlook}")
    print(f"Figures: {len(result['figures'])} written to {Path(result['figures'][0]).parent}")
    print(result["summary"][["strategy", "sharpe"]].to_string(index=False))


def _as_of_utc(as_of: date, now_utc: datetime | None) -> pd.Timestamp:
    """The cut-off instant: `now_utc` if given, the current time for today, else 23:59:59 ET."""
    if now_utc is not None:
        return _instant(now_utc)
    if as_of == datetime.now(ET).date():
        return pd.Timestamp(datetime.now(timezone.utc))
    return _instant(datetime.combine(as_of, time(23, 59, 59), tzinfo=ET))


def _instant(moment: datetime) -> pd.Timestamp:
    """An instant as a timezone-aware UTC Timestamp; a naive value is taken as UTC."""
    stamp = pd.Timestamp(moment)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _utc(values: pd.Series) -> pd.Series:
    """Timestamps as timezone-aware UTC, from strings, naive or aware values."""
    return pd.to_datetime(values, utc=True)


def _generated() -> str:
    """The generation time in UTC, for the report headers."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _sharpe_score(summary: pd.DataFrame, label: str) -> float:
    """The label's Sharpe ratio in the summary, or -inf when it is missing or not finite."""
    values = pd.to_numeric(summary.loc[summary["strategy"].eq(label), "sharpe"], errors="coerce")
    score = float(values.iloc[0]) if len(values) else math.nan
    return score if math.isfinite(score) else -math.inf


def _best_label(summary: pd.DataFrame, labels: Iterable[str]) -> str:
    """The label with the highest Sharpe ratio (the first label when none has a finite one)."""
    return max(labels, key=lambda label: _sharpe_score(summary, label))


def _write_figures(
    settings: Settings, ds: pipeline.Dataset, ev: Evaluation, as_of_utc: pd.Timestamp
) -> list[Path]:
    """Write the five figures, in FIGURE_FILES order, and return their paths."""
    folder = settings.reports_dir / FIGURE_DIR
    folder.mkdir(parents=True, exist_ok=True)
    weekly = weekly_sentiment(_known_by(ds.news, "published_utc", as_of_utc))
    released = _known_by(ds.events, "release_utc", as_of_utc)
    surprises = released.loc[released["surprise_z"].notna()]
    return [
        sentiment_heatmap(weekly, folder / FIGURE_FILES[0]),
        surprise_heatmap(surprises, folder / FIGURE_FILES[1]),
        sentiment_timeline(weekly, released, folder / FIGURE_FILES[2]),
        strategy_matrix(ev.strategy_pnl, folder / FIGURE_FILES[3]),
        equity_curves(_equity_inputs(ev), folder / FIGURE_FILES[4]),
    ]


def _equity_inputs(ev: Evaluation) -> dict[str, pd.Series]:
    """Daily returns of the best model, the naive baseline (if different) and the two benchmarks."""
    best = _best_label(ev.summary, MODEL_LABELS.values())
    labels = [best]
    if MODEL_LABELS["baseline"] != best:
        labels.append(MODEL_LABELS["baseline"])
    labels += [LONG_SPY, BUY_HOLD]
    return {label: ev.books[label]["ret"] for label in labels}


def _known_by(frame: pd.DataFrame, column: str, as_of_utc: pd.Timestamp) -> pd.DataFrame:
    """The rows whose timestamp in `column` is at or before the as-of instant."""
    return frame.loc[_utc(frame[column]).le(as_of_utc)]


def _write_evaluation(
    settings: Settings,
    ds: pipeline.Dataset,
    ev: Evaluation,
    as_of: date,
    as_of_utc: pd.Timestamp,
    step: int,
) -> Path:
    """Render the out-of-sample evaluation report and write it to the reports folder."""
    start, end = pipeline.window(settings, as_of)
    news_after = int(_utc(ds.news["published_utc"]).gt(as_of_utc).sum())
    no_actual = int(ds.events["actual"].isna().sum())
    text = render_evaluation(
        generated_utc=_generated(),
        coverage={
            "Events by category": _event_categories(ds.events),
            "Release-time anchors": _release_anchors(ds.events),
            "News by provider and theme": _news_coverage(ds.news),
            "Prices by symbol": _price_coverage(ds.prices),
            "Slots and trades": _slot_counts(ds, ev),
        },
        setup=_setup(settings, ds, ev, as_of, (start, end), step),
        predictive=ev.predictive,
        strategies=ev.summary,
        top_strategies=top_strategies(ev.strategy_pnl),
        figures=[
            (caption, f"{FIGURE_DIR}/{name}")
            for caption, name in zip(FIGURE_CAPTIONS, FIGURE_FILES)
        ],
        limitations=LIMITATIONS,
        findings=_key_findings(ev),
        notes=(
            f"Articles published after the as-of time: {news_after}",
            "Events in the window without an actual value (upcoming or speeches): "
            f"{no_actual}",
        ),
    )
    return write_text(settings.reports_dir / EVALUATION_REPORT, text)


def _key_findings(ev: Evaluation) -> list[str]:
    """Headline results for the top of the evaluation report."""
    summary = ev.summary.set_index("strategy")
    sharpe = pd.to_numeric(summary["sharpe"], errors="coerce")
    findings = [
        f"Out of sample: {len(ev.oos_slots)} release slots from {ev.oos_start} to "
        f"{ev.oos_end}, each traded at the first price after the release, with "
        "2 bps of costs per side."
    ]
    if sharpe.notna().any():
        best = sharpe.idxmax()
        row = summary.loc[best]
        findings.append(
            f"Best Sharpe ratio: {best}, {row['sharpe']:.2f} (90% interval "
            f"{row['sharpe_lo']:.2f} to {row['sharpe_hi']:.2f}), Sortino {row['sortino']:.2f}, "
            f"max drawdown {row['max_drawdown']:.1%}, total return {row['total_return']:.1%}."
        )
        others = [
            f"{label} {value:.2f}" for label, value in sharpe.items()
            if label != best and np.isfinite(value)
        ]
        if others:
            findings.append("Other Sharpe ratios: " + "; ".join(others) + ".")
    regression = ev.predictive.loc[ev.predictive["target"].eq("net_ret")]
    ranked = [
        f"{row['model']} {row['rank_ic']:.3f} (t = {row['rank_ic_t']:.2f})"
        for _, row in regression.iterrows()
        if np.isfinite(row["rank_ic"]) and np.isfinite(row["rank_ic_t"])
    ]
    if ranked:
        findings.append(
            "Strategy ranking, rank IC: " + "; ".join(ranked)
            + ". Only |t| of 2 or more marks a reliable ranking."
        )
    auc = pd.to_numeric(
        ev.predictive.loc[ev.predictive["target"].eq("win"), "auc"], errors="coerce"
    ).dropna()
    if not auc.empty:
        findings.append(
            f"Win-rate forecasts: AUC {auc.min():.2f} to {auc.max():.2f} (0.50 is a coin flip)."
        )
    lo = pd.to_numeric(summary["sharpe_lo"], errors="coerce")
    hi = pd.to_numeric(summary["sharpe_hi"], errors="coerce")
    significant = [label for label in summary.index if lo[label] > 0 or hi[label] < 0]
    if significant:
        findings.append(
            "Sharpe intervals that exclude zero: " + ", ".join(significant) + "."
        )
    else:
        findings.append(
            "Every 90% Sharpe interval includes zero, so with this sample no strategy's "
            "edge is distinguishable from noise."
        )
    return findings


def _event_categories(events: pd.DataFrame) -> pd.DataFrame:
    """Per category: events, events with an actual, and the first and last New York dates."""
    days = _utc(events["release_utc"]).dt.tz_convert(ET).dt.strftime("%Y-%m-%d")
    frame = pd.DataFrame({
        "category": events["category"].to_numpy(),
        "day": days.to_numpy(),
        "has_actual": events["actual"].notna().to_numpy(),
    })
    return frame.groupby("category", sort=True).agg(
        events=("day", "size"),
        with_actual=("has_actual", "sum"),
        first=("day", "min"),
        last=("day", "max"),
    ).reset_index()


def _release_anchors(events: pd.DataFrame) -> pd.DataFrame:
    """Per anchor event: releases and the distinct New York release times (HH:MM)."""
    pattern = "|".join(re.escape(name) for name in ANCHOR_EVENTS)
    picked = events.loc[events["event_name"].str.contains(pattern, regex=True)]
    times = _utc(picked["release_utc"]).dt.tz_convert(ET).dt.strftime("%H:%M")
    frame = pd.DataFrame({
        "event": picked["event_name"].to_numpy(),
        "time": times.to_numpy(),
    })
    return frame.groupby("event", sort=True).agg(
        releases=("time", "size"),
        times_et=("time", lambda values: ", ".join(sorted(set(values)))),
    ).reset_index()


def _news_coverage(news: pd.DataFrame) -> pd.DataFrame:
    """Per provider and theme: articles and the first and last UTC publication dates."""
    days = _utc(news["published_utc"]).dt.strftime("%Y-%m-%d")
    frame = pd.DataFrame({
        "provider": news["provider"].to_numpy(),
        "theme": news["theme"].to_numpy(),
        "day": days.to_numpy(),
    })
    return frame.groupby(["provider", "theme"], sort=True).agg(
        articles=("day", "size"),
        first=("day", "min"),
        last=("day", "max"),
    ).reset_index()


def _price_coverage(prices: pd.DataFrame) -> pd.DataFrame:
    """Per symbol: sessions and the first and last dates."""
    return prices.groupby("symbol", sort=True).agg(
        sessions=("date", "size"),
        first=("date", "min"),
        last=("date", "max"),
    ).reset_index()


def _slot_counts(ds: pipeline.Dataset, ev: Evaluation) -> pd.DataFrame:
    """One row: slots, slots with a surprise, panel rows and out-of-sample slots."""
    return pd.DataFrame([{
        "slots": len(ds.slots),
        "slots_with_surprise": int(ds.slots["has_surprise"].eq(True).sum()),
        "panel_rows": len(ds.panel),
        "oos_slots": len(ev.oos_slots),
    }])


def _setup(
    settings: Settings,
    ds: pipeline.Dataset,
    ev: Evaluation,
    as_of: date,
    window: tuple[date, date],
    step: int,
) -> dict[str, object]:
    """The walk-forward set-up, as label to value for the report."""
    start, end = window
    folds = ev.preds.loc[ev.preds["model"].eq(REGRESSORS[0]), "fold"].nunique()
    return {
        "As-of date": as_of.isoformat(),
        "Window": f"{start} to {end}",
        "Sessions in the calendar": len(ds.closes),
        "Minimum training slots": settings.min_train_slots,
        "Walk-forward step": step,
        "Out-of-sample period": f"{ev.oos_start} to {ev.oos_end}",
        "Folds": int(folds),
        "Models": f"regressors {', '.join(REGRESSORS)}; classifiers {', '.join(CLASSIFIERS)}",
        "Selection": SELECTION,
        "Costs": f"{settings.cost_bps} bps per side",
        "Purging": PURGING,
    }


def _write_outlook(
    settings: Settings,
    ds: pipeline.Dataset,
    ev: Evaluation,
    as_of: date,
    as_of_utc: pd.Timestamp,
) -> Path:
    """Render the macro outlook page and write it to the reports folder."""
    regressor = max(
        PLAYBOOK_REGRESSORS, key=lambda key: _sharpe_score(ev.summary, MODEL_LABELS[key])
    )
    sentiment, bullets = sentiment_trend(ds.news, as_of_utc)
    table, note = playbook(
        ds, settings, as_of_utc, regressor=regressor, classifier=PLAYBOOK_CLASSIFIER
    )
    note += _evidence_note(ev.predictive, regressor)
    text = render_outlook(
        generated_utc=_generated(),
        as_of=as_of.isoformat(),
        latest=latest_releases(ds.events, as_of_utc),
        sentiment=sentiment,
        sentiment_summary=bullets,
        headlines=headlines(ds.news, as_of_utc),
        upcoming=upcoming_events(ds.events, as_of_utc),
        playbook=table,
        playbook_note=note,
    )
    return write_text(settings.reports_dir / OUTLOOK_REPORT, text)


def _evidence_note(predictive: pd.DataFrame, model: str) -> str:
    """One sentence on how much out-of-sample evidence backs the playbook's model."""
    row = predictive.loc[predictive["model"] == model]
    if row.empty:
        return ""
    ic, t = float(row["rank_ic"].iloc[0]), float(row["rank_ic_t"].iloc[0])
    if not (np.isfinite(ic) and np.isfinite(t)):
        return ""
    strength = "weak evidence" if abs(t) < 2.0 else "supported out of sample"
    return (
        f" This model's out-of-sample rank IC is {ic:.3f} (t = {t:.2f}), so treat the ranking"
        f" as {strength}."
    )


if __name__ == "__main__":
    raise SystemExit(main())
