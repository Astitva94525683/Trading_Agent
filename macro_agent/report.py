"""Markdown rendering for the evaluation report and the macro outlook.

Layout follows docs/design.md section 10. The functions here only format
results that other modules computed; they compute nothing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path

import pandas as pd

# Scale factor and format template for each numeric kind accepted by fmt().
_NUMBER_FORMATS: dict[str, tuple[float, str]] = {
    "pct": (100.0, "{:.1f}%"),
    "pct2": (100.0, "{:.2f}%"),
    "num2": (1.0, "{:.2f}"),
    "num3": (1.0, "{:.3f}"),
    "int": (1.0, "{:,.0f}"),
}

# Column name -> fmt kind. Applied wherever the column appears in a table.
_FORMATS: dict[str, str] = {
    "sharpe": "num2",
    "sharpe_lo": "num2",
    "sharpe_hi": "num2",
    "sortino": "num2",
    "max_drawdown": "pct",
    "total_return": "pct",
    "cagr": "pct",
    "ann_vol": "pct",
    "exposure": "pct",
    "n_trades": "int",
    "win_rate": "pct",
    "avg_ret": "pct2",
    "avg_return": "pct2",
    "median_return": "pct2",
    "profit_factor": "num2",
    "rank_ic": "num3",
    "rank_ic_t": "num2",
    "hit_rate": "pct",
    "mae": "pct2",
    "auc": "num3",
    "brier": "num3",
    "n_slots": "int",
    "n_preds": "int",
    "n": "int",
    "mean_ret": "pct2",
    "surprise_z": "num2",
    "macro_tone": "num3",
    "sentiment": "num3",
    "pred": "pct2",
    "win_prob": "pct",
}

_RANK_IC_NOTE = (
    "Rank IC: mean per-slot Spearman correlation between predicted and realised "
    "net return across strategies. Hit rate: share of predictions with the right "
    "sign (classifiers: probability above 0.5 matches the outcome). AUC and Brier "
    "score apply to the win classifiers."
)

_SHARPE_NOTE = (
    "Sharpe interval: 90% moving-block bootstrap (blocks of 5 sessions). Metrics "
    "use daily calendar-time returns, 252 sessions a year, risk-free rate 0."
)


def _is_missing(value: object) -> bool:
    """Return True for None, NaN, NaT and pd.NA."""
    if value is None or value is pd.NA:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):  # list-like values are never missing
        return False


def fmt(value: object, kind: str = "text") -> str:
    """Format one value as text.

    Missing values give "n/a". Kinds: pct (12.3%), pct2 (0.12%), num2, num3,
    int (1,234) and text (str(value)). A numeric kind shows a non-numeric
    value unchanged; an unknown kind raises ValueError.
    """
    if kind != "text" and kind not in _NUMBER_FORMATS:
        raise ValueError(f"unknown format kind: {kind!r}")
    if _is_missing(value):
        return "n/a"
    if kind == "text":
        return str(value)
    scale, template = _NUMBER_FORMATS[kind]
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return template.format(number * scale)


def _one_line(text: str) -> str:
    """Replace line breaks with spaces so the text stays on one line."""
    return text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")


def _cell(value: object, kind: str = "text") -> str:
    """Format a table cell, with pipes escaped."""
    return _one_line(fmt(value, kind)).replace("|", "\\|")


def md_table(df: pd.DataFrame, formats: Mapping[str, str] | None = None) -> str:
    """Render a DataFrame as a GitHub Markdown table.

    Columns listed in ``formats`` (column -> fmt kind) are formatted with that
    kind; the other columns use "text". An empty frame renders as "_No rows._".
    """
    if df.empty:
        return "_No rows._"
    kinds = formats or {}
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(_cell(column) for column in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for record in df.itertuples(index=False, name=None):
        cells = [
            _cell(value, kinds.get(column, "text"))
            for column, value in zip(columns, record)
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _bullets(items: Sequence[str]) -> str:
    """Render items as one Markdown bullet per line."""
    return "\n".join(f"- {_one_line(str(item))}" for item in items)


def _headline_list(headlines: pd.DataFrame) -> str:
    """Render headlines as a numbered list; brackets in titles are escaped."""
    if headlines.empty:
        return "_No recent headlines._"
    rows = zip(
        headlines["title"],
        headlines["url"],
        headlines["published"],
        headlines["theme"],
    )
    lines = []
    for number, (title, url, published, theme) in enumerate(rows, start=1):
        label = _one_line(fmt(title)).replace("[", "\\[").replace("]", "\\]")
        lines.append(
            f"{number}. [{label}]({fmt(url)}), {fmt(published)} ({fmt(theme)})"
        )
    return "\n".join(lines)


def _join(blocks: Sequence[str]) -> str:
    """Join non-empty blocks with blank lines; the text ends with one newline."""
    parts = [block.strip() for block in blocks]
    return "\n\n".join(part for part in parts if part) + "\n"


def render_evaluation(
    *,
    generated_utc: str,
    coverage: Mapping[str, pd.DataFrame],
    setup: Mapping[str, object],
    predictive: pd.DataFrame,
    strategies: pd.DataFrame,
    top_strategies: pd.DataFrame,
    figures: Sequence[tuple[str, str]],
    limitations: Sequence[str],
    notes: Sequence[str] = (),
) -> str:
    """Render the out-of-sample evaluation report as Markdown.

    ``figures`` holds (caption, relative path) pairs. The Notes section is
    written only when ``notes`` is non-empty.
    """
    blocks = [
        "# Evaluation report",
        f"_Generated {generated_utc}. Out-of-sample results from purged "
        "walk-forward validation (docs/design.md section 7)._",
        "## Data coverage",
    ]
    for title, table in coverage.items():
        blocks += [f"### {title}", md_table(table, _FORMATS)]
    blocks += [
        "## Walk-forward set-up",
        "\n".join(
            f"- **{key}:** {_one_line(fmt(value))}" for key, value in setup.items()
        ),
        "## Predictive accuracy",
        md_table(predictive, _FORMATS),
        _RANK_IC_NOTE,
        "## Strategy performance",
        md_table(strategies, _FORMATS),
        _SHARPE_NOTE,
        "## Best strategies out of sample",
        md_table(top_strategies, _FORMATS),
        "## Figures",
        "\n\n".join(f"![{caption}]({path})" for caption, path in figures),
    ]
    if notes:
        blocks += ["## Notes", _bullets(notes)]
    blocks += ["## Limitations", _bullets(limitations)]
    return _join(blocks)


def render_outlook(
    *,
    generated_utc: str,
    as_of: date | str,
    latest: pd.DataFrame,
    sentiment: pd.DataFrame,
    sentiment_summary: Sequence[str],
    headlines: pd.DataFrame,
    upcoming: pd.DataFrame,
    playbook: pd.DataFrame,
    playbook_note: str,
) -> str:
    """Render the macro outlook page as Markdown."""
    return _join(
        [
            f"# Macro outlook, {as_of}",
            f"_Generated {generated_utc}._",
            "## Latest releases and surprises",
            md_table(latest, _FORMATS),
            "## Sentiment trend",
            _bullets(sentiment_summary),
            md_table(sentiment, _FORMATS),
            "## Recent headlines",
            _headline_list(headlines),
            "## Upcoming events",
            md_table(upcoming, _FORMATS),
            "## Conditional playbook",
            playbook_note,
            md_table(playbook, _FORMATS),
        ]
    )


def write_text(path: str | Path, text: str) -> Path:
    """Write text as UTF-8, creating parent folders, and return the path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target
