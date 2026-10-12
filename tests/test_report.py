"""Tests for macro_agent.report (docs/design.md section 10)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from macro_agent.report import (
    fmt,
    md_table,
    render_evaluation,
    render_outlook,
    write_text,
)

RANK_IC_LINE = (
    "Rank IC: mean per-slot Spearman correlation between predicted and realised "
    "net return across strategies. Hit rate: share of predictions with the right "
    "sign (classifiers: probability above 0.5 matches the outcome). AUC and Brier "
    "score apply to the win classifiers."
)
SHARPE_LINE = (
    "Sharpe interval: 90% moving-block bootstrap (blocks of 5 sessions). Metrics "
    "use daily calendar-time returns, 252 sessions a year, risk-free rate 0."
)
GENERATED_EVALUATION_LINE = (
    "_Generated 2026-10-12T09:00:00Z. Out-of-sample results from purged "
    "walk-forward validation (docs/design.md section 7)._"
)
EVALUATION_HEADINGS = [
    "# Evaluation report",
    "## Data coverage",
    "### Prices",
    "## Walk-forward set-up",
    "## Predictive accuracy",
    "## Strategy performance",
    "## Best strategies out of sample",
    "## Figures",
    "## Limitations",
]


# fmt


@pytest.mark.parametrize(
    ("value", "kind", "expected"),
    [
        (0.1234, "pct", "12.3%"),
        (-0.1234, "pct", "-12.3%"),
        (0.00123, "pct2", "0.12%"),
        (1.234, "num2", "1.23"),
        (0.1234, "num3", "0.123"),
        (1234, "int", "1,234"),
        (1234.0, "int", "1,234"),
        ("abc", "text", "abc"),
        (42, "text", "42"),
    ],
)
def test_fmt_formats_each_kind(value: object, kind: str, expected: str) -> None:
    assert fmt(value, kind) == expected


def test_fmt_defaults_to_text_kind() -> None:
    assert fmt(0.5) == "0.5"


@pytest.mark.parametrize("kind", ["text", "pct", "pct2", "num2", "num3", "int"])
@pytest.mark.parametrize("missing", [None, float("nan"), np.float32("nan"), pd.NA])
def test_fmt_missing_values_are_na(missing: object, kind: str) -> None:
    assert fmt(missing, kind) == "n/a"


def test_fmt_shows_non_numeric_value_unchanged() -> None:
    assert fmt("tbd", "pct") == "tbd"


def test_fmt_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError):
        fmt(1.0, "percent")


# md_table


def test_md_table_header_rule_and_formatted_rows() -> None:
    df = pd.DataFrame({"name": ["a", "b"], "sharpe": [1.234, 0.5]})
    assert md_table(df, {"sharpe": "num2"}).splitlines() == [
        "| name | sharpe |",
        "| --- | --- |",
        "| a | 1.23 |",
        "| b | 0.50 |",
    ]


def test_md_table_without_formats_uses_text() -> None:
    df = pd.DataFrame({"x": [0.1234]})
    assert md_table(df).splitlines()[-1] == "| 0.1234 |"


def test_md_table_missing_cell_is_na() -> None:
    df = pd.DataFrame({"sharpe": [np.nan]})
    assert md_table(df, {"sharpe": "num2"}).splitlines()[-1] == "| n/a |"


def test_md_table_escapes_pipes_and_newlines() -> None:
    df = pd.DataFrame({"note": ["a|b\nc"]})
    assert md_table(df).splitlines()[-1] == "| a\\|b c |"


def test_md_table_empty_frame() -> None:
    assert md_table(pd.DataFrame({"x": []})) == "_No rows._"


# render_evaluation


def _evaluation_inputs(**overrides: Any) -> dict[str, Any]:
    inputs: dict[str, Any] = {
        "generated_utc": "2026-10-12T09:00:00Z",
        "coverage": {"Prices": pd.DataFrame({"symbol": ["SPY"], "n": [1234]})},
        "setup": {"Train window": "252 sessions", "Embargo": "5 sessions"},
        "predictive": pd.DataFrame(
            {"model": ["ridge"], "rank_ic": [0.0456], "hit_rate": [0.512]}
        ),
        "strategies": pd.DataFrame(
            {"strategy": ["momentum"], "sharpe": [1.234], "max_drawdown": [-0.1234]}
        ),
        "top_strategies": pd.DataFrame({"strategy": ["momentum"], "sharpe": [1.234]}),
        "figures": [("Equity curve", "figures/equity.png")],
        "limitations": ["Short sample."],
    }
    inputs.update(overrides)
    return inputs


def test_render_evaluation_headings_in_order() -> None:
    text = render_evaluation(**_evaluation_inputs())
    headings = [line for line in text.splitlines() if line.startswith("#")]
    assert headings == EVALUATION_HEADINGS


def test_render_evaluation_fixed_lines() -> None:
    lines = render_evaluation(**_evaluation_inputs()).splitlines()
    assert GENERATED_EVALUATION_LINE in lines
    assert "- **Train window:** 252 sessions" in lines
    assert RANK_IC_LINE in lines
    assert SHARPE_LINE in lines


def test_render_evaluation_formats_values() -> None:
    lines = render_evaluation(**_evaluation_inputs()).splitlines()
    assert "| SPY | 1,234 |" in lines
    assert "| ridge | 0.046 | 51.2% |" in lines
    assert "| momentum | 1.23 | -12.3% |" in lines
    assert "| momentum | 1.23 |" in lines


def test_render_evaluation_includes_figure_line() -> None:
    lines = render_evaluation(**_evaluation_inputs()).splitlines()
    assert "![Equity curve](figures/equity.png)" in lines


def test_render_evaluation_omits_notes_when_empty() -> None:
    text = render_evaluation(**_evaluation_inputs(notes=()))
    assert "## Notes" not in text


def test_render_evaluation_places_notes_before_limitations() -> None:
    text = render_evaluation(**_evaluation_inputs(notes=["Fold 3 is short."]))
    headings = [line for line in text.splitlines() if line.startswith("#")]
    assert headings == [*EVALUATION_HEADINGS[:-1], "## Notes", "## Limitations"]
    assert "- Fold 3 is short." in text.splitlines()


def test_render_evaluation_spacing_and_final_newline() -> None:
    text = render_evaluation(**_evaluation_inputs())
    assert "\n\n\n" not in text
    assert text.endswith("\n")
    assert not text.endswith("\n\n")


# render_outlook


def _outlook_inputs(**overrides: Any) -> dict[str, Any]:
    inputs: dict[str, Any] = {
        "generated_utc": "2026-10-12T09:00:00Z",
        "as_of": "2026-10-12",
        "latest": pd.DataFrame({"release": ["CPI"], "surprise_z": [1.234]}),
        "sentiment": pd.DataFrame(
            {"week": ["2026-10-05"], "sentiment": [0.25], "n": [7]}
        ),
        "sentiment_summary": ["Tone turned less negative."],
        "headlines": pd.DataFrame(
            {
                "title": ["Fed holds [again]"],
                "url": ["https://example.com/a"],
                "published": ["2026-10-11"],
                "theme": ["rates"],
            }
        ),
        "upcoming": pd.DataFrame({"event": ["NFP"], "date": ["2026-10-16"]}),
        "playbook": pd.DataFrame(
            {"regime": ["hawkish"], "win_prob": [0.55], "pred": [0.0123]}
        ),
        "playbook_note": "Conditional on the regime label.",
    }
    inputs.update(overrides)
    return inputs


def test_render_outlook_headings_and_header_lines() -> None:
    text = render_outlook(**_outlook_inputs())
    headings = [line for line in text.splitlines() if line.startswith("#")]
    assert headings == [
        "# Macro outlook, 2026-10-12",
        "## Latest releases and surprises",
        "## Sentiment trend",
        "## Recent headlines",
        "## Upcoming events",
        "## Conditional playbook",
    ]
    assert text.startswith(
        "# Macro outlook, 2026-10-12\n\n_Generated 2026-10-12T09:00:00Z._\n\n"
    )
    assert text.endswith("\n")
    assert not text.endswith("\n\n")


def test_render_outlook_formats_tables_and_bullets() -> None:
    lines = render_outlook(**_outlook_inputs()).splitlines()
    assert "| CPI | 1.23 |" in lines
    assert "- Tone turned less negative." in lines
    assert "| 2026-10-05 | 0.250 | 7 |" in lines
    assert "| NFP | 2026-10-16 |" in lines
    assert "Conditional on the regime label." in lines
    assert "| hawkish | 55.0% | 1.23% |" in lines


def test_render_outlook_headline_line_escapes_brackets() -> None:
    lines = render_outlook(**_outlook_inputs()).splitlines()
    assert (
        "1. [Fed holds \\[again\\]](https://example.com/a), 2026-10-11 (rates)"
        in lines
    )


def test_render_outlook_empty_headlines() -> None:
    empty = pd.DataFrame(columns=["title", "url", "published", "theme"])
    text = render_outlook(**_outlook_inputs(headlines=empty))
    assert "## Recent headlines\n\n_No recent headlines._\n\n" in text


# write_text


def test_write_text_creates_folders_and_writes_utf8(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "reports" / "outlook.md"
    returned = write_text(target, "# Título\n")
    assert isinstance(returned, Path)
    assert returned == target
    assert target.read_text(encoding="utf-8") == "# Título\n"


def test_write_text_accepts_string_path(tmp_path: Path) -> None:
    returned = write_text(str(tmp_path / "sub" / "a.md"), "x\n")
    assert isinstance(returned, Path)
    assert returned.read_text(encoding="utf-8") == "x\n"
