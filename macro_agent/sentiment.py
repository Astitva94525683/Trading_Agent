"""News sentiment for the macro agent (design.md section 6).

Each article gets a macro tone from a small phrase lexicon, a theme from keyword
counts, and a VADER compound score. ``weekly_sentiment`` turns scored articles
into a weekly series by theme.
"""
from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd

# Phrase -> direction: +1 stronger, hotter or hawkish; -1 weaker, cooler or dovish.
MACRO_LEXICON: dict[str, int] = {
    "hawkish": 1,
    "rate hike": 1,
    "rate hikes": 1,
    "raise rates": 1,
    "raised rates": 1,
    "tightening": 1,
    "higher for longer": 1,
    "sticky inflation": 1,
    "inflation accelerated": 1,
    "inflation rose": 1,
    "prices rose": 1,
    "hotter than expected": 1,
    "hotter-than-expected": 1,
    "stronger than expected": 1,
    "stronger-than-expected": 1,
    "beat expectations": 1,
    "beat estimates": 1,
    "robust": 1,
    "resilient": 1,
    "strong hiring": 1,
    "payrolls surged": 1,
    "overheating": 1,
    "upside surprise": 1,
    "tight labor market": 1,
    "rebound": 1,
    "boom": 1,
    "dovish": -1,
    "rate cut": -1,
    "rate cuts": -1,
    "cut rates": -1,
    "lower rates": -1,
    "easing": -1,
    "cooling": -1,
    "cooled": -1,
    "slowdown": -1,
    "slowing": -1,
    "recession": -1,
    "layoffs": -1,
    "job cuts": -1,
    "job losses": -1,
    "weaker than expected": -1,
    "weaker-than-expected": -1,
    "softer than expected": -1,
    "softer-than-expected": -1,
    "missed expectations": -1,
    "missed estimates": -1,
    "disinflation": -1,
    "inflation eased": -1,
    "inflation slowed": -1,
    "prices fell": -1,
    "contraction": -1,
    "contracted": -1,
    "unemployment rose": -1,
    "rising unemployment": -1,
    "downside surprise": -1,
    "slack": -1,
    "downturn": -1,
    "stagnation": -1,
    "deflation": -1,
}

# Themes in priority order: ties in classify_theme go to the earlier theme.
THEME_KEYWORDS: dict[str, tuple[str, ...]] = {
    "fed": (
        "fed", "federal reserve", "fomc", "powell", "interest rate", "monetary policy",
        "rate cut", "rate hike",
    ),
    "inflation": ("inflation", "cpi", "pce", "ppi", "prices", "price index"),
    "labor": (
        "jobs", "payrolls", "employment", "unemployment", "labor market", "jobless",
        "hiring", "wages",
    ),
    "growth": (
        "gdp", "growth", "recession", "retail sales", "manufacturing", "pmi", "ism",
        "consumer spending", "economy",
    ),
}


def _phrase_regex(phrase: str) -> str:
    """Regex for one phrase: its words in order, any whitespace between them, whole-word bounds."""
    return r"\b" + r"\s+".join(re.escape(word) for word in phrase.split()) + r"\b"


# Longest phrases first, so a phrase is matched before any shorter phrase inside it.
_LEXICON_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = tuple(
    (re.compile(_phrase_regex(phrase), re.IGNORECASE), sign)
    for phrase, sign in sorted(MACRO_LEXICON.items(), key=lambda item: len(item[0]), reverse=True)
)

_THEME_PATTERNS: dict[str, re.Pattern[str]] = {
    theme: re.compile("|".join(_phrase_regex(keyword) for keyword in keywords), re.IGNORECASE)
    for theme, keywords in THEME_KEYWORDS.items()
}

_ANALYZER: Any = None  # VADER analyzer, created on first use and then reused


def _blank(match: re.Match[str]) -> str:
    """Replacement that hides a matched span from the patterns that follow."""
    return " " * len(match.group(0))


def macro_tone(text: str | None) -> float:
    """Macro tone in [-1, 1]: (pos - neg) / (pos + neg) over every lexicon hit; 0.0 if none."""
    if not isinstance(text, str) or not text.strip():
        return 0.0
    pos = neg = 0
    work = text
    for pattern, sign in _LEXICON_PATTERNS:
        work, hits = pattern.subn(_blank, work)
        if sign > 0:
            pos += hits
        else:
            neg += hits
    total = pos + neg
    return (pos - neg) / total if total else 0.0


def classify_theme(text: str | None) -> str:
    """Theme with the most whole-word keyword hits; ties go to the earlier theme, else 'general'."""
    if not isinstance(text, str) or not text.strip():
        return "general"
    counts = {theme: len(pattern.findall(text)) for theme, pattern in _THEME_PATTERNS.items()}
    best = max(counts.values())
    if best == 0:
        return "general"
    return next(theme for theme, count in counts.items() if count == best)


def vader_sentiment(text: str | None) -> float:
    """VADER compound score in [-1, 1]; 0.0 for empty text.

    The vaderSentiment package is imported on first use, and one analyzer is reused after that.
    """
    global _ANALYZER
    if not isinstance(text, str) or not text.strip():
        return 0.0
    if _ANALYZER is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        except ImportError as exc:
            raise ImportError("vaderSentiment is required: pip install vaderSentiment") from exc
        _ANALYZER = SentimentIntensityAnalyzer()
    return float(_ANALYZER.polarity_scores(text)["compound"])


def _text_values(df: pd.DataFrame, column: str) -> list[str]:
    """One column as strings; a missing column or a NaN value reads as ''."""
    if column not in df.columns:
        return [""] * len(df)
    return ["" if pd.isna(value) else str(value) for value in df[column]]


def score_news(df: pd.DataFrame, use_vader: bool = True) -> pd.DataFrame:
    """Copy of df with macro_tone, sentiment and theme columns added or replaced.

    The scored text is title + " " + text. sentiment is NaN when use_vader is False.
    An existing non-null theme is kept, and the other rows are classified.
    """
    texts = [
        f"{title} {body}"
        for title, body in zip(_text_values(df, "title"), _text_values(df, "text"), strict=True)
    ]
    given_themes = df["theme"].tolist() if "theme" in df.columns else [None] * len(df)
    out = df.copy()
    out["macro_tone"] = [macro_tone(text) for text in texts]
    if use_vader:
        out["sentiment"] = [vader_sentiment(text) for text in texts]
    else:
        out["sentiment"] = [math.nan] * len(texts)
    out["theme"] = [
        given if pd.notna(given) else classify_theme(text)
        for given, text in zip(given_themes, texts, strict=True)
    ]
    return out


def weekly_sentiment(df: pd.DataFrame, freq: str = "W-FRI") -> pd.DataFrame:
    """Weekly mean sentiment and macro tone by theme, with an 'all' row per period.

    Input needs published_utc, theme, sentiment and macro_tone. Periods are assigned from the
    UTC timestamps, and freq sets each period's last day (W-FRI by default). Output columns:
    period_end (datetime.date, the period's last day), theme, n (article count), sentiment and
    macro_tone (means). Rows are sorted by period_end and theme.
    """
    columns = ["period_end", "theme", "n", "sentiment", "macro_tone"]
    if df.empty:
        return pd.DataFrame(columns=columns)
    df = df.reset_index(drop=True)
    published = pd.to_datetime(df["published_utc"], utc=True).dt.tz_convert(None)
    work = pd.DataFrame(
        {
            "period_end": published.dt.to_period(freq).dt.end_time.dt.date,
            "theme": df["theme"],
            "sentiment": df["sentiment"].astype("float64"),
            "macro_tone": df["macro_tone"].astype("float64"),
        }
    )
    stats = {
        "n": ("sentiment", "size"),
        "sentiment": ("sentiment", "mean"),
        "macro_tone": ("macro_tone", "mean"),
    }
    by_theme = work.groupby(["period_end", "theme"]).agg(**stats).reset_index()
    overall = work.groupby("period_end").agg(**stats).reset_index()
    overall.insert(1, "theme", "all")
    out = pd.concat([by_theme, overall], ignore_index=True)
    return out.sort_values(["period_end", "theme"], ignore_index=True)[columns]
