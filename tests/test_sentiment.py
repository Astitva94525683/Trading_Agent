"""Tests for macro_agent.sentiment: macro tone, themes, VADER wrapper and weekly series."""
from __future__ import annotations

import datetime as dt
import sys

import pandas as pd
import pytest

from macro_agent import sentiment
from macro_agent.sentiment import (
    MACRO_LEXICON,
    THEME_KEYWORDS,
    classify_theme,
    macro_tone,
    score_news,
    vader_sentiment,
    weekly_sentiment,
)

# --- macro_tone -------------------------------------------------------------


def test_dovish_headline_is_negative():
    assert macro_tone("Fed signals rate cuts as inflation cooled") == pytest.approx(-1.0)


def test_hotter_than_expected_headline_is_fully_positive():
    assert macro_tone("Hotter-than-expected CPI keeps Fed hawkish") == pytest.approx(1.0)


def test_mixed_text_gives_net_fraction():
    # hawkish (+1), cooling (-1), robust (+1): (2 - 1) / 3
    assert macro_tone("Hawkish Fed, but cooling inflation and robust jobs") == pytest.approx(1 / 3)


def test_rate_cuts_counts_once():
    # "rate cuts" is one negative hit against one positive ("robust") -> 0.0.
    # Counting "rate cut" inside it as well would give -1/3.
    assert macro_tone("Rate cuts ahead, but robust jobs") == pytest.approx(0.0)


def test_every_occurrence_is_counted():
    assert macro_tone("robust, robust, robust, cooled") == pytest.approx(0.5)


def test_whole_words_and_case_only():
    assert macro_tone("ROBUST") == 1.0
    assert macro_tone("robustness and slackening") == 0.0


def test_phrase_words_may_be_split_by_line_breaks():
    assert macro_tone("Hotter than\nexpected") == 1.0


def test_empty_or_no_hits_is_zero():
    assert macro_tone(None) == 0.0
    assert macro_tone("") == 0.0
    assert macro_tone("The weather in Paris is lovely") == 0.0


@pytest.mark.parametrize("phrase, sign", list(MACRO_LEXICON.items()))
def test_each_lexicon_phrase_scores_its_sign(phrase, sign):
    assert macro_tone(f"Note: {phrase}.") == float(sign)


# --- classify_theme ---------------------------------------------------------


def test_theme_order_is_fixed():
    assert list(THEME_KEYWORDS) == ["fed", "inflation", "labor", "growth"]


@pytest.mark.parametrize(
    "theme, keyword",
    [(theme, keyword) for theme, keywords in THEME_KEYWORDS.items() for keyword in keywords],
)
def test_each_keyword_classifies_to_its_theme(theme, keyword):
    assert classify_theme(keyword) == theme


def test_example_headlines():
    assert classify_theme("CPI rose 0.3% as inflation picked up") == "inflation"
    assert classify_theme("Nonfarm payrolls beat and wages rose") == "labor"
    assert classify_theme("FOMC minutes show Powell is cautious") == "fed"


def test_most_hits_wins_regardless_of_order():
    # growth has three hits (gdp, growth, economy) and inflation one (cpi)
    assert classify_theme("GDP growth slows, economy weakens, CPI cools") == "growth"


def test_tie_goes_to_earlier_theme():
    assert classify_theme("Fed holds; GDP steady") == "fed"  # fed vs growth, one hit each
    assert classify_theme("Jobs and GDP") == "labor"  # labor vs growth
    assert classify_theme("CPI and jobs") == "inflation"  # inflation vs labor


def test_no_hits_falls_back_to_general():
    assert classify_theme("The weather in Paris is lovely") == "general"
    assert classify_theme("Inflationary pressures build") == "general"  # whole words only
    assert classify_theme(None) == "general"
    assert classify_theme("") == "general"


# --- score_news -------------------------------------------------------------


def test_score_news_without_vader_keeps_given_themes_and_fills_missing():
    df = pd.DataFrame(
        {
            "title": ["Fed hikes rates", "CPI cooled", None],
            "text": ["Rate hike and hawkish tone", None, "Jobs report shows layoffs"],
            "theme": ["growth", None, float("nan")],
        }
    )
    out = score_news(df, use_vader=False)
    assert out["theme"].tolist() == ["growth", "inflation", "labor"]
    assert out["macro_tone"].tolist() == pytest.approx([1.0, -1.0, -1.0])
    assert out["sentiment"].isna().all()
    assert list(df.columns) == ["title", "text", "theme"]  # the input is not modified


def test_score_news_treats_missing_columns_as_empty():
    out = score_news(pd.DataFrame({"text": ["Recession fears grow"]}), use_vader=False)
    assert out.loc[0, "macro_tone"] == -1.0
    assert out.loc[0, "theme"] == "growth"  # "recession" is a growth keyword


# --- weekly_sentiment -------------------------------------------------------


def test_weekly_sentiment_groups_by_week_and_theme():
    df = pd.DataFrame(
        {
            "published_utc": pd.to_datetime(
                [
                    "2026-10-05T14:00:00Z",  # Mon, week ending Fri 2026-10-09
                    "2026-10-08T09:30:00Z",  # Thu, same week
                    "2026-10-09T23:59:00Z",  # Fri evening, same week
                    "2026-10-10T00:30:00Z",  # Sat, so the week ending Fri 2026-10-16
                    "2026-10-13T12:00:00Z",  # Tue, same week
                ],
                utc=True,
            ),
            "theme": ["fed", "inflation", "fed", "labor", "labor"],
            "sentiment": [0.5, -0.5, 0.0, 0.2, 0.4],
            "macro_tone": [1.0, -1.0, 0.0, 0.5, 0.5],
        }
    )
    out = weekly_sentiment(df)
    expected = pd.DataFrame(
        {
            "period_end": [dt.date(2026, 10, 9)] * 3 + [dt.date(2026, 10, 16)] * 2,
            "theme": ["all", "fed", "inflation", "all", "labor"],
            "n": [3, 2, 1, 2, 2],
            "sentiment": [0.0, 0.25, -0.5, 0.3, 0.3],
            "macro_tone": [0.0, 0.5, -1.0, 0.5, 0.5],
        }
    )
    pd.testing.assert_frame_equal(out, expected, check_dtype=False, atol=1e-9)
    assert all(type(day) is dt.date for day in out["period_end"])


def test_missing_theme_counts_only_in_the_all_row():
    df = pd.DataFrame(
        {
            "published_utc": pd.to_datetime(
                ["2026-10-07T10:00:00Z", "2026-10-07T11:00:00Z"], utc=True
            ),
            "theme": ["growth", None],
            "sentiment": [0.2, 0.4],
            "macro_tone": [1.0, -1.0],
        }
    )
    out = weekly_sentiment(df)
    assert out["theme"].tolist() == ["all", "growth"]
    assert out.loc[0, "n"] == 2
    assert out.loc[0, "sentiment"] == pytest.approx(0.3)


def test_weekly_sentiment_empty_input():
    empty = pd.DataFrame(columns=["published_utc", "theme", "sentiment", "macro_tone"])
    out = weekly_sentiment(empty)
    assert out.empty
    assert list(out.columns) == ["period_end", "theme", "n", "sentiment", "macro_tone"]


# --- VADER wrapper ----------------------------------------------------------


def test_vader_empty_text_is_zero_without_the_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "vaderSentiment", None)  # any import would fail
    monkeypatch.setitem(sys.modules, "vaderSentiment.vaderSentiment", None)
    assert vader_sentiment(None) == 0.0
    assert vader_sentiment("") == 0.0


def test_missing_vader_package_raises_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "vaderSentiment", None)
    monkeypatch.setitem(sys.modules, "vaderSentiment.vaderSentiment", None)
    monkeypatch.setattr(sentiment, "_ANALYZER", None)
    with pytest.raises(ImportError, match="vaderSentiment is required: pip install vaderSentiment"):
        vader_sentiment("Strong rebound")
    with pytest.raises(ImportError):
        score_news(pd.DataFrame({"text": ["Strong rebound"]}))


def test_vader_scores_are_signed_and_bounded():
    pytest.importorskip("vaderSentiment")
    assert vader_sentiment("This is wonderful news") > 0
    assert vader_sentiment("This is terrible news") < 0
    assert -1.0 <= vader_sentiment("Wonderful but terrible") <= 1.0


def test_vader_analyzer_is_created_once():
    pytest.importorskip("vaderSentiment")
    vader_sentiment("wonderful")
    first = sentiment._ANALYZER
    vader_sentiment("terrible")
    assert sentiment._ANALYZER is first


def test_score_news_fills_vader_sentiment():
    pytest.importorskip("vaderSentiment")
    df = pd.DataFrame({"title": ["Wonderful rebound", "Terrible crash"], "text": [None, "Awful"]})
    out = score_news(df)
    assert out["sentiment"].notna().all()
    assert out["sentiment"].between(-1, 1).all()
    assert out.loc[0, "sentiment"] > 0
    assert out.loc[1, "sentiment"] < 0
