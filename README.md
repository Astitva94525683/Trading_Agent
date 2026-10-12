# Trading_Agent: macro economics agent

A Python machine-learning pipeline that predicts the best macro trading strategy around US economic releases, built for the CrowdWisdomTrading Quantitative Data Scientist assessment. It works in four stages:

1. Scrape the macro calendar (CPI, FOMC, jobs reports and other USD releases) for the last 180 days with Apify, and pull news with Exa and Tavily.
2. Join the calendar, news and market prices in a SQLite database.
3. Engineer time-based, surprise, sentiment and market features.
4. Forecast the P&L and win rate of 48 strategy permutations (4 ETFs × 4 direction rules × 3 holding periods), validated with purged walk-forward testing.

The full design, including how release times are aligned between UTC and New York time without look-ahead, is in [`docs/design.md`](docs/design.md).

## Status

| Phase | What | State |
|---|---|---|
| A | Time alignment, parsing, database, strategy labels, features, walk-forward, backtest and metrics, sentiment | done, unit-tested offline |
| B | Apify/Exa/Tavily/Yahoo ingestion, ML models, charts, evaluation report, macro outlook | next (needs network access and API keys) |

## Layout

```
macro_agent/
  config.py       settings and constants (assets, rules, horizons, costs)
  timeutils.py    UTC/ET conversion, trading sessions, entry and exit rules
  parsing.py      calendar values ("3.2%", "256K"), event categories, news timestamps
  db.py           SQLite schema, upserts, SQL views joining events, news and prices
  strategies.py   the 48 strategy permutations and their trade labels
  features.py     event slots, time, surprise, news and market features, model panel
  sentiment.py    macro tone lexicon, news themes, VADER, weekly series
  walkforward.py  purged expanding walk-forward and the naive benchmark
  backtest.py     trade selection and the calendar-time portfolio
  metrics.py      Sharpe, Sortino, max drawdown, CAGR, trade stats, bootstrap CI
tests/            unit tests for every module
docs/design.md    design and method
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env   # then add your keys, or export them as environment variables
python -m pytest
```

The pipeline reads `APIFY_TOKEN`, `TAVILY_API_KEY` and `EXA_API_KEY` from the environment or from a git-ignored `.env` file. Keys are never committed.
