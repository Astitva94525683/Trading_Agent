# Design: macro economics agent

This is the working design for the CrowdWisdomTrading assessment. Module specs here are the contract the code and tests follow.

## 1. What the brief asks for

A Python machine-learning pipeline that predicts the best macro trading strategy around economic releases:

1. **Event ingestion (Apify).** Scrape a macro calendar (CPI, FOMC, employment reports and similar) for the last 180 days. Keep release timestamps aligned between UTC and New York time, event names, actual values and forecasts. Add the most recent relevant economic news from Tavily (Exa is also requested in the submission).
2. **Features and outlook.** Time-based features, plus a one-page summary of the macro outlook.
3. **Model and walk-forward validation.** Forecast expected trade P&L or win rate for each strategy permutation, validated strictly out of sample.
4. **Visual matrix.** Charts that show macro sentiment changing over time.

Deliverables: the GitHub repository, the Apify/Tavily/Exa actor IDs used (the tokens go by email, never into git), and an evaluation report with Sharpe ratio, Sortino ratio and max drawdown.

## 2. Pipeline

```
Apify calendar ─┐
Exa news (180d) ─┤                 ┌─ slots + features ─┐
Tavily news (7d) ┼─> SQLite (data/macro.db) ─┤                    ├─> walk-forward model ─> backtest ─> reports + charts
Yahoo prices ────┘                 └─ trade labels ─────┘
```

Raw API responses are saved under `data/raw/` so the pipeline can be re-run offline (`--offline`) without spending API credits.

## 3. Time alignment (`macro_agent/timeutils.py`)

All timestamps are stored as UTC ISO strings (`YYYY-MM-DDTHH:MM:SSZ`); New York time (`America/New_York`, written ET) is derived with `zoneinfo`, so daylight-saving changes are handled (2026: DST starts 8 March, ends 1 November). A naive timestamp from a source is interpreted in that source's stated time zone.

US ETFs trade 09:30–16:00 ET. The trading calendar is the set of session dates in the price data (holidays drop out automatically); `SessionCalendar.from_dates` builds open and close times per date, with optional early closes.

**Entry rule (no look-ahead).** A trade enters at the first price printed after the release:

| Release time (ET) on date d | Entry |
|---|---|
| d is a session and release < open | open of d (`kind="open"`) |
| d is a session and open ≤ release < close | close of d (`kind="close"`) |
| release ≥ close, or d is not a session | open of the next session (`kind="open"`) |

**Exit rule.** `horizon` = sessions held. An open entry on session s exits at the close of `s + (horizon − 1)` sessions; a close entry exits at the close of `s + horizon` sessions. So horizon 1 is open→close of the release day for pre-market releases (CPI, jobs at 08:30), and close→next close for in-session releases (FOMC at 14:00).

## 4. Events (`macro_agent/parsing.py`, `macro_agent/features.py`)

- **Values.** Calendar strings such as `3.2%`, `-0.1%`, `256K`, `1.25M` are parsed to floats; K/M/B/T are multiplied out so actual and forecast compare in the same units, `%` values stay as printed.
- **Category.** Keyword rules map an event name to one of `fed, inflation, labor, growth, consumer, housing, other`.
- **Orientation.** Surprise = actual − forecast, sign-flipped for indicators where higher means weaker (unemployment rate, jobless/unemployment claims). Positive surprise therefore means "stronger or hotter than expected".
- **Scaled surprise (no look-ahead).** `surprise_z` = surprise ÷ median |surprise| of the same event's *earlier* releases, clipped to ±3. With fewer than two earlier releases (or a zero median) it falls back to the sign (±1, or 0 for no surprise).
- **Slots.** Events whose information arrives before the same entry point (same entry session and kind) form one slot; for example 08:30 CPI and 08:30 jobless claims on the same day. A slot is the unit the model trades. Its surprise is the impact-weighted mean of its events' `surprise_z` (high = 1.0, medium = 0.5, low = 0.25).

## 5. Strategy permutations and labels (`macro_agent/strategies.py`)

A strategy is `(asset, rule, horizon)`: 4 assets × 4 rules × 3 horizons = 48 permutations.

- Assets: SPY, TLT, GLD, UUP.
- Rules → position sign: `long` = +1, `short` = −1, `long_on_beat` = sign(surprise), `short_on_beat` = −sign(surprise). The beat rules do not trade when the slot has no surprise (missing or exactly 0).
- Trade return: `gross = sign × (exit_price / entry_price − 1)`, `net = gross − 2 × cost` with cost = 2 bps per side. `win = net > 0`.

## 6. Features (`macro_agent/features.py`)

Every feature uses only information available at the slot's entry time.

- **Time-based:** release hour (ET), weekday, day of month, week of month, month, open vs close entry, days since the previous slot, days since the previous slot of the same category, days to the next FOMC decision and since the last one (the FOMC schedule is published in advance), number of slots in the same week.
- **Macro surprise:** slot `surprise_z`, its absolute value, event-category counts, and an exponentially weighted surprise index (half-life 30 days) per category built from earlier slots only, similar in spirit to an economic surprise index.
- **News sentiment:** article count and mean sentiment/tone over the 7 and 30 days before entry, the 7-day minus 30-day change, and 7-day tone per theme. A news timestamp without a time of day counts as 23:59:59 UTC that day so it can never leak into a same-day entry.
- **Market state** (per asset): 5- and 20-session returns and 20-session realised volatility up to the previous close, the reaction from the previous close to the entry price (known at entry), and the previous VIX close.
- **Strategy-aware:** asset, rule, horizon, sign, and sign-adjusted reaction, surprise and momentum.

## 7. Model and walk-forward validation (`macro_agent/walkforward.py`, `macro_agent/model.py`)

- **Panel:** one row per (slot, strategy); target = net return (regression) and win (classification).
- **Purged expanding walk-forward:** slots are ordered by entry time. For each test slot (after `min_train_slots` warm-up slots) the training set is every panel row whose **exit** is strictly before the test slot's entry. Labels that had not been realised at decision time are therefore excluded (purging), so a 5-session trade from two days earlier cannot leak into training.
- **Models:** `MeanBaseline` (each strategy's historical mean, the naive benchmark), ridge regression, and gradient boosting (scikit-learn `HistGradientBoostingRegressor` / `Classifier`, shallow and regularised for the small sample).
- **Selection:** for each out-of-sample slot, trade the strategy with the highest predicted net return if that prediction is above zero; otherwise stay flat.

## 8. Backtest and metrics (`macro_agent/backtest.py`, `macro_agent/metrics.py`)

- **Calendar-time portfolio:** each session's portfolio return is the equal-weighted mean of the daily returns of the trades open that day (0 when flat). A trade's daily return is `sign × (P_t / P_{t−1} − 1)`, where `P_{t−1}` is the entry price on the first day. Costs are taken on the entry and exit days.
- **Metrics** on daily returns, 252 sessions a year, risk-free rate 0: Sharpe = mean / std (ddof 1) × √252; Sortino = mean / downside deviation × √252, where downside deviation = √(mean(min(r, 0)²)) over all days; max drawdown = minimum of equity / running peak − 1, with the starting equity of 1.0 counted as a peak. Also total return, CAGR, annualised volatility, exposure, and per-trade count, win rate, average return and profit factor.
- **Benchmarks:** the naive baseline, always long SPY for one session on every slot, and SPY buy-and-hold over the same out-of-sample window. A block-bootstrap confidence interval accompanies the Sharpe ratio.

## 9. Database (`macro_agent/db.py`)

SQLite at `data/macro.db`.

- **Raw tables:** `events`, `news` and `prices`, written with idempotent upserts.
- **Derived tables:** `slots`, `features`, `labels` and `predictions`, written with pandas `to_sql(replace)`.
- **Views (SQL joins across datasets):**
  - `v_event_news_7d`: for each event, the count and mean sentiment/tone of news published in the 7 days before release.
  - `v_event_market`: for each (event, symbol), the last close before the release date and the first close on or after it.

## 10. Outputs

- `reports/evaluation_report.md`: data coverage, the walk-forward set-up, predictive metrics (rank IC, hit rate, AUC), and strategy metrics with confidence intervals for the model and the benchmarks, plus limitations.
- `reports/macro_outlook.md`: a one-page outlook covering the latest releases and surprises, the sentiment trend, top recent Tavily headlines with links, upcoming events from the calendar, and a conditional playbook (the model's best strategy if the next key release beats or misses).
- `reports/figures/`, the visual matrix: theme × week sentiment heatmap, category × month surprise heatmap, sentiment index over time with event markers, a strategy matrix (asset × rule by horizon, out-of-sample P&L), and equity and drawdown curves.
- **Chart style (dataviz guidance):**
  - Surface `#fcfcfb`; ink `#0b0b0b`, secondary ink `#52514e`, muted ink `#898781`, gridlines `#e1e0d9`.
  - Categorical colours in fixed order: `#2a78d6`, `#eb6834`, `#1baf7a`, `#eda100`.
  - Diverging blue↔red scales with a gray `#f0efec` midpoint.
  - One y-axis per chart, a legend for two or more series, and thin lines.

## 11. Module map

| Module | Phase | Contents |
|---|---|---|
| `config.py` | A | settings and constants |
| `timeutils.py` | A | UTC/ET conversion, `SessionCalendar`, entry/exit rules |
| `parsing.py` | A | value parser, event category, orientation, impact weight, ids |
| `db.py` | A | schema, upserts, views |
| `strategies.py` | A | strategy grid, price pivot, trade labels |
| `metrics.py`, `backtest.py` | A | performance metrics, trade selection, calendar-time portfolio |
| `walkforward.py` | A | purged splits, `MeanBaseline`, walk-forward driver |
| `sentiment.py` | A | macro tone lexicon, theme classifier, VADER wrapper, weekly series |
| `features.py` | A | event prep, slots, time/surprise/news/market features, panel |
| `ingest/` | B | Apify calendar, Exa, Tavily, Yahoo prices, raw snapshots |
| `model.py` | B | ridge and gradient boosting in the walk-forward |
| `charts.py`, `report.py`, `__main__.py` | B | figures, reports, CLI |

## 12. Known limitations (to restate in the report)

- **Small sample.** 180 days gives roughly 100 slots, so out-of-sample results carry wide confidence intervals.
- **Daily bars.** Entry is delayed until the next open or close; the immediate reaction to a release is only partly captured.
- **Forecasts.** These are the calendar site's consensus, not a survey of record.
- **Sentiment.** The lexicon and VADER scoring are coarse.
- **Costs.** These are flat assumptions.

## 13. Status and next steps

**Phase A (offline core), done.** All modules in the map marked A are implemented with unit tests (`python -m pytest`). They run on pandas and numpy alone.

**Phase B needs network access and API keys** (`APIFY_TOKEN`, `TAVILY_API_KEY`, `EXA_API_KEY` as environment variables). In order:

1. `pip install -r requirements-dev.txt`. The VADER tests in `tests/test_sentiment.py` then run instead of skipping.
2. **Calendar probe.**
   - Read the input schema of the `xtracto/forexfactory-calendar` actor (`GET https://api.apify.com/v2/acts/xtracto~forexfactory-calendar`, then its default build's `inputSchema`) and run it on one week.
   - Check the time zone of its timestamps: US CPI must land at 08:30 ET and the FOMC rate decision at 14:00 ET.
   - Fall back to `hipersoft/forexfactory-scraper` or `automation-lab/tradingview-economic-calendar-scraper` if the output lacks actual or forecast values.
   - Write the normalizer from real output and keep a small sample as a test fixture.
3. **Ingest** (`macro_agent/ingest/`, REST via `requests`, raw JSON saved under `data/raw/` for `--offline` re-runs).
   - Calendar: Apify asynchronous run, then poll, then fetch the dataset, for the window [today − 180d, today + 30d].
   - Exa: weekly searches per news theme over the 180 days (`/search` with published-date filters).
   - Tavily: news from the last 7 days for the outlook.
   - Prices: Yahoo daily OHLC via yfinance for SPY, TLT, GLD, UUP and ^VIX, starting 60 days before the window.
4. **Model** (`model.py`).
   - Ridge and `HistGradientBoostingRegressor` / `Classifier` on `features.MODEL_FEATURES` plus one-hot `CATEGORICAL_FEATURES`, run through `walkforward.walk_forward`.
   - `MeanBaseline` needs `strategy_id` among its inputs.
   - Classifier folds whose training labels hold one class must fall back to the base rate.
5. **Charts and reports.** `charts.py`, `report.py` and a CLI `python -m macro_agent run [--offline]`, with an offline end-to-end smoke test on synthetic data.
6. **Real run.**
   - Data checks: event counts by category, the CPI 08:30 / FOMC 14:00 ET anchors, no news after its slot's entry, price coverage.
   - Then the evaluation report and the macro outlook.
