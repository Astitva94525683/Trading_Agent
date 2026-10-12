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

US ETFs trade 09:30–16:00 ET. The trading calendar is the set of dates on which SPY has a price (holidays drop out automatically). Other symbols' extra dates are dropped, because Yahoo prints ^VIX on exchange holidays and shows a partial bar for the current day. `SessionCalendar.from_dates` builds open and close times per date, with optional early closes (not used yet).

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
- **Models** (`model.py`):
  - Regressors on net return: `MeanBaseline` (each strategy's historical mean, the naive benchmark), ridge (α = 10, after median imputation and scaling, with one-hot asset and rule), and `HistGradientBoostingRegressor` (depth 3, learning rate 0.05, 150 iterations, at least 40 rows per leaf, L2 1.0; NaN handled natively, all-NaN columns dropped).
  - Classifiers on win: the per-strategy base rate, logistic regression (C = 0.1) and `HistGradientBoostingClassifier` with the same settings. A fold whose training labels hold one class predicts the base rate.
  - Predictive metrics: rank IC (per-slot Spearman correlation between predicted and realised net return, averaged, with a t-statistic), hit rate, MAE, and AUC and Brier score for the classifiers.
- **Selection:** for each out-of-sample slot, trade the strategy with the highest predicted net return if that prediction is above zero; otherwise stay flat.

## 8. Backtest and metrics (`macro_agent/backtest.py`, `macro_agent/metrics.py`)

- **Calendar-time portfolio:** each session's portfolio return is the equal-weighted mean of the daily returns of the trades open that day (0 when flat). A trade's daily return is `sign × (P_t / P_{t−1} − 1)`, where `P_{t−1}` is the entry price on the first day. Costs are taken on the entry and exit days.
- **Metrics** on daily returns, 252 sessions a year, risk-free rate 0: Sharpe = mean / std (ddof 1) × √252; Sortino = mean / downside deviation × √252, where downside deviation = √(mean(min(r, 0)²)) over all days; max drawdown = minimum of equity / running peak − 1, with the starting equity of 1.0 counted as a peak. Also total return, CAGR, annualised volatility, exposure, and per-trade count, win rate, average return and profit factor.
- **Benchmarks** (`evaluate.py`): the naive baseline, always long SPY for one session on every slot, and SPY buy-and-hold over the same out-of-sample window (first out-of-sample entry to last exit). A 90% moving-block bootstrap interval (blocks of 5 sessions) accompanies each Sharpe ratio.

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
  - The playbook (`outlook.py`) builds the slot of the next high-impact inflation, labor, Fed or growth release, sets its surprise to +1 (beat) or −1 (miss), and scores every strategy with the regressor that had the best out-of-sample Sharpe ratio, refitted on all realised trades. A win probability comes from the gradient-boosting classifier. Sessions after the last price date are assumed to be weekdays without holidays.
- `reports/figures/`, the visual matrix: theme × week sentiment heatmap, category × month surprise heatmap, sentiment index over time with event markers, a strategy matrix (asset × rule by horizon, out-of-sample P&L), and equity and drawdown curves.
- **Chart style (dataviz guidance):**
  - Surface `#fcfcfb`; ink `#0b0b0b`, secondary ink `#52514e`, muted ink `#898781`, gridlines `#e1e0d9`.
  - Categorical colours in fixed order: `#2a78d6`, `#eb6834`, `#1baf7a`, `#eda100`.
  - Diverging blue↔red scales with a gray `#f0efec` midpoint.
  - One y-axis per chart, a legend for two or more series, and thin lines.

## 11. Module map

| Module | Phase | Contents |
|---|---|---|
| `config.py` | A | settings and constants, the FOMC schedule (added in B) |
| `timeutils.py` | A | UTC/ET conversion, `SessionCalendar`, entry/exit rules |
| `parsing.py` | A | value parser, event category, orientation, impact weight, ids |
| `db.py` | A | schema, upserts, views |
| `strategies.py` | A | strategy grid, price pivot, trade labels |
| `metrics.py`, `backtest.py` | A | performance metrics, trade selection, calendar-time portfolio |
| `walkforward.py` | A | purged splits, `MeanBaseline`, walk-forward driver |
| `sentiment.py` | A | macro tone lexicon, theme classifier, VADER wrapper, weekly series |
| `features.py` | A | event prep, slots, time/surprise/news/market features, panel |
| `ingest/raw.py`, `ingest/http.py` | B | raw JSON snapshots; HTTP with retries (keys in headers only) |
| `ingest/calendar.py` | B | Apify actor run, poll and dataset fetch; calendar normalizer |
| `ingest/news.py` | B | Exa weekly history and Tavily recent news; normalizers |
| `ingest/prices.py` | B | Yahoo daily OHLC via yfinance |
| `pipeline.py` | B | ingest into SQLite; build slots, features, labels and the panel |
| `model.py` | B | regressors and classifiers in the walk-forward; predictive metrics |
| `evaluate.py` | B | model trade selection, benchmarks, metrics, strategy P&L matrix |
| `outlook.py` | B | latest releases, sentiment trend, headlines, upcoming events, playbook |
| `charts.py`, `report.py`, `__main__.py` | B | figures, Markdown reports, CLI |

## 12. Known limitations (to restate in the report)

- **Small sample.** 180 days gives roughly 100 slots, so out-of-sample results carry wide confidence intervals.
- **Daily bars.** Entry is delayed until the next open or close; the immediate reaction to a release is only partly captured.
- **Forecasts.** These are the calendar site's consensus, not a survey of record.
- **Sentiment.** The lexicon and VADER scoring are coarse.
- **Costs.** These are flat assumptions.

## 13. Status and next steps

**Phase A (offline core), done.** All modules in the map marked A are implemented with unit tests (`python -m pytest`). They run on pandas and numpy alone.

**Phase B (ingestion, models, outputs), built and tested offline.** Everything in the module map marked B is implemented with unit tests, and an offline end-to-end smoke test runs the CLI on synthetic snapshots. News and prices were fetched live on 2026-10-12, and their raw snapshots are in `data/raw/` (the database `data/macro.db` is rebuilt from them and is not committed).

Decisions made from the live probes:

- **Calendar:** the actor is `xtracto/forexfactory-calendar`.
  - Input: `{"dateRange": "range", "rangeFrom", "rangeTo", "currencies": ["USD"], "minImpact": "medium"}`, with the actor's default residential proxy, which it needs to return actual values.
  - Release times come from `datetimeISO`, which is in UTC (the README's real sample has CAD GDP at 12:30Z, which is 08:30 ET). `timeLabel` is in the scraper's own time zone and is ignored.
  - The event id is ForexFactory's own (`ff<id>`), so a rescheduled event is not duplicated. Holidays and items without a clock time ("All Day", "Tentative") are skipped.
  - The token goes in an `Authorization: Bearer` header.
- **Exa:** `/search` with `category: "news"`, 10 results and 1,000 characters of text, for each theme and week. Exa gives dates without a time of day, so every article counts from 23:59:59 UTC that day. Finished weeks are cached and never fetched again.
- **Tavily:** `topic: "news"`, `time_range: "week"`, 10 results per theme.
- **Prices:** yfinance with `auto_adjust=True`, so open, high, low and close are adjusted for dividends (this matters for TLT).
- **FOMC dates:** the Federal Reserve's published schedule for 2025–2027 is in `config.FOMC_DECISION_DATES`, so "days to next FOMC" does not depend on the calendar's 30-day lookahead.

**Real run, 2026-10-12** (`python -m macro_agent run --as-of 2026-10-12`; rebuild offline with `--offline`):

- **Calendar probe.** One week (2026-09-10 to 09-17, USD, medium and high impact) returned 19 items, all from the live calendar with actual values. US CPI is at 08:30 ET and the Federal Funds Rate at 14:00 ET, so `datetimeISO` is UTC as assumed. The sample is now the fixture `tests/fixtures/forexfactory_items_real.json`.
- **Apify cost.** $0.04 for the probe and $0.58 for the full window (286 items at $0.002, plus a $0.005 start and under $0.001 of platform and proxy usage). Runs are capped with `maxTotalChargeUsd` (`Settings.apify_max_charge_usd`, $1.50), 1 GB of memory and a 900-second timeout. Each snapshot records the run's usage; the charged item count settles a few seconds after the run ends.
- **Data checks** (in `reports/evaluation_report.md`):
  - 283 USD events.
  - CPI and nonfarm payrolls at 08:30 ET, ADP at 08:15, FOMC at 14:00.
  - 1,014 articles. An independent recount matches the 7-day news count of all 117 slots, so no article published at or after a slot's entry is counted.
  - 164 sessions per symbol.
- **Result.** 117 slots, 86 of them out of sample (2026-05-28 to 2026-10-08).
  - The naive per-strategy mean has the only significant rank IC (0.124, t = 3.4) and a Sharpe ratio of 1.16.
  - Ridge reaches 1.27 with a rank IC that is not significant. Gradient boosting reaches −1.14.
  - SPY buy-and-hold reaches 0.84.
  - Every 90% Sharpe interval includes zero, so with about 90 out-of-sample sessions no strategy is distinguishable from noise. The outlook says so next to the playbook.

Possible next steps: a longer history (the brief fixes 180 days), intraday prices for the immediate reaction, and early-close sessions in the calendar.
