"""Out-of-sample evaluation of the walk-forward models against two benchmarks.

Covers model selection, benchmarks and metrics, as set out in docs/design.md sections 7 and 8.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import pandas as pd

from macro_agent.backtest import portfolio_returns, select_trades
from macro_agent.config import Settings
from macro_agent.metrics import bootstrap_sharpe_ci, summarize
from macro_agent.model import CLASSIFIERS, REGRESSORS, predictive_metrics, walk_forward_predictions

MODEL_LABELS = {
    "baseline": "Naive baseline (strategy mean)",
    "ridge": "Ridge",
    "hgb": "Gradient boosting",
}
LONG_SPY = "Always long SPY, 1 session"
BUY_HOLD = "SPY buy and hold"
SUMMARY_COLUMNS = (
    "strategy", "sharpe", "sharpe_lo", "sharpe_hi", "sortino", "max_drawdown", "total_return",
    "cagr", "ann_vol", "exposure", "n_days", "n_trades", "win_rate", "avg_return",
    "profit_factor",
)
_TRADE_FIELDS = [
    "asset", "sign", "entry_kind", "entry_session", "exit_session", "net_ret", "entry_utc",
]


@dataclass
class Evaluation:
    """Out-of-sample results. books and trades are keyed by strategy label."""

    preds: pd.DataFrame
    predictive: pd.DataFrame
    books: dict[str, pd.DataFrame]
    trades: dict[str, pd.DataFrame]
    summary: pd.DataFrame
    strategy_pnl: pd.DataFrame
    oos_slots: list[str]
    oos_start: date
    oos_end: date


def evaluate(
    panel: pd.DataFrame,
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    settings: Settings,
    regressors: Sequence[str] = REGRESSORS,
    classifiers: Sequence[str] = CLASSIFIERS,
    step: int = 1,
) -> Evaluation:
    """Compare the walk-forward models with two benchmarks over one out-of-sample window.

    Each regressor keeps its best trade per slot when that trade's prediction is positive.
    Raises ValueError when there are too few slots to make any prediction.
    """
    preds = walk_forward_predictions(panel, settings.min_train_slots, regressors, classifiers, step)
    if preds.empty:
        raise ValueError("not enough slots for walk-forward predictions")
    predictive = predictive_metrics(preds, panel)

    entry = panel.groupby("slot_id")["entry_utc"].min()
    predicted = entry.index.isin(preds["slot_id"].unique())
    oos_slots = entry[predicted].sort_values(kind="stable").index.tolist()
    oos = panel[panel["slot_id"].isin(oos_slots)]
    oos_start = oos["entry_session"].min()
    oos_end = min(oos["exit_session"].max(), closes.index.max())

    trades: dict[str, pd.DataFrame] = {}
    books: dict[str, pd.DataFrame] = {}
    for name in regressors:
        label = MODEL_LABELS[name]
        scored = (
            preds[preds["model"] == name]
            .set_index("row")[["slot_id", "strategy_id", "pred"]]
            .join(panel[_TRADE_FIELDS])
        )
        trades[label] = select_trades(scored, top_k=1, min_pred=0.0)
        books[label] = portfolio_returns(
            trades[label], opens, closes, settings.cost_bps, start=oos_start, end=oos_end
        )

    spy_id = f"{settings.assets[0]}|long|1"
    trades[LONG_SPY] = oos[oos["strategy_id"] == spy_id]
    books[LONG_SPY] = portfolio_returns(
        trades[LONG_SPY], opens, closes, settings.cost_bps, start=oos_start, end=oos_end
    )
    trades[BUY_HOLD] = oos.iloc[:0]
    books[BUY_HOLD] = _buy_and_hold(closes[settings.assets[0]], oos_start, oos_end)

    rows = []
    for label in [*(MODEL_LABELS[name] for name in regressors), LONG_SPY, BUY_HOLD]:
        book = books[label]
        trade_returns = None if label == BUY_HOLD else trades[label]["net_ret"]
        stats = summarize(book["ret"], trade_returns=trade_returns, n_active=book["n_active"])
        sharpe_lo, sharpe_hi = bootstrap_sharpe_ci(
            book["ret"], n_boot=2000, block=5, alpha=0.10, seed=0
        )
        rows.append({"strategy": label, **stats, "sharpe_lo": sharpe_lo, "sharpe_hi": sharpe_hi})
    summary = pd.DataFrame(rows).reindex(columns=list(SUMMARY_COLUMNS))

    strategy_pnl = (
        oos.groupby("strategy_id")
        .agg(
            asset=("asset", "first"),
            rule=("rule", "first"),
            horizon=("horizon", "first"),
            value=("net_ret", "mean"),
            win_rate=("win", "mean"),
            n=("net_ret", "size"),
        )
        .reset_index()
        .sort_values("value", ascending=False, kind="stable", ignore_index=True)
    )
    return Evaluation(
        preds=preds,
        predictive=predictive,
        books=books,
        trades=trades,
        summary=summary,
        strategy_pnl=strategy_pnl,
        oos_slots=oos_slots,
        oos_start=oos_start,
        oos_end=oos_end,
    )


def top_strategies(strategy_pnl: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """The n strategies with the highest out-of-sample mean net return, best first."""
    top = strategy_pnl.head(n).rename(columns={"value": "mean_ret"})
    return top[["strategy_id", "mean_ret", "win_rate", "n"]].reset_index(drop=True)


def _buy_and_hold(close: pd.Series, start: date, end: date) -> pd.DataFrame:
    """Close-to-close returns of one asset over [start, end], fully invested and cost-free."""
    close = close.sort_index()
    daily = close / close.shift(1) - 1.0  # the first session is measured from the prior close
    ret = daily[(daily.index >= start) & (daily.index <= end)].dropna()
    return pd.DataFrame({"ret": ret, "n_active": 1})
