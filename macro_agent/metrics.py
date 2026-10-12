"""Performance metrics for periodic (daily) strategy returns.

Inputs may be lists, arrays or Series. NaNs are dropped first. Undefined
results are returned as float("nan") instead of raising: fewer than two
observations for a standard deviation, zero spread for a Sharpe ratio, or zero
downside deviation for a Sortino ratio. Conventions are in docs/design.md section 8.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd

ArrayLike = Sequence[float] | np.ndarray | pd.Series

NAN = float("nan")


def _clean(returns: ArrayLike) -> np.ndarray:
    """Return the observations as a 1-D float array with NaNs removed."""
    if isinstance(returns, pd.Series):
        values = returns.to_numpy(dtype=float, na_value=np.nan)
    else:
        values = np.asarray(returns, dtype=float)
    values = values.ravel()
    return values[~np.isnan(values)]


def sharpe_ratio(returns: ArrayLike, periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio: mean / std (ddof=1) * sqrt(periods_per_year). Risk-free rate is 0.

    NaN with fewer than two observations or zero spread.
    """
    x = _clean(returns)
    # Spread is tested with ptp: a constant series can have a float std of ~1e-17.
    if x.size < 2 or np.ptp(x) == 0.0:
        return NAN
    return float(x.mean() / x.std(ddof=1) * math.sqrt(periods_per_year))


def sortino_ratio(returns: ArrayLike, periods_per_year: int = 252) -> float:
    """Annualised Sortino ratio: mean / downside deviation * sqrt(periods_per_year).

    Downside deviation is sqrt(mean(min(r, 0)**2)) over all observations.
    NaN with fewer than two observations or zero downside deviation.
    """
    x = _clean(returns)
    if x.size < 2:
        return NAN
    downside = math.sqrt(float(np.mean(np.minimum(x, 0.0) ** 2)))
    if downside == 0.0:
        return NAN
    return float(x.mean() / downside * math.sqrt(periods_per_year))


def max_drawdown(returns: ArrayLike) -> float:
    """Largest peak-to-trough fall of the equity curve, <= 0.

    Equity starts at 1.0, which counts as the first peak. Returns 0.0 for an
    empty or all-gain series.
    """
    x = _clean(returns)
    if x.size == 0:
        return 0.0
    equity = np.concatenate(([1.0], np.cumprod(1.0 + x)))
    peak = np.maximum.accumulate(equity)
    return float(np.min(equity / peak - 1.0))


def total_return(returns: ArrayLike) -> float:
    """Compounded return, prod(1 + r) - 1. Returns 0.0 for an empty series."""
    x = _clean(returns)
    return float(np.prod(1.0 + x) - 1.0)


def cagr(returns: ArrayLike, periods_per_year: int = 252) -> float:
    """Compound annual growth rate, (1 + total_return) ** (periods_per_year / n) - 1.

    NaN for an empty series, or when the equity has gone negative (returns below -100%).
    """
    x = _clean(returns)
    if x.size == 0:
        return NAN
    growth = float(np.prod(1.0 + x))
    if growth < 0.0:
        return NAN
    return float(growth ** (periods_per_year / x.size) - 1.0)


def annual_volatility(returns: ArrayLike, periods_per_year: int = 252) -> float:
    """Annualised volatility, std (ddof=1) * sqrt(periods_per_year).

    NaN with fewer than two observations. A constant series has zero volatility, so 0.0.
    """
    x = _clean(returns)
    if x.size < 2:
        return NAN
    if np.ptp(x) == 0.0:
        return 0.0
    return float(x.std(ddof=1) * math.sqrt(periods_per_year))


def trade_stats(trade_returns: ArrayLike) -> dict[str, float | int]:
    """Summary of per-trade returns, with NaNs dropped.

    Keys: n_trades, win_rate (share of trades with return > 0), avg_return,
    median_return, profit_factor (sum of gains / |sum of losses|; inf when there
    are gains and no losses). All keys except n_trades are NaN when there are no trades.
    """
    x = _clean(trade_returns)
    n = x.size
    if n == 0:
        return {
            "n_trades": 0,
            "win_rate": NAN,
            "avg_return": NAN,
            "median_return": NAN,
            "profit_factor": NAN,
        }
    gains = float(x[x > 0.0].sum())
    losses = float(x[x < 0.0].sum())
    if losses == 0.0:
        profit_factor = math.inf if gains > 0.0 else NAN
    else:
        profit_factor = gains / abs(losses)
    return {
        "n_trades": int(n),
        "win_rate": float(np.mean(x > 0.0)),
        "avg_return": float(x.mean()),
        "median_return": float(np.median(x)),
        "profit_factor": profit_factor,
    }


def summarize(
    daily_returns: ArrayLike,
    trade_returns: ArrayLike | None = None,
    n_active: ArrayLike | None = None,
    periods_per_year: int = 252,
) -> dict[str, float | int]:
    """Headline metrics for a daily return series.

    Always returns sharpe, sortino, max_drawdown, total_return, cagr, ann_vol and
    n_days. Adds exposure (share of days with n_active > 0) when n_active is
    given, and every trade_stats key when trade_returns is given.
    """
    out: dict[str, float | int] = {
        "sharpe": sharpe_ratio(daily_returns, periods_per_year),
        "sortino": sortino_ratio(daily_returns, periods_per_year),
        "max_drawdown": max_drawdown(daily_returns),
        "total_return": total_return(daily_returns),
        "cagr": cagr(daily_returns, periods_per_year),
        "ann_vol": annual_volatility(daily_returns, periods_per_year),
        "n_days": int(_clean(daily_returns).size),
    }
    if n_active is not None:
        active = _clean(n_active)
        out["exposure"] = float(np.mean(active > 0.0)) if active.size else NAN
    if trade_returns is not None:
        out.update(trade_stats(trade_returns))
    return out


def bootstrap_sharpe_ci(
    returns: ArrayLike,
    n_boot: int = 2000,
    block: int = 5,
    alpha: float = 0.10,
    seed: int = 0,
    periods_per_year: int = 252,
) -> tuple[float, float]:
    """Moving-block bootstrap confidence interval for the Sharpe ratio.

    Each resample draws blocks of `block` consecutive returns with replacement
    until it reaches the original length. `block` is capped at the series length.
    The interval is the (alpha/2, 1 - alpha/2) percentiles of the resampled Sharpe
    ratios, which come from numpy default_rng(seed). Resamples with an undefined
    Sharpe ratio are skipped. Returns (nan, nan) for fewer than two observations.
    """
    if n_boot < 1 or block < 1 or not 0.0 < alpha < 1.0:
        raise ValueError("need n_boot >= 1, block >= 1 and 0 < alpha < 1")
    x = _clean(returns)
    n = x.size
    if n < 2:
        return NAN, NAN
    block = min(block, n)
    n_blocks = math.ceil(n / block)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    offsets = np.arange(block)
    sharpes = np.empty(n_boot)
    for b in range(n_boot):
        idx = (starts[b][:, None] + offsets).ravel()[:n]
        sharpes[b] = sharpe_ratio(x[idx], periods_per_year)
    defined = sharpes[np.isfinite(sharpes)]
    if defined.size == 0:
        return NAN, NAN
    lo, hi = np.percentile(defined, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return float(lo), float(hi)
