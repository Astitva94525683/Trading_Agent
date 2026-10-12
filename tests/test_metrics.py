"""Tests for macro_agent.metrics, using small series with hand-computed answers."""

import math

import numpy as np
import pandas as pd
import pytest

from macro_agent.metrics import (
    annual_volatility,
    bootstrap_sharpe_ci,
    cagr,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    summarize,
    total_return,
    trade_stats,
)


def test_sharpe_hand_computed():
    # mean 0.1; deviations 0, -0.1, 0.1 -> sum of squares 0.02, sample variance 0.01, std 0.1
    assert sharpe_ratio([0.1, 0.0, 0.2], periods_per_year=1) == pytest.approx(1.0)
    assert sharpe_ratio([0.1, 0.0, 0.2]) == pytest.approx(math.sqrt(252))


def test_sortino_hand_computed():
    # mean 0.05; min(r, 0) = 0, -0.1, 0, 0 -> mean square 0.0025 -> downside deviation 0.05
    assert sortino_ratio([0.1, -0.1, 0.2, 0.0], periods_per_year=1) == pytest.approx(1.0)
    assert sortino_ratio([0.1, -0.1, 0.2, 0.0], periods_per_year=4) == pytest.approx(2.0)


def test_annual_volatility_hand_computed():
    # sample std of [0.1, 0.0, 0.2] is 0.1
    assert annual_volatility([0.1, 0.0, 0.2], periods_per_year=1) == pytest.approx(0.1)
    assert annual_volatility([0.1, 0.0, 0.2], periods_per_year=4) == pytest.approx(0.2)


@pytest.mark.parametrize(
    "returns, expected",
    [
        ([0.1, -0.5, 0.2], -0.5),  # equity 1, 1.1, 0.55, 0.66: 0.55 / 1.1 - 1 = -0.5
        ([-0.1, 0.05], -0.1),  # the starting 1.0 is the first peak: 0.9 / 1.0 - 1
        ([-0.1, 0.2, -0.2], -0.2),  # equity 1, 0.9, 1.08, 0.864: 0.864 / 1.08 - 1
        ([0.01, 0.02], 0.0),  # all gains
        ([], 0.0),  # empty
    ],
)
def test_max_drawdown(returns, expected):
    assert max_drawdown(returns) == pytest.approx(expected)


def test_total_return_and_cagr():
    assert total_return([0.1, -0.5, 0.2]) == pytest.approx(1.1 * 0.5 * 1.2 - 1)  # -0.34
    assert total_return([]) == 0.0
    assert cagr([0.1, 0.1], periods_per_year=1) == pytest.approx(1.21 ** 0.5 - 1)  # 0.1
    assert cagr([0.1, 0.1], periods_per_year=2) == pytest.approx(0.21)


def test_undefined_results_are_nan():
    assert math.isnan(sharpe_ratio([0.1]))  # fewer than two observations
    assert math.isnan(sharpe_ratio([]))
    assert math.isnan(sharpe_ratio([0.05, np.nan]))  # one observation once NaNs are dropped
    assert math.isnan(sharpe_ratio([0.01, 0.01, 0.01]))  # zero std
    assert math.isnan(sortino_ratio([0.1, 0.2]))  # zero downside deviation
    assert math.isnan(sortino_ratio([0.1]))
    assert math.isnan(annual_volatility([0.1]))
    assert math.isnan(cagr([]))


def test_constant_returns_have_zero_annual_volatility():
    assert annual_volatility([0.01, 0.01, 0.01]) == 0.0


@pytest.mark.parametrize("wrap", [list, np.array, pd.Series])
def test_nans_are_dropped_before_computing(wrap):
    returns = wrap([0.1, np.nan, 0.0, 0.2])
    assert sharpe_ratio(returns, periods_per_year=1) == pytest.approx(1.0)
    assert annual_volatility(returns, periods_per_year=1) == pytest.approx(0.1)


def test_trade_stats_values():
    stats = trade_stats([0.1, -0.05, 0.2, -0.1])
    assert stats["n_trades"] == 4
    assert stats["win_rate"] == pytest.approx(0.5)
    assert stats["avg_return"] == pytest.approx(0.0375)
    assert stats["median_return"] == pytest.approx(0.025)
    assert stats["profit_factor"] == pytest.approx(0.3 / 0.15)


def test_profit_factor_without_losses_is_inf():
    assert trade_stats([0.1, 0.2])["profit_factor"] == math.inf
    assert trade_stats([-0.1, -0.2])["profit_factor"] == 0.0


def test_trade_stats_without_trades_is_nan():
    stats = trade_stats([])
    assert stats["n_trades"] == 0
    for key in ("win_rate", "avg_return", "median_return", "profit_factor"):
        assert math.isnan(stats[key])


def _sample_returns(n: int = 180, seed: int = 3) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0005, 0.01, size=n)


def test_bootstrap_ci_is_deterministic_for_fixed_seed():
    returns = _sample_returns()
    first = bootstrap_sharpe_ci(returns, seed=7)
    assert bootstrap_sharpe_ci(returns, seed=7) == first
    lo, hi = first
    assert math.isfinite(lo) and math.isfinite(hi)
    assert lo <= hi


def test_bootstrap_narrower_alpha_lies_inside_wider_interval():
    # Same seed means the same resamples, and percentiles are monotone in alpha.
    returns = _sample_returns()
    lo90, hi90 = bootstrap_sharpe_ci(returns, alpha=0.10, seed=7)
    lo50, hi50 = bootstrap_sharpe_ci(returns, alpha=0.50, seed=7)
    assert lo90 <= lo50 <= hi50 <= hi90


def test_bootstrap_block_equal_to_length_reproduces_the_point_estimate():
    # With block = len(returns) the only possible resample is the original series.
    returns = _sample_returns()
    lo, hi = bootstrap_sharpe_ci(returns, block=len(returns))
    point = sharpe_ratio(returns)
    assert lo == pytest.approx(point)
    assert hi == pytest.approx(point)


def test_bootstrap_with_too_few_observations_is_nan():
    lo, hi = bootstrap_sharpe_ci([0.1])
    assert math.isnan(lo) and math.isnan(hi)


def test_summarize_headline_keys_and_values():
    out = summarize([0.1, 0.0, 0.2], periods_per_year=1)
    assert set(out) == {
        "sharpe",
        "sortino",
        "max_drawdown",
        "total_return",
        "cagr",
        "ann_vol",
        "n_days",
    }
    assert out["sharpe"] == pytest.approx(1.0)
    assert out["total_return"] == pytest.approx(1.1 * 1.0 * 1.2 - 1)  # 0.32
    assert out["max_drawdown"] == 0.0
    assert out["ann_vol"] == pytest.approx(0.1)
    assert out["n_days"] == 3


def test_summarize_adds_exposure_and_trade_stats():
    out = summarize(
        [0.1, 0.0, 0.2],
        trade_returns=[0.1, -0.05, 0.2, -0.1],
        n_active=[1, 0, 2],
        periods_per_year=1,
    )
    assert out["exposure"] == pytest.approx(2 / 3)
    assert out["n_trades"] == 4
    assert out["win_rate"] == pytest.approx(0.5)
    assert out["profit_factor"] == pytest.approx(2.0)


def test_summarize_drops_nan_days_from_n_days():
    out = summarize([0.1, np.nan, 0.0, 0.2], periods_per_year=1)
    assert out["n_days"] == 3
    assert out["sharpe"] == pytest.approx(1.0)
