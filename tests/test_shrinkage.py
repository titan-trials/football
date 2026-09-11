"""
Tests for features/shrinkage.py.

The important ones are the planted-signal tests: generate players whose
true rates are known, and check that the estimator recovers the right prior
strength. A shrinkage function that silently returns the wrong k does not
crash, it just quietly makes every estimate worse -- which is how the
baseball model shipped raw unshrunk RBI rates for weeks.
"""
import numpy as np
import polars as pl
import pytest

from features.shrinkage import (
    estimate_prior_strength, estimate_prior_strength_counts, group_prior_mean, shrink,
)


def test_no_true_spread_gives_huge_k():
    """
    Players who differ only by binomial noise must shrink to the mean.
    Large k is the correct answer, not a failure.
    """
    rng = np.random.default_rng(0)
    n_players, trials = 200, 100
    true_p = 0.6  # identical for everyone
    successes = rng.binomial(trials, true_p, n_players)
    k = estimate_prior_strength(successes, np.full(n_players, trials))
    assert k >= 1000, f"expected hard shrinkage, got k={k}"


def test_large_true_spread_gives_small_k():
    """Genuinely different players must be allowed to keep their own rates."""
    rng = np.random.default_rng(1)
    n_players, trials = 300, 200
    true_p = rng.beta(2, 2, n_players)  # wide real spread
    successes = rng.binomial(trials, true_p)
    k = estimate_prior_strength(successes, np.full(n_players, trials))
    assert k < 50, f"expected light shrinkage, got k={k}"


def test_recovers_planted_prior_strength():
    """
    Plant a Beta(a, b) population, where the true k is a + b, and check the
    estimate lands in the right neighbourhood.
    """
    rng = np.random.default_rng(2)
    a, b = 12.0, 8.0          # true k = 20
    n_players, trials = 4000, 150
    true_p = rng.beta(a, b, n_players)
    successes = rng.binomial(trials, true_p)
    k = estimate_prior_strength(successes, np.full(n_players, trials))
    assert 14 < k < 28, f"expected k near {a + b}, got {k}"


def test_too_few_players_shrinks_hard():
    assert estimate_prior_strength([5, 6], [20, 20]) == 500.0


def test_shrink_moves_toward_prior():
    # 3 for 10 with a 0.5 prior and k=10 -> (3 + 5) / 20 = 0.4
    assert shrink([3], [10], 0.5, 10.0)[0] == pytest.approx(0.4)


def test_shrink_with_no_trials_returns_prior():
    assert shrink([0], [0], 0.25, 10.0)[0] == pytest.approx(0.25)


def test_shrink_accepts_per_row_priors():
    """Role-based priors mean prior_mean is an array, not a scalar."""
    got = shrink([3, 3], [10, 10], np.array([0.5, 0.1]), 10.0)
    assert got[0] == pytest.approx(0.4)
    assert got[1] == pytest.approx(0.2)


def test_counts_version_recovers_spread():
    """
    Targets per game: players with genuinely different volumes must get a
    modest k; identical players must get a large one.
    """
    rng = np.random.default_rng(4)
    n_players, games = 300, 16

    # Real spread in per-game target volume.
    true_rate = rng.gamma(shape=4.0, scale=1.5, size=n_players)
    counts = rng.poisson(true_rate[:, None], size=(n_players, games))
    k_spread = estimate_prior_strength_counts(
        counts.sum(axis=1), np.full(n_players, games), counts.var(axis=1, ddof=1)
    )

    # No real spread.
    counts_flat = rng.poisson(6.0, size=(n_players, games))
    k_flat = estimate_prior_strength_counts(
        counts_flat.sum(axis=1), np.full(n_players, games), counts_flat.var(axis=1, ddof=1)
    )

    assert k_spread < k_flat, f"spread k={k_spread} should be below flat k={k_flat}"
    assert k_spread < 20


def test_counts_version_handles_thin_players():
    assert estimate_prior_strength_counts([10, 12], [3, 3], [2.0, 2.0]) == 500.0


def test_group_prior_uses_cell_rate_when_thick():
    df = pl.DataFrame({
        "position": ["WR"] * 4 + ["TE"] * 4,
        "v": [50, 60, 55, 45, 10, 12, 8, 10],
        "t": [100] * 8,
    })
    out = group_prior_mean(df, "v", "t", by=["position"])
    wr = out.filter(pl.col("position") == "WR")["v_prior"][0]
    te = out.filter(pl.col("position") == "TE")["v_prior"][0]
    assert wr == pytest.approx(0.525)
    assert te == pytest.approx(0.10)
    assert wr > te, "cells must keep their own priors when thick enough"


def test_group_prior_falls_back_when_cell_is_thin():
    """A 2-player cell must not be allowed to invent its own prior."""
    df = pl.DataFrame({
        "position": ["WR"] * 6 + ["FB"] * 2,
        "v": [50, 60, 55, 45, 52, 48, 0, 1],
        "t": [100] * 8,
    })
    out = group_prior_mean(df, "v", "t", by=["position"])
    fb = out.filter(pl.col("position") == "FB")["v_prior"][0]
    global_rate = df["v"].sum() / df["t"].sum()
    assert fb == pytest.approx(global_rate)
