"""
Tests for model/scoring.py.

These are property tests with known answers, not smoke tests. The point is
that CRPS reduces to absolute error for a point forecast (so the two
metrics are on one scale), that a perfect forecast scores 0, and that
brier_skill pools rather than averages -- the bug that produced two
different HR edges for the same rows in baseball.
"""
import numpy as np
import pytest

from model.scoring import (
    brier, brier_skill, crps_ensemble, crps_pmf, paired_bootstrap, prob_over,
)


def test_crps_point_forecast_equals_absolute_error():
    """A one-sample ensemble is a point mass, so CRPS must equal |x - y|."""
    y = np.array([10.0, 25.0, 3.0])
    pred = np.array([[12.0], [25.0], [0.0]])
    got = crps_ensemble(y, pred)
    np.testing.assert_allclose(got, np.abs(pred[:, 0] - y), atol=1e-12)


def test_crps_perfect_forecast_is_zero():
    y = np.array([7.0, 7.0])
    samples = np.full((2, 500), 7.0)
    np.testing.assert_allclose(crps_ensemble(y, samples), 0.0, atol=1e-12)


def test_crps_rewards_sharpness_when_correct():
    """Two unbiased forecasts, same mean: the tighter one must score better."""
    rng = np.random.default_rng(0)
    y = np.array([50.0])
    tight = rng.normal(50, 5, (1, 4000))
    loose = rng.normal(50, 25, (1, 4000))
    assert crps_ensemble(y, tight)[0] < crps_ensemble(y, loose)[0]


def test_crps_punishes_overconfident_and_wrong():
    """Sharp but biased must lose to wide but centred."""
    rng = np.random.default_rng(1)
    y = np.array([50.0])
    sharp_wrong = rng.normal(20, 3, (1, 4000))
    wide_right = rng.normal(50, 25, (1, 4000))
    assert crps_ensemble(y, sharp_wrong)[0] > crps_ensemble(y, wide_right)[0]


def test_crps_pmf_matches_ensemble_for_same_distribution():
    """The discrete and ensemble forms must agree on the same distribution."""
    support = np.arange(0, 11, dtype=float)
    probs = np.zeros((1, 11))
    probs[0, [2, 5, 8]] = 1 / 3
    y = np.array([5.0])

    samples = np.array([[2.0] * 3000 + [5.0] * 3000 + [8.0] * 3000])
    assert abs(crps_pmf(y, support, probs)[0] - crps_ensemble(y, samples)[0]) < 0.05


def test_brier_skill_zero_for_base_rate_forecast():
    """Predicting the base rate every time must score exactly zero skill."""
    y = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=float)
    p = np.full(8, y.mean())
    assert abs(brier_skill(y, p)) < 1e-12


def test_brier_skill_positive_for_better_than_base_rate():
    y = np.array([1, 1, 1, 0, 0, 0], dtype=float)
    good = np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
    assert brier_skill(y, good) > 0.5


def test_brier_skill_pools_and_does_not_average():
    """
    The baseball bug, reproduced. Two weeks with different base rates: the
    average of per-week skills must differ from the pooled skill, and
    brier_skill must return the pooled one.
    """
    y1 = np.array([1, 1, 1, 1, 0], dtype=float)   # base rate 0.8
    p1 = np.full(5, 0.8)
    y2 = np.array([1, 0, 0, 0, 0], dtype=float)   # base rate 0.2
    p2 = np.full(5, 0.2)

    y_all = np.concatenate([y1, y2])
    p_all = np.concatenate([p1, p2])

    per_week_avg = 0.5 * (brier_skill(y1, p1) + brier_skill(y2, p2))
    pooled = brier_skill(y_all, p_all)

    # Each week's forecast is that week's base rate, so per-week skill is 0
    # for both -- but pooled against the combined 0.5 base rate it is not.
    assert abs(per_week_avg) < 1e-12
    assert pooled > 0.1, "pooled skill must not collapse to the per-week average"


def test_prob_over_is_strict():
    support = np.array([0.0, 1.0, 2.0, 3.0])
    probs = np.array([[0.25, 0.25, 0.25, 0.25]])
    assert prob_over(support, probs, 1.5)[0] == pytest.approx(0.5)
    # Whole-number line: 2 is not "over 2".
    assert prob_over(support, probs, 2.0)[0] == pytest.approx(0.25)


def test_paired_bootstrap_detects_a_real_difference():
    rng = np.random.default_rng(3)
    n = 800
    y = rng.binomial(1, 0.5, n).astype(float)
    good = np.where(y == 1, 0.75, 0.25)
    bad = np.full(n, 0.5)
    point, lo, hi = paired_bootstrap(y, good, bad, n_boot=500)
    assert point > 0
    assert lo > 0, "CI should exclude zero when the difference is large"


def test_brier_matches_hand_computation():
    y = np.array([1.0, 0.0])
    p = np.array([0.75, 0.25])
    assert brier(y, p) == pytest.approx(((0.25) ** 2 + (0.25) ** 2) / 2)
