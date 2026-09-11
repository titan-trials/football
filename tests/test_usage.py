"""
Tests for features/usage.py.

The centrepiece is test_joint_moments_match_simulation: the covariance
identity is derived algebra, and derived algebra that nobody checked is how
the baseball K-prop shipped a rate it had never validated. So it is checked
against a Monte Carlo of the actual generative process.

The truncation test exists because of team-run-model.md's MAX_RUNS gotcha:
a support cap tuned to the cases you happen to test is a cap that fails on
the case you did not.
"""
import numpy as np
import polars as pl
import pytest

from features.usage import (
    MAX_OPPORTUNITIES, ShareModel, TeamVolumeModel, joint_moments, negbin_pmf,
    opportunity_pmf, teammate_correlation,
)


# --- negbin_pmf --------------------------------------------------------

def test_negbin_matches_requested_moments():
    support = np.arange(0, 201)
    mean, var = 32.3, 63.8            # the measured league numbers
    p = negbin_pmf(mean, var, support)
    got_mean = p @ support
    got_var = p @ (support ** 2) - got_mean ** 2
    assert got_mean == pytest.approx(mean, rel=1e-3)
    assert got_var == pytest.approx(var, rel=1e-2)


def test_negbin_falls_back_to_poisson_when_not_overdispersed():
    support = np.arange(0, 101)
    p = negbin_pmf(10.0, 10.0, support)
    got_mean = p @ support
    got_var = p @ (support ** 2) - got_mean ** 2
    assert got_var == pytest.approx(got_mean, rel=1e-2)


def test_negbin_sums_to_one():
    support = np.arange(0, MAX_OPPORTUNITIES + 1)
    assert negbin_pmf(32.0, 64.0, support).sum() == pytest.approx(1.0)


def test_negbin_rejects_bad_mean():
    with pytest.raises(ValueError):
        negbin_pmf(0.0, 5.0, np.arange(0, 10))


def test_support_cap_does_not_bite_at_realistic_volumes():
    """
    team-run-model.md's MAX_RUNS lesson: truncating the tail and
    renormalising pulls the mean down. At the highest team pass volume ever
    observed (68), the discarded mass above MAX_OPPORTUNITIES must still be
    negligible.
    """
    support = np.arange(0, MAX_OPPORTUNITIES + 1)
    p = negbin_pmf(45.0, 45.0 * 1.97, support)   # far above league mean
    got_mean = p @ support
    assert got_mean == pytest.approx(45.0, rel=0.02), (
        f"truncation moved the mean to {got_mean}; raise MAX_OPPORTUNITIES"
    )


# --- the covariance identity ------------------------------------------

def test_joint_moments_match_simulation():
    """
    Simulate the actual generative process -- draw N, then allocate
    multinomially -- and check every moment the closed form claims.
    """
    rng = np.random.default_rng(0)
    support = np.arange(0, MAX_OPPORTUNITIES + 1)
    vol = negbin_pmf(32.3, 63.8, support)
    p_i, p_j, p_rest = 0.25, 0.18, 0.57

    n_sim = 400_000
    N = rng.choice(support, size=n_sim, p=vol)
    draws = np.array([rng.multinomial(n, [p_i, p_j, p_rest]) for n in N[:40_000]])
    Ti, Tj = draws[:, 0], draws[:, 1]

    m = joint_moments(vol, support, p_i, p_j)
    assert Ti.mean() == pytest.approx(m["E_i"], rel=0.03)
    assert Tj.mean() == pytest.approx(m["E_j"], rel=0.03)
    assert Ti.var() == pytest.approx(m["Var_i"], rel=0.06)
    assert np.cov(Ti, Tj)[0, 1] == pytest.approx(m["Cov_ij"], rel=0.15, abs=0.05)


def test_teammates_positively_correlated_when_volume_overdispersed():
    """
    The counter-intuitive prediction, and the one that was checked against
    2,009 real pairs. Var(N) > E[N] implies Cov > 0.
    """
    support = np.arange(0, MAX_OPPORTUNITIES + 1)
    vol = negbin_pmf(32.3, 63.8, support)       # var/mean 1.97, the real ratio
    assert teammate_correlation(vol, support, 0.25, 0.18) > 0


def test_teammates_negatively_correlated_when_volume_is_fixed():
    """
    The other side of the identity: with N deterministic, Var(N) = 0, and
    the allocation is purely zero-sum.
    """
    support = np.arange(0, MAX_OPPORTUNITIES + 1)
    vol = np.zeros(len(support))
    vol[35] = 1.0
    assert teammate_correlation(vol, support, 0.25, 0.18) < 0


def test_covariance_crosses_zero_at_var_equals_mean():
    support = np.arange(0, MAX_OPPORTUNITIES + 1)
    vol = negbin_pmf(30.0, 30.0, support)       # Poisson: Var == E
    assert abs(teammate_correlation(vol, support, 0.25, 0.18)) < 0.02


# --- marginal pmf ------------------------------------------------------

def test_opportunity_pmf_mean_is_volume_times_share():
    support = np.arange(0, MAX_OPPORTUNITIES + 1)
    vol = negbin_pmf(32.0, 63.0, support)
    pmf = opportunity_pmf(vol, support, 0.25)
    out = np.arange(0, MAX_OPPORTUNITIES + 1)
    assert pmf @ out == pytest.approx(32.0 * 0.25, rel=0.02)


def test_opportunity_pmf_is_a_distribution():
    support = np.arange(0, MAX_OPPORTUNITIES + 1)
    vol = negbin_pmf(32.0, 63.0, support)
    pmf = opportunity_pmf(vol, support, 0.2)
    assert pmf.sum() == pytest.approx(1.0)
    assert (pmf >= 0).all()


def test_opportunity_pmf_wider_than_binomial_alone():
    """
    Volume uncertainty must ADD variance. A model that conditioned on a
    point estimate of N would be overconfident -- the V4 reliability
    finding, where too-wide spread showed up as confident and wrong at the
    extremes, in reverse.
    """
    support = np.arange(0, MAX_OPPORTUNITIES + 1)
    share = 0.25
    vol = negbin_pmf(32.0, 63.0, support)
    pmf = opportunity_pmf(vol, support, share)
    out = np.arange(0, MAX_OPPORTUNITIES + 1)
    mean = pmf @ out
    var = pmf @ (out ** 2) - mean ** 2
    binom_only = 32.0 * share * (1 - share)
    assert var > binom_only * 1.2


def test_zero_share_is_a_point_mass_at_zero():
    support = np.arange(0, MAX_OPPORTUNITIES + 1)
    vol = negbin_pmf(32.0, 63.0, support)
    pmf = opportunity_pmf(vol, support, 0.0)
    assert pmf[0] == pytest.approx(1.0)


def test_opportunity_pmf_rejects_bad_share():
    support = np.arange(0, MAX_OPPORTUNITIES + 1)
    vol = negbin_pmf(32.0, 63.0, support)
    with pytest.raises(ValueError):
        opportunity_pmf(vol, support, 1.5)


# --- fitting -----------------------------------------------------------

def _fake_team_games(n_weeks=17, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    true = {"AAA": 40.0, "BBB": 32.0, "CCC": 24.0}
    for team, mu in true.items():
        for wk in range(1, n_weeks + 1):
            rows.append({"season": 2025, "week": wk, "team": team,
                         "N": int(rng.poisson(mu))})
    return pl.DataFrame(rows), true


def test_volume_model_recovers_team_ordering():
    tg, true = _fake_team_games()
    m = TeamVolumeModel.fit(tg, trailing=17)
    assert m.mean_for("AAA") > m.mean_for("BBB") > m.mean_for("CCC")


def test_volume_model_shrinks_toward_league():
    """Every shrunk mean must sit between the raw mean and the league mean."""
    tg, true = _fake_team_games()
    m = TeamVolumeModel.fit(tg, trailing=17)
    for team, mu in true.items():
        raw = tg.filter(pl.col("team") == team)["N"].mean()
        lo, hi = sorted([raw, m.league_mean])
        assert lo - 1e-6 <= m.mean_for(team) <= hi + 1e-6


def test_volume_model_unknown_team_falls_back_to_league():
    tg, _ = _fake_team_games()
    m = TeamVolumeModel.fit(tg, trailing=17)
    assert m.mean_for("ZZZ") == pytest.approx(m.league_mean)


def test_volume_model_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        TeamVolumeModel.fit(pl.DataFrame({"team": ["A"], "N": [30]}))


def test_volume_pmf_market_tilt_moves_the_mean():
    tg, _ = _fake_team_games()
    m = TeamVolumeModel.fit(tg, trailing=17)
    support = m.support
    base = m.pmf("AAA") @ support
    tilted = m.pmf("AAA", market_tilt=3.0) @ support
    assert tilted - base == pytest.approx(3.0, abs=0.2)


def _fake_player_games(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    roster = [("p1", "WR", 0.30), ("p2", "WR", 0.22), ("p3", "TE", 0.18), ("p4", "RB", 0.12)]
    for wk in range(1, 18):
        N = rng.poisson(33)
        alloc = rng.multinomial(N, [r[2] for r in roster] + [1 - sum(r[2] for r in roster)])
        for (pid, pos, _), got in zip(roster, alloc):
            rows.append({"season": 2025, "week": wk, "team": "AAA",
                         "player_id": pid, "position": pos, "opportunities": int(got)})
    return pl.DataFrame(rows)


def test_share_model_recovers_ordering_and_normalises():
    pg = _fake_player_games()
    s = ShareModel.fit(pg, trailing=17)
    v = s.team_vector("AAA", ["p1", "p2", "p3", "p4"])
    assert v.sum() == pytest.approx(1.0)
    assert v[0] > v[1] > v[2] > v[3]


def test_share_model_unknown_player_is_zero():
    pg = _fake_player_games()
    s = ShareModel.fit(pg, trailing=17)
    assert s.share_for("AAA", "nobody") == 0.0


def test_share_model_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        ShareModel.fit(pl.DataFrame({"team": ["A"], "opportunities": [3]}))


def test_team_vector_handles_all_unknown_players():
    pg = _fake_player_games()
    s = ShareModel.fit(pg, trailing=17)
    v = s.team_vector("AAA", ["x", "y"])
    assert v.sum() == pytest.approx(1.0)
    assert v[0] == pytest.approx(0.5)
