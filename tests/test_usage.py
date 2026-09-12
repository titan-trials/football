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


def _fake_player_games(seed=0, n_teams=1):
    """
    n_teams > 1 so that each (role_pos, role_bucket) cell clears the
    5-player floor ShareModel.fit requires before it will trust a cell.
    A one-team fixture cannot exercise the role prior at all -- the cells
    are too thin and the model correctly refuses them.
    """
    rng = np.random.default_rng(seed)
    rows = []
    roster = [("p1", "WR", 0.30), ("p2", "WR", 0.22), ("p3", "TE", 0.18), ("p4", "RB", 0.12)]
    for t in range(n_teams):
        team = f"T{t:02d}"
        for wk in range(1, 18):
            N = rng.poisson(33)
            alloc = rng.multinomial(N, [r[2] for r in roster] + [1 - sum(r[2] for r in roster)])
            for (pid, pos, _), got in zip(roster, alloc):
                rows.append({"season": 2025, "week": wk, "team": team,
                             "player_id": f"{team}_{pid}", "position": pos,
                             "opportunities": int(got)})
    return pl.DataFrame(rows)


def _fake_roles(n_teams):
    buckets = {"p1": ("WR", 1), "p2": ("WR", 2), "p3": ("TE", 1), "p4": ("RB", 1)}
    return pl.DataFrame([
        {"team": f"T{t:02d}", "player_id": f"T{t:02d}_{pid}",
         "role_pos": rp, "role_bucket": rb}
        for t in range(n_teams) for pid, (rp, rb) in buckets.items()
    ])


def test_share_model_recovers_ordering_and_normalises():
    pg = _fake_player_games()
    s = ShareModel.fit(pg, trailing=17)
    v = s.team_vector("T00", ["T00_p1", "T00_p2", "T00_p3", "T00_p4"])
    assert v.sum() == pytest.approx(1.0)
    assert v[0] > v[1] > v[2] > v[3]


def test_unknown_player_gets_role_prior_not_zero():
    """
    THE REGRESSION TEST for the bug validation found: an unseen player used
    to return share 0.0, a point mass at zero, and the model called a rookie
    WR1 impossible. He must now get his depth-chart role prior.
    """
    pg = _fake_player_games(n_teams=8)
    s = ShareModel.fit(pg, trailing=17, roles=_fake_roles(8))
    assert s.role_priors, "role priors must actually be populated"
    rookie = s.rate_for("T00", "never_seen", role=("WR", 1))
    assert rookie > 0.0, "an unseen WR1 must not be a point mass at zero"
    deep = s.rate_for("T00", "never_seen_2", role=("WR", 2))
    assert rookie > deep, "role prior must be monotone in depth rank"


def test_thin_role_cell_is_refused_not_invented():
    """A cell with under 5 players must fall back, not fit its own prior."""
    pg = _fake_player_games(n_teams=1)
    s = ShareModel.fit(pg, trailing=17, roles=_fake_roles(1))
    assert s.role_priors == {}


def test_unknown_player_falls_back_through_position_then_global():
    pg = _fake_player_games()
    s = ShareModel.fit(pg, trailing=17)
    by_pos = s.rate_for("T00", "x", position="WR")
    by_none = s.rate_for("T00", "y")
    assert by_pos > 0 and by_none > 0
    assert by_none == pytest.approx(s.global_rate, rel=1e-6)


def test_zero_history_shrink_returns_exactly_the_prior():
    """shrink(0, 0, prior, k) = prior. No special case should exist."""
    pg = _fake_player_games()
    s = ShareModel.fit(pg, trailing=17)
    assert s.rate_for("T00", "nobody", position="WR") == pytest.approx(
        s.position_priors["WR"], rel=1e-9)


def test_share_model_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        ShareModel.fit(pl.DataFrame({"team": ["A"], "opportunities": [3]}))


def test_team_vector_handles_all_unknown_players():
    pg = _fake_player_games()
    s = ShareModel.fit(pg, trailing=17)
    v = s.team_vector("T00", ["x", "y"])
    assert v.sum() == pytest.approx(1.0)
    assert v[0] == pytest.approx(0.5)


def test_team_vector_ranks_unknown_players_by_role():
    """A team of all-unseen players must still be ordered by depth chart."""
    pg = _fake_player_games(n_teams=8)
    s = ShareModel.fit(pg, trailing=17, roles=_fake_roles(8))
    v = s.team_vector("T00", ["new1", "new2"], roles=[("WR", 1), ("WR", 2)])
    assert v.sum() == pytest.approx(1.0)
    assert v[0] > v[1]


# --- era drift ---------------------------------------------------------

def _drifting_team_games(seed=0):
    """Volume declining season over season, as it really does: 33.2 -> 30.3."""
    rng = np.random.default_rng(seed)
    rows = []
    means = {2021: 33.0, 2022: 32.0, 2023: 31.5, 2024: 31.0, 2025: 30.0}
    for season, mu in means.items():
        for team in [f"T{i:02d}" for i in range(12)]:
            for wk in range(1, 18):
                rows.append({"season": season, "week": wk, "team": team,
                             "N": int(rng.poisson(mu))})
    return pl.DataFrame(rows)


def test_league_prior_uses_trailing_seasons_not_all_history():
    """
    THE ERA-DRIFT REGRESSION TEST. Pooling every season puts the league
    prior above the current one, which measured +6.9% high on real data and
    biased predicted volume by +1.26 targets.
    """
    tg = _drifting_team_games()
    pooled = TeamVolumeModel.fit(tg, trailing=8, prior_seasons=0)
    trailing = TeamVolumeModel.fit(tg, trailing=8, prior_seasons=2)
    assert trailing.league_mean < pooled.league_mean
    assert trailing.league_mean == pytest.approx(30.5, abs=0.6)
    assert pooled.league_mean == pytest.approx(31.5, abs=0.6)


def test_trailing_prior_falls_back_when_window_too_thin():
    """A window with under 50 rows must widen, not fit a variance from noise."""
    tg = pl.DataFrame([{"season": 2025, "week": w, "team": "AAA", "N": 30}
                       for w in range(1, 18)])
    m = TeamVolumeModel.fit(tg, trailing=8, prior_seasons=1)
    assert m.league_mean == pytest.approx(30.0)


# ---------------------------------------------------------------------
# Share dispersion (beta-binomial)
# ---------------------------------------------------------------------

def test_betabinomial_collapses_to_the_binomial_at_high_concentration():
    """
    The old model must be a LIMIT of the new one, not a sibling of it.

    As c -> infinity the Beta(c*p, c*(1-p)) collapses to a point mass at p
    and the beta-binomial becomes the binomial. If that limit does not
    hold, `share_concentration=None` and a large concentration would give
    different answers and no A/B could be trusted.
    """
    from features.usage import negbin_pmf, opportunity_pmf

    support = np.arange(0, 81)
    vol = negbin_pmf(32.0, 63.0, support)
    fixed = opportunity_pmf(vol, support, 0.22)
    nearly = opportunity_pmf(vol, support, 0.22, share_concentration=1e7)
    assert np.max(np.abs(fixed - nearly)) < 1e-6


def test_share_dispersion_widens_without_moving_the_mean():
    """
    E[T] = E[N] * p under both models, because E[p] = p for the beta. Only
    the spread changes. This is what makes the change safe to ship: it
    cannot reintroduce a bias, and any bias that appears after turning it
    on is a bug elsewhere.

    The analytic target:

        Var(T) = p^2 Var(N) + p(1-p) E[N] + (E[N^2] - E[N]) * Var(p)
        Var(p) = p(1-p) / (c + 1)
    """
    from features.usage import negbin_pmf, opportunity_pmf

    support = np.arange(0, 81)
    mean_n, var_n, p, c = 32.0, 63.0, 0.22, 40.0
    vol = negbin_pmf(mean_n, var_n, support)
    EN = float(vol @ support)
    EN2 = float(vol @ (support ** 2))

    def moments(pmf):
        s = np.arange(len(pmf))
        m = float(pmf @ s)
        return m, float(pmf @ (s ** 2)) - m ** 2

    m_fix, v_fix = moments(opportunity_pmf(vol, support, p))
    m_bb, v_bb = moments(opportunity_pmf(vol, support, p, share_concentration=c))

    assert abs(m_bb - m_fix) < 1e-6, "the beta-binomial moved the mean"
    assert abs(m_bb - EN * p) < 1e-6, "mean is not E[N] * p"

    var_p = p * (1 - p) / (c + 1)
    expected = v_fix + (EN2 - EN) * var_p
    assert abs(v_bb - expected) < 1e-4, (
        f"variance {v_bb:.4f} does not match the law-of-total-variance "
        f"prediction {expected:.4f}")
    assert v_bb > v_fix, "dispersion did not widen the distribution"


def test_share_concentration_recovers_a_known_dispersion():
    """
    Generate shares from a KNOWN beta, run the estimator, and check it gets
    the concentration back. Without this the estimator could be off by the
    sampling-noise subtraction -- the one step that is easy to get wrong and
    impossible to notice, because forgetting it inflates the variance in
    the direction we already want it to move.
    """
    from features.usage import estimate_share_concentration

    rng = np.random.default_rng(7)
    true_c, p, n_players, n_games, N = 60.0, 0.20, 400, 10, 34
    rows = []
    for i in range(n_players):
        share = rng.beta(true_c * p, true_c * (1 - p))
        for g in range(n_games):
            t = rng.binomial(N, share)
            rows.append({"season": 2025, "week": g + 1, "team": f"T{i}",
                         "player_id": f"p{i}", "opportunities": float(t),
                         "role_pos": "WR", "role_bucket": 1})
            # a filler player so the team total is N, making the realised
            # share exactly t / N as the estimator assumes
            rows.append({"season": 2025, "week": g + 1, "team": f"T{i}",
                         "player_id": f"filler{i}", "opportunities": float(N - t),
                         "role_pos": "WR", "role_bucket": 2})

    by_role, pooled = estimate_share_concentration(
        pl.DataFrame(rows), trailing=n_games, min_games=4, min_players=5)
    got = by_role[("WR", 1)]
    assert 0.6 * true_c < got < 1.7 * true_c, (
        f"recovered concentration {got:.1f} is not close to the true {true_c}")


def test_share_concentration_is_not_estimated_without_roles():
    """
    Concentration is keyed on depth-chart role. Without role columns the
    estimator must return nothing rather than silently pooling every
    position into one number -- a WR1's share volatility and a QB1's are
    not the same quantity.
    """
    from features.usage import estimate_share_concentration

    pg = pl.DataFrame({
        "season": [2025] * 4, "week": [1, 2, 3, 4], "team": ["KC"] * 4,
        "player_id": ["a"] * 4, "opportunities": [5.0, 6.0, 4.0, 7.0],
    })
    by_role, pooled = estimate_share_concentration(pg)
    assert by_role == {}
    assert not np.isfinite(pooled)
