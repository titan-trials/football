"""
Tests for features/efficiency.py.

The load-bearing ones are the convolution identities. `yards_pmf` is an
exact discrete convolution, so its mean and variance have closed forms and
must match them to floating-point precision -- not approximately, the way a
simulation would. A convolution that is subtly misaligned by one index
still produces a plausible-looking distribution, which is precisely the
class of bug that ships.
"""
import numpy as np
import polars as pl
import pytest

from features.efficiency import (
    ADOT_CENTRES, PT_SUPPORT, TOTAL_SUPPORT, AdotModel, PerTargetModel,
    adot_band, prob_over, retained_mass, yards_pmf,
)


# --- band assignment ---------------------------------------------------

def test_adot_band_edges():
    assert adot_band(-2.0) == 0
    assert adot_band(4.9) == 0
    assert adot_band(5.0) == 1
    assert adot_band(10.9) == 2
    assert adot_band(11.0) == 3
    assert adot_band(50.0) == len(ADOT_CENTRES) - 1


# --- fitting -----------------------------------------------------------

def _fake_targets(n=20000, seed=0):
    """
    Deeper targets are caught less often and go further when caught -- the
    real structure, planted so the fit can be checked against it.
    """
    rng = np.random.default_rng(seed)
    adot = rng.uniform(0, 18, n)
    p_catch = np.clip(0.85 - 0.018 * adot, 0.3, 0.95)
    caught = rng.random(n) < p_catch
    yards = np.where(caught, np.round(rng.gamma(2.0, (4 + 0.5 * adot) / 2.0)), 0.0)
    return pl.DataFrame({"player_adot": adot, "yards_gained": yards})


def test_fit_recovers_planted_structure():
    m = PerTargetModel.fit(_fake_targets(), outcome_col="yards_gained", shape_col="player_adot")
    s = m.summary().to_pandas()
    assert s["p_zero"].is_monotonic_increasing, "deeper targets must be caught less"
    assert s["p_20plus"].is_monotonic_increasing, "deeper catches must go further"


def test_band_pmfs_are_distributions():
    m = PerTargetModel.fit(_fake_targets(), outcome_col="yards_gained", shape_col="player_adot")
    for b in range(len(ADOT_CENTRES)):
        assert m.band_pmfs[b].sum() == pytest.approx(1.0)
        assert (m.band_pmfs[b] >= 0).all()


def test_thin_band_borrows_pooled_rather_than_fitting_itself():
    """A band with too few targets must not invent its own distribution."""
    d = pl.DataFrame({"player_adot": [3.0] * 5000 + [16.0] * 10,
                      "yards_gained": [5.0] * 5000 + [80.0] * 10})
    m = PerTargetModel.fit(d, outcome_col="yards_gained", shape_col="player_adot")
    deep = m.band_pmfs[len(ADOT_CENTRES) - 1]
    assert deep[PT_SUPPORT == 80].sum() < 0.5, "thin band fitted itself"


def test_fit_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing outcome column"):
        PerTargetModel.fit(pl.DataFrame({"player_adot": [5.0]}),
                           outcome_col="yards_gained", shape_col="player_adot")


# --- interpolation -----------------------------------------------------

def test_interpolation_is_continuous_across_a_band_edge():
    """
    Snapping to a band would make a receiver at 10.9 and one at 11.1 get
    materially different tails. Interpolating must not.
    """
    m = PerTargetModel.fit(_fake_targets(), outcome_col="yards_gained", shape_col="player_adot")
    lo = m.pmf_for(10.9)
    hi = m.pmf_for(11.1)
    assert abs((lo @ PT_SUPPORT) - (hi @ PT_SUPPORT)) < 0.1


def test_interpolation_is_monotone_in_adot():
    m = PerTargetModel.fit(_fake_targets(), outcome_col="yards_gained", shape_col="player_adot")
    means = [m.pmf_for(a) @ PT_SUPPORT for a in [2.0, 6.0, 10.0, 13.0, 17.0]]
    assert all(b >= a - 1e-9 for a, b in zip(means, means[1:]))


def test_interpolation_clamps_outside_the_range():
    m = PerTargetModel.fit(_fake_targets(), outcome_col="yards_gained", shape_col="player_adot")
    np.testing.assert_allclose(m.pmf_for(-50.0), m.pmf_for(ADOT_CENTRES[0]))
    np.testing.assert_allclose(m.pmf_for(99.0), m.pmf_for(ADOT_CENTRES[-1]))


def test_interpolated_pmf_is_a_distribution():
    m = PerTargetModel.fit(_fake_targets(), outcome_col="yards_gained", shape_col="player_adot")
    p = m.pmf_for(7.3)
    assert p.sum() == pytest.approx(1.0)
    assert (p >= 0).all()


# --- the convolution ---------------------------------------------------

def _delta_target_pmf(k):
    """Exactly k targets, with certainty."""
    sup = np.arange(0, 30)
    pmf = np.zeros(len(sup))
    pmf[k] = 1.0
    return pmf, sup


def test_convolution_mean_is_exactly_k_times_per_target_mean():
    m = PerTargetModel.fit(_fake_targets(), outcome_col="yards_gained", shape_col="player_adot")
    pt = m.pmf_for(9.0)
    mu = pt @ PT_SUPPORT
    for k in [1, 3, 7, 12]:
        tp, sup = _delta_target_pmf(k)
        got = yards_pmf(tp, sup, pt) @ TOTAL_SUPPORT
        assert got == pytest.approx(k * mu, rel=1e-9), f"k={k}"


def test_convolution_variance_is_exactly_k_times_per_target_variance():
    m = PerTargetModel.fit(_fake_targets(), outcome_col="yards_gained", shape_col="player_adot")
    pt = m.pmf_for(9.0)
    mu = pt @ PT_SUPPORT
    var1 = pt @ (PT_SUPPORT ** 2) - mu ** 2
    for k in [1, 4, 9]:
        tp, sup = _delta_target_pmf(k)
        p = yards_pmf(tp, sup, pt)
        mean = p @ TOTAL_SUPPORT
        var = p @ (TOTAL_SUPPORT ** 2) - mean ** 2
        assert var == pytest.approx(k * var1, rel=1e-8), f"k={k}"


def test_zero_targets_is_a_point_mass_at_zero():
    m = PerTargetModel.fit(_fake_targets(), outcome_col="yards_gained", shape_col="player_adot")
    tp, sup = _delta_target_pmf(0)
    p = yards_pmf(tp, sup, m.pmf_for(9.0))
    assert p[TOTAL_SUPPORT == 0] == pytest.approx(1.0)


def test_one_target_reproduces_the_per_target_distribution():
    """The k=1 case must be the input, re-indexed. Catches off-by-one."""
    m = PerTargetModel.fit(_fake_targets(), outcome_col="yards_gained", shape_col="player_adot")
    pt = m.pmf_for(9.0)
    tp, sup = _delta_target_pmf(1)
    p = yards_pmf(tp, sup, pt)
    for y in [-5, 0, 3, 12, 40]:
        assert p[TOTAL_SUPPORT == y].sum() == pytest.approx(
            pt[PT_SUPPORT == y].sum(), abs=1e-12), f"mismatch at {y} yards"


def test_yards_pmf_is_a_distribution():
    m = PerTargetModel.fit(_fake_targets(), outcome_col="yards_gained", shape_col="player_adot")
    tp = np.array([0.1, 0.2, 0.3, 0.25, 0.15])
    sup = np.arange(0, 5)
    p = yards_pmf(tp, sup, m.pmf_for(9.0))
    assert p.sum() == pytest.approx(1.0)
    assert (p >= -1e-15).all()


def test_mixing_over_targets_matches_manual_mixture():
    """The mixture must be linear in the target pmf."""
    m = PerTargetModel.fit(_fake_targets(), outcome_col="yards_gained", shape_col="player_adot")
    pt = m.pmf_for(9.0)
    sup = np.arange(0, 30)
    mix = np.zeros(len(sup)); mix[2] = 0.4; mix[5] = 0.6
    got = yards_pmf(mix, sup, pt)

    a, _ = _delta_target_pmf(2)
    b, _ = _delta_target_pmf(5)
    want = 0.4 * yards_pmf(a, sup, pt) + 0.6 * yards_pmf(b, sup, pt)
    np.testing.assert_allclose(got, want, atol=1e-12)


def test_truncation_retains_essentially_all_mass():
    """
    team-run-model.md's MAX_RUNS lesson. Checked at a volume well above any
    real receiver, since a cap tuned to the cases you tested is a cap that
    fails on the case you did not.
    """
    m = PerTargetModel.fit(_fake_targets(), outcome_col="yards_gained", shape_col="player_adot")
    pt = m.pmf_for(ADOT_CENTRES[-1])
    tp, sup = _delta_target_pmf(20)
    assert retained_mass(tp, sup, pt) > 0.9999


def test_prob_over_is_strict():
    pmf = np.zeros(len(TOTAL_SUPPORT))
    pmf[TOTAL_SUPPORT == 45] = 0.5
    pmf[TOTAL_SUPPORT == 46] = 0.5
    assert prob_over(pmf, 45.5) == pytest.approx(0.5)
    assert prob_over(pmf, 45.0) == pytest.approx(0.5)
    assert prob_over(pmf, 44.0) == pytest.approx(1.0)


# --- AdotModel ---------------------------------------------------------

def _fake_adot_games(n_teams=8, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    roster = [("p1", ("WR", 1), 12.0), ("p2", ("WR", 2), 9.0),
              ("p3", ("TE", 1), 7.0), ("p4", ("RB", 1), 1.0)]
    for t in range(n_teams):
        team = f"T{t:02d}"
        for wk in range(1, 18):
            for pid, _, mu in roster:
                n = int(rng.poisson(6)) + 1
                rows.append({"season": 2025, "week": wk, "team": team,
                             "player_id": f"{team}_{pid}", "shape_n": n,
                             "shape_sum": float(rng.normal(mu * n, 3 * np.sqrt(n)))})
    roles = pl.DataFrame([
        {"team": f"T{t:02d}", "player_id": f"T{t:02d}_{pid}",
         "role_pos": rp, "role_bucket": rb}
        for t in range(n_teams) for pid, (rp, rb), _ in roster
    ])
    return pl.DataFrame(rows), roles


def test_adot_model_recovers_ordering():
    pg, roles = _fake_adot_games()
    m = AdotModel.fit(pg, roles=roles, trailing=17)
    a = [m.value_for("T00", f"T00_{p}") for p in ["p1", "p2", "p3", "p4"]]
    assert a[0] > a[1] > a[2] > a[3]


def test_adot_unknown_player_gets_role_prior_not_league():
    pg, roles = _fake_adot_games()
    m = AdotModel.fit(pg, roles=roles, trailing=17)
    deep = m.value_for("T00", "rookie", role=("WR", 1))
    short = m.value_for("T00", "rookie2", role=("RB", 1))
    assert deep > short
    assert deep != pytest.approx(m.league_value)


def test_adot_unknown_with_no_role_falls_back_to_league():
    pg, roles = _fake_adot_games()
    m = AdotModel.fit(pg, roles=roles, trailing=17)
    assert m.value_for("T00", "nobody") == pytest.approx(m.league_value, rel=1e-9)


def test_adot_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing columns"):
        AdotModel.fit(pl.DataFrame({"team": ["A"], "shape_n": [3]}))
