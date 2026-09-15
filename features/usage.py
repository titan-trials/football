"""
Stage 1: the usage model.

WHAT THIS IS
------------
An opportunity distribution per player per game -- how many targets, how
many carries -- as a pmf, not a mean. `team-run-model.md` states the reason
in four words: a mean is not a distribution. A prop line at 5.5 receptions
is a question about the tail, and a point estimate cannot answer it.

THE DECOMPOSITION, AND WHY IT IS THIS ONE
-----------------------------------------
    targets_i  =  N  x  p_i

        N   = team pass volume that game        (team-level, volatile)
        p_i = player's share of it              (player-level, stable)

This is not the only way to model opportunity, but it is the one that makes
three separate problems fall out of a single structure.

  1. It puts the signal where the signal is. Measured on 26,292
     player-weeks: targets have split-half reliability 0.863 and lag-1 week
     correlation 0.525. Team pass volume, by contrast, has an out-of-sample
     R^2 of 0.0896 against its own trailing mean. Almost everything
     forecastable about a player's opportunity is his SHARE, not his
     team's volume. Modelling the product directly buries that.

  2. It generates teammate dependence instead of bolting it on. Given
     N, allocating to k receivers is multinomial, so

         Var(T_i)     = E[N] p_i(1-p_i) + Var(N) p_i^2
         Cov(T_i,T_j) = p_i p_j ( Var(N) - E[N] )        i != j

     Both signs are then predictions, not assumptions, and both were
     checked against 2,009 real teammate pairs. See the module test.

  3. It isolates the market question to one term. The closing line plausibly
     moves N. It has no mechanism to move p_i. So "does the market help"
     becomes a question about one model, answerable separately -- which is
     how MARKET_IN_VOLUME got measured and switched off.

WHAT THE COVARIANCE CHECK FOUND
-------------------------------
Measured on 2,009 teammate pairs, team-seasons with >=12 weeks, players
above 8% target share:

    both players active, raw targets      mean Cov  +0.483   41% negative
    fixed-share multinomial predicts      mean Cov  +0.583   14% negative
    all weeks, injuries included          mean Cov  +0.122   47% negative
    residual, conditional on realised N   mean rho  -0.147   69% negative

Read those four lines in order, because they say different things:

  * Teammates are POSITIVELY correlated in raw targets. That surprises
    people, and it is just the overdispersion: Var(N)=63.8 against
    E[N]=32.3, so Var(N)-E[N] > 0 and the identity above predicts a
    positive sign. A team that drops back 45 times feeds everybody.
  * The fixed-share multinomial gets that roughly right (+0.58 vs +0.48),
    slightly over-predicting.
  * Injuries drag the unconditional covariance down by -0.36. That is a
    real effect and a DIFFERENT mechanism -- availability, not allocation.
    Do not model it here; model it as availability.
  * Conditional on the realised volume, residuals are NEGATIVELY
    correlated, -0.147. This is the reallocation effect: given 38 dropbacks,
    a target to one receiver is a target not thrown to another.

The practical consequence is the opposite of the usual warning, and it is
the reason the recon doc's claim about this was wrong. For evaluation,

    Var(mean of k)  =  sigma^2/k * (1 + (k-1) rho)

so NEGATIVE rho makes k correlated rows worth MORE than k independent ones,
not fewer: at rho = -0.147, three teammates are worth 4.24 independent rows.
But that only holds conditional on getting N right. Miss the volume and the
positive unconditional correlation applies instead, and all three rows are
wrong together.

Which is the honest summary: the sign of teammate correlation depends on
whether your error is in volume or in allocation. So resample GAMES, not
rows -- that captures both regimes without having to pick one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import polars as pl

from config import PRIOR_TRAILING_SEASONS
from features.shrinkage import estimate_prior_strength_counts, shrink

MAX_OPPORTUNITIES = 80  # support cap for every pmf produced here


# ---------------------------------------------------------------------
# Team volume
# ---------------------------------------------------------------------

def negbin_pmf(mean: float, var: float, support: np.ndarray) -> np.ndarray:
    """
    Negative binomial by method of moments, as in team-run-model.py.

        r = mean^2 / (var - mean)        p = r / (r + mean)

    Team pass volume is overdispersed -- measured var/mean 1.97 on 3,742
    team-games -- so Poisson is wrong in the tail, which is the only part a
    prop line asks about. Falls back to Poisson when var <= mean, which
    happens on thin samples and is the right degenerate case.

    MAX_OPPORTUNITIES matters here. team-run-model.md's gotcha: truncating
    the tail and renormalising pulls the mean DOWN, invisibly at small
    means and materially at large ones. 80 is far enough out that the
    discarded mass is negligible for any real team; do not lower it to
    "tighten" anything.
    """
    from scipy.stats import nbinom, poisson

    if not np.isfinite(mean) or mean <= 0:
        raise ValueError(f"mean must be positive and finite, got {mean}")
    if var <= mean * 1.0001:
        probs = poisson.pmf(support, mean)
    else:
        r = mean ** 2 / (var - mean)
        p = r / (r + mean)
        probs = nbinom.pmf(support, r, p)
    total = probs.sum()
    if total <= 0:
        raise ValueError("degenerate pmf")
    return probs / total


@dataclass
class TeamVolumeModel:
    """
    Distribution of team pass attempts (or rush attempts) for one game.

    Fit from a team's own trailing games, shrunk toward the league mean.
    The dispersion is pooled across the league rather than fitted per team,
    on the same reasoning team-run-model.py used for runs: "teams differ in
    how many they get, not in how lumpy it is." With 17 games a season a
    per-team dispersion estimate is noise.
    """
    support: np.ndarray
    league_mean: float
    league_var: float
    team_means: dict
    prior_k: float

    @classmethod
    def fit(cls, team_games: pl.DataFrame, value_col: str = "N",
            trailing: int = 8, max_support: int = MAX_OPPORTUNITIES,
            prior_seasons: int = PRIOR_TRAILING_SEASONS) -> "TeamVolumeModel":
        """
        team_games: one row per team-game with columns season, week, team,
        and `value_col`. Rows must already be restricted to games the caller
        is allowed to see -- this class does no as-of filtering of its own.

        prior_seasons: how many recent seasons form the LEAGUE prior. This
        is not a tuning knob, it is a correctness fix.

        MEASURED 2026-09-11. League mean team targets per game:

            2019  33.21    2022  31.70    2025  30.28
            2020  33.54    2023  31.92
            2021  32.94    2024  31.09

        Monotone decline. A prior pooled over 2019-2024 sits at 32.38
        against an actual 2025 of 30.28 -- **+6.9% high**. The baseball
        project measured all-history strikeout rates running **+6.8%** high
        against a trailing 12 months and fixed it the same way; this is the
        same error to within a tenth of a point.

        Left unfixed it showed up as a +1.26 target volume bias and a
        -0.093 calibration gap in the top probability bucket. A trailing
        two-season prior cuts the drift to +1.3%.
        """
        need = {"season", "week", "team", value_col}
        missing = need - set(team_games.columns)
        if missing:
            raise ValueError(f"team_games missing columns: {sorted(missing)}")

        seasons = sorted(team_games["season"].unique().to_list())
        keep = seasons[-prior_seasons:] if prior_seasons > 0 else seasons
        recent_rows = team_games.filter(pl.col("season").is_in(keep))
        # Fall back to everything if the trailing window is too thin to
        # estimate a variance from -- early in the first season of a cache.
        if recent_rows.height < 50:
            recent_rows = team_games

        vals = recent_rows[value_col].to_numpy().astype(float)
        league_mean = float(np.mean(vals))
        league_var = float(np.var(vals, ddof=1))

        df = team_games.sort(["team", "season", "week"])
        recent = (
            df.group_by("team", maintain_order=True)
              .agg([
                  pl.col(value_col).tail(trailing).sum().alias("_sum"),
                  pl.col(value_col).tail(trailing).len().alias("_n"),
                  pl.col(value_col).tail(trailing).var().alias("_var"),
              ])
        )
        k = estimate_prior_strength_counts(
            recent["_sum"].to_numpy(),
            recent["_n"].to_numpy(),
            np.nan_to_num(recent["_var"].to_numpy(), nan=league_var),
            min_trials=3,
        )
        shrunk = shrink(recent["_sum"].to_numpy(), recent["_n"].to_numpy(), league_mean, k)
        team_means = dict(zip(recent["team"].to_list(), shrunk.tolist()))

        return cls(
            support=np.arange(0, max_support + 1),
            league_mean=league_mean,
            league_var=league_var,
            team_means=team_means,
            prior_k=float(k),
        )

    def mean_for(self, team: str) -> float:
        return float(self.team_means.get(team, self.league_mean))

    def pmf(self, team: str, market_tilt: float = 0.0) -> np.ndarray:
        """
        pmf over team volume. `market_tilt` shifts the mean by that many
        attempts and is only ever non-zero when model_flags.MARKET_IN_VOLUME
        is on -- see that flag for why it is off and what it measured.

        Dispersion is carried from the league: the variance is scaled so
        var/mean matches the league ratio at this team's mean.
        """
        mean = max(self.mean_for(team) + market_tilt, 0.5)
        ratio = max(self.league_var / self.league_mean, 1.0001)
        return negbin_pmf(mean, mean * ratio, self.support)


# ---------------------------------------------------------------------
# Player share
# ---------------------------------------------------------------------

def estimate_share_concentration(player_games: pl.DataFrame,
                                 value_col: str = "opportunities",
                                 trailing: int = 8,
                                 min_games: int = 4,
                                 min_players: int = 8,
                                 floor: float = 1.0,
                                 cap: float = 400.0) -> tuple[dict, float]:
    """
    How volatile a player's SHARE is, as a beta concentration, per role.

    Returns ({(role_pos, role_bucket): c}, pooled_c).

    THE FORMULA
    -----------
    For one player with games g = 1..n, realised share s_g = T_g / N_g and
    pooled rate p = sum(T_g) / sum(N_g):

        observed spread      V_obs      = Var(s_g)
        sampling alone       V_sampling = mean( p(1-p) / N_g )
        real share spread    V_p        = V_obs - V_sampling
        concentration        c          = p(1-p) / V_p  -  1

    The subtraction is the whole point and it is the same
    within-versus-between decomposition as `estimate_prior_strength`. A
    player's realised shares bounce around even if his true share never
    moves, purely because 6 targets out of 34 dropbacks is a small sample.
    Charging that bounce to Var(p) would double-count the sampling noise
    the binomial already models and inflate the variance twice.

    WORKED EXAMPLE -- a WR1 with p = 0.25 over 8 games, E[N] = 34:

        V_sampling = 0.25 * 0.75 / 34            = 0.00551
        suppose V_obs (his actual game-to-game)  = 0.00900
        V_p        = 0.00900 - 0.00551           = 0.00349
        c          = 0.1875 / 0.00349 - 1        = 52.7

    So his share behaves like 53 pseudo-dropbacks of prior: meaningful
    week-to-week movement, but not a coin flip. Feed c = 52.7 into a
    beta-binomial at N = 34 and Var(T) rises from the binomial's 6.38 to
    about 9.9 -- which is the size of the gap `validate_width.py` measured.

    WHY POOLED BY ROLE
    ------------------
    Per player this estimate is hopeless: n is 8, V_obs has enormous
    sampling error of its own, and the one game he left with a hamstring
    dominates it. Per role it pools hundreds of players. The MEDIAN is
    taken rather than the mean because the distribution of per-player c is
    heavy-tailed on both ends -- a player whose V_obs lands below
    V_sampling produces a negative V_p and an infinite c, and one bad
    estimate would otherwise move the pooled number a long way.

    `floor` and `cap` exist for those degenerate ends. A c below 1 is a
    share that swings from 0 to 1 every week, which no real role does; a c
    above the cap is indistinguishable from the fixed share this replaces.
    """
    need = {"season", "week", "team", "player_id", value_col}
    missing = need - set(player_games.columns)
    if missing:
        raise ValueError(f"player_games missing columns: {sorted(missing)}")
    if not {"role_pos", "role_bucket"}.issubset(set(player_games.columns)):
        return {}, float("nan")

    tot = (player_games.group_by(["season", "week", "team"])
                       .agg(pl.col(value_col).sum().alias("_N")))
    df = (player_games.join(tot, on=["season", "week", "team"], how="left")
                      .filter(pl.col("_N") > 0)
                      .sort(["player_id", "season", "week"]))

    per = (df.group_by(["player_id", "role_pos", "role_bucket"], maintain_order=True)
             .agg([
                 pl.col(value_col).tail(trailing).sum().alias("_T"),
                 pl.col("_N").tail(trailing).sum().alias("_Nsum"),
                 pl.col("_N").tail(trailing).len().alias("_n"),
                 (pl.col(value_col) / pl.col("_N")).tail(trailing).var().alias("_vobs"),
                 (1.0 / pl.col("_N")).tail(trailing).mean().alias("_invN"),
             ])
             .filter((pl.col("_n") >= min_games) & (pl.col("_Nsum") > 0)
                     & pl.col("_vobs").is_not_null()))
    if per.is_empty():
        return {}, float("nan")

    p = (per["_T"] / per["_Nsum"]).to_numpy()
    pq = p * (1.0 - p)
    v_obs = per["_vobs"].to_numpy()
    v_samp = pq * per["_invN"].to_numpy()
    v_p = v_obs - v_samp

    # A player whose observed spread is at or below the sampling floor
    # carries no information about Var(p) -- his c is +infinity, not a large
    # number. Drop him rather than letting a cap-valued point vote.
    usable = (v_p > 0) & (pq > 0)
    c = np.full(len(p), np.nan)
    c[usable] = pq[usable] / v_p[usable] - 1.0
    c = np.clip(c, floor, cap)

    out = {}
    keys = list(zip(per["role_pos"].to_list(),
                    [int(b) for b in per["role_bucket"].to_list()]))
    buckets: dict = {}
    for key, ci in zip(keys, c):
        if np.isfinite(ci):
            buckets.setdefault(key, []).append(float(ci))
    for key, vals in buckets.items():
        if len(vals) >= min_players:
            out[key] = float(np.median(vals))

    allc = [v for vals in buckets.values() for v in vals]
    pooled = float(np.median(allc)) if allc else float("nan")
    return out, pooled


SHARE_GRID = np.round(np.arange(0.0, 1.0001, 0.02), 4)


def estimate_share_mixture(player_games: pl.DataFrame,
                           value_col: str = "opportunities",
                           trailing: int = 8,
                           min_rows: int = 100,
                           grid: np.ndarray = SHARE_GRID) -> dict:
    """
    The EMPIRICAL distribution of a role's share, on a fixed grid.

    Returns {(role_pos, role_bucket): (grid, weights, role_mean)}.

    WHY A HISTOGRAM AND NOT A WIDER BETA
    ------------------------------------
    The beta-binomial fixed most of the missing width but left QBs at a
    variance ratio of 0.649 against 0.868 for receivers. The reason is not
    that the beta was too narrow -- it is that a beta is UNIMODAL and a
    quarterback's share is not. Measured on 2022-2025, ACT weeks, share of
    team pass attempts:

        QB1  n=1994  mean 0.922   <0.10: 4.4%   0.10-0.80: 5.7%   >=0.80: 90.0%
        QB2  n=2073  mean 0.127   <0.10: 82.1%  0.10-0.80: 8.4%   >=0.80: 9.5%
        QB3  n= 367  mean 0.120   <0.10: 84.2%  0.10-0.80: 5.7%   >=0.80: 10.1%
        QB4  n= 137  mean 0.000   <0.10: 100%

    QB2's mean is 0.127 and he is at 0.127 essentially never: he throws no
    passes four weeks in five, and the whole game one week in ten. Any
    unimodal distribution centred on 0.127 describes a week that does not
    happen. It gets the mean right and every probability wrong, which is
    the exact failure this project has now hit three times.

    QB4 is the clean case: 137 player-games, share 0.000, standard
    deviation 0.000. He has never thrown a pass. The old model gave him
    8.59 predicted passing yards.

    A histogram needs no assumption about the number of modes, so it
    represents "0 or 1, rarely between" without being told that is the
    shape. The cost is resolution -- 51 grid points at 0.02 -- which is
    finer than the share is knowable.

    WHAT IS CARRIED AND WHAT IS NOT
    -------------------------------
    The role's SHAPE is pooled; the player's LEVEL is not. At serve time
    the grid is rescaled by (his shrunk share / the role mean), so a
    starter who takes a little less than a typical QB1 keeps that, and the
    bimodality comes from the role. Scaling rather than replacing matters
    for positions where the player's own history is the signal -- target
    share has split-half reliability 0.863, and throwing it away to use a
    role average would be a much larger loss than the width gain.
    """
    need = {"season", "week", "team", "player_id", value_col}
    missing = need - set(player_games.columns)
    if missing:
        raise ValueError(f"player_games missing columns: {sorted(missing)}")
    if not {"role_pos", "role_bucket"}.issubset(set(player_games.columns)):
        return {}

    tot = (player_games.group_by(["season", "week", "team"])
                       .agg(pl.col(value_col).sum().alias("_N")))
    df = (player_games.join(tot, on=["season", "week", "team"], how="left")
                      .filter(pl.col("_N") > 0)
                      .with_columns((pl.col(value_col) / pl.col("_N")).alias("_s")))

    out = {}
    for (rp, rb), grp in df.group_by(["role_pos", "role_bucket"]):
        if grp.height < min_rows:
            continue
        sv = np.clip(grp["_s"].to_numpy().astype(float), 0.0, 1.0)
        # Nearest grid point, so a share of exactly 1.0 lands on 1.0 rather
        # than being spread by a binning convention.
        idx = np.clip(np.round(sv / (grid[1] - grid[0])).astype(int),
                      0, len(grid) - 1)
        w = np.bincount(idx, minlength=len(grid)).astype(float)
        w /= w.sum()
        out[(rp, int(rb))] = (grid, w, float(sv.mean()))
    return out


def _fit_multipliers(df: pl.DataFrame, model, value_col: str,
                     trailing: int, positions: set) -> dict:
    """
    Each player's rate expressed as a MULTIPLE of the role he held at the
    time, rather than as an absolute rate.

        expected_i = sum over his trailing games of role_prior(role held THEN)
        actual_i   = sum of what he actually got in them
        mu_i       = (actual_i + k) / (expected_i + k)          -> shrunk to 1

    and at predict time his rate is `mu_i * role_prior(role he holds NOW)`.

    WHY. Every deep bucket over-predicts, in the same direction, across
    every position: WR7 2.1x, RB4 3.2x, TE4 1.8x, QB3 4.5x. The role
    PRIORS are not the problem -- measured against what those roles
    actually produce, they are exact to two decimals. The problem is the
    player's own history.

    Measured on 2025 week 10, current WR7s:

        mean depth rank over their trailing 8 games     3.56
        what they averaged in that window               2.631 targets
        what a WR7 actually averages                    1.131 targets
                                                        -> 2.33x

    A player at WR7 today was a WR3 or WR4 a month ago, and his own
    trailing average is a WR3.5 average. `prior_k` is 0.500 -- it sits on
    its floor, because the estimator correctly reports that players within
    a role differ a lot and their own history is informative -- so the
    role prior gets 6% of the weight at n=8 and his stale rate carries the
    rest. Raising k is the wrong fix: it would flatten the real
    player-to-player differences the estimator is detecting.

    This is the SAME error that put the WR7 prior at 1.18 targets a game,
    found and fixed a day earlier, reappearing one level down. There it
    was the prior that mixed roles; here it is the player's own rate. The
    general lesson is worth stating plainly: **a rate earned in one role
    does not transfer to another, and a depth chart is a thing players
    move around on.**

    What DOES transfer is how good he is relative to whatever role he held,
    which is what mu_i is. A receiver who earned 1.3x a typical WR3's
    targets is plausibly a 1.3x WR7 after a demotion; he is not still a
    WR3. The separation is the point: mu_i is who he is, the role prior is
    where he is, and only the second one changed.

    A player with no history gets expected_i = 0 and therefore mu_i = 1.0
    exactly -- his role prior, unchanged. That is the existing behaviour
    for rookies and it survives, for the same reason it did before: it
    falls out of `shrink`, not out of a special case.
    """
    have_roles = {"role_pos", "role_bucket"}.issubset(set(df.columns))
    if not have_roles or not positions:
        return {}
    df = df.filter(pl.col("role_pos").is_in(sorted(positions)))
    if df.is_empty():
        return {}

    # The prior for the role held in THAT game, per row.
    pri = []
    for rp, rb, pos in zip(df["role_pos"].to_list(),
                           df["role_bucket"].to_list(),
                           df["position"].to_list()):
        key = None if rp is None or rb is None else (rp, int(rb))
        pri.append(model._prior_for(key, pos))
    d = df.with_columns(pl.Series("_exp", pri, dtype=pl.Float64))

    agg = (d.group_by(["team", "player_id"], maintain_order=True)
             .agg([pl.col(value_col).tail(trailing).sum().alias("_a"),
                   pl.col("_exp").tail(trailing).sum().alias("_e"),
                   pl.col(value_col).tail(trailing).len().alias("_n"),
                   pl.col(value_col).tail(trailing).var().alias("_v")])
             .filter(pl.col("_n") > 0))
    if agg.is_empty():
        return {}

    # k is in units of EXPECTED OPPORTUNITIES, not games: "this many of his
    # role's opportunities are worth a prior of 1.0x". Estimated the same
    # method-of-moments way as everything else here.
    k = estimate_prior_strength_counts(
        agg["_a"].to_numpy(), np.maximum(agg["_e"].to_numpy(), 1e-9),
        np.nan_to_num(agg["_v"].to_numpy(), nan=0.0), min_trials=2)

    a, e = agg["_a"].to_numpy(), agg["_e"].to_numpy()
    mu = (a + k) / (e + k)
    return {(t, p): float(m) for t, p, m in
            zip(agg["team"].to_list(), agg["player_id"].to_list(), mu.tolist())}


def _fit_cross_team(df: pl.DataFrame, model, value_col: str,
                    trailing: int) -> tuple:
    """
    Each player's rate and the role he held earning it, IGNORING TEAM.

    WHY. `ShareModel.rates` is keyed on (team, player_id), which is right
    for a share of a specific team's targets and catastrophic the moment a
    player moves. His whole history files under the old team, the lookup
    for the new one misses, and he falls through to a generic role prior.

    Measured on the 2026 week 1 slate:

        Darren Waller, TE, Miami -> Carolina
          his shrunk rate under MIA        3.622 targets/game
          ('CAR', waller) in rates?        False
          so he gets the TE3 role prior    0.784      -- a 4.6x cut

        and he is not alone:
          own history found                323 players
          HISTORY UNDER AN OLD TEAM        108 players
          genuinely no history              91 players

    **Twenty-one percent of the slate lost its entire history to an
    offseason move.** The model was not saying those players got worse; it
    had no idea who they were. It priced Waller as a generic third-string
    tight end against a book that had him at 2.5 receptions, and the
    resulting 49-point "edge" was amnesia rather than an opinion.

    What transfers across a move is not his rate -- his new team throws a
    different number of passes and he sits at a different depth -- but how
    good he was RELATIVE to the role he held. That ratio times his new
    role's prior is the estimate. Same multiplier idea as
    `_fit_multipliers`, used in the one place it is unambiguously better
    than the alternative: against a player the model otherwise knows
    nothing about.
    """
    if not {"role_pos", "role_bucket"}.issubset(set(df.columns)):
        return {}
    d = df.sort(["player_id", "season", "week"])
    agg = (d.group_by("player_id", maintain_order=True)
             .agg([pl.col(value_col).tail(trailing).sum().alias("_s"),
                   pl.col(value_col).tail(trailing).len().alias("_n"),
                   pl.col("role_pos").tail(trailing).last().alias("_rp"),
                   pl.col("role_bucket").tail(trailing).median().alias("_rb"),
                   pl.col("position").tail(trailing).last().alias("_pos")])
             .filter(pl.col("_n") > 0))

    # TWO VERSIONS WERE MEASURED. This is the one that won, and it is not
    # the one that looks right on the motivating example.
    #
    #   version                 CRPS recv  CRPS rec  mean edge   Waller
    #   no cross-team            6.635      0.537     -0.1363    0.73 targets
    #   role-scaled (this)       6.629      0.536     -0.1305    0.67 targets
    #   full history, shrunk     6.638      0.537     -0.1451    2.82 targets
    #
    # Carrying his FULL old-team rate and shrinking it toward the new role
    # prior puts Waller at 2.82 targets and 53% against the book's 60% --
    # the answer the example demands. It also makes the backtest and the
    # market agreement WORSE, because shares are normalised: handing 108
    # moved players their old volume takes it from the players who stayed.
    #
    # So the role-scaled multiplier ships. It is a small, real improvement
    # on both measures, and it does NOT fix Waller: 0.853 x a 0.784 TE3
    # prior is still 0.67 targets. His gap is the depth chart calling him
    # a TE3, not the missing history -- which is what the evidence says
    # even though the missing history is real and was worth fixing.
    out, level = {}, {}
    for r in agg.iter_rows(named=True):
        rate = r["_s"] / r["_n"]
        role_then = (None if r["_rp"] is None or r["_rb"] is None
                     else (r["_rp"], int(round(r["_rb"]))))
        base = model._prior_for(role_then, r["_pos"])
        if base <= 0:
            continue
        # Clipped: a tiny old-role prior otherwise yields a multiplier in
        # the hundreds off two lucky games.
        out[r["player_id"]] = float(np.clip(rate / base, 0.05, 6.0))
        level[r["player_id"]] = float(rate)
    return out, level


@dataclass
class ShareModel:
    """
    Each player's shrunk share of his team's opportunities.

    THE PRIOR IS A DEPTH-CHART ROLE, NOT A POSITION
    -----------------------------------------------
    The first version of this class returned 0.0 for a player it had never
    seen, which is a point mass at zero. Validation caught it: the bottom
    calibration bucket predicted 0.001 against an actual 0.016. A rookie WR1
    in week 1 is not a rare case, and the model called him impossible.

    That is the baseball Tier-3 finding again -- `rbi_rate` and friends used
    raw means with `fillna(0.0)`, so a debut hitter got a home run rate
    below anyone alive. Everything shrinks, and everything shrinks toward
    something it might plausibly be.

    Measured on 493 first-ever games: position alone explains R^2 = 0.053 of
    target share; position x depth rank explains **0.311**.

    The mechanism needs no special case. `shrink` with zero history is

        (0 + k * prior) / (0 + k)  =  prior

    so a player with no games gets exactly his role prior, and one with two
    games gets mostly it. There is no `if unknown` branch anywhere below.

    Shares are renormalised to sum to 1 AFTER shrinkage, so the multinomial
    step downstream is well defined. That is not cosmetic: shrinking each
    player independently does not preserve the sum, and an un-normalised
    vector would quietly leak or invent opportunities.
    """
    rates: dict            # (team, player_id) -> shrunk opportunities per game
    counts: dict           # (team, player_id) -> (sum, n) of trailing games
    prior_k: float
    role_priors: dict      # (role_pos, role_bucket) -> opportunities per game
    position_priors: dict  # position -> opportunities per game  (fallback)
    global_rate: float
    # (role_pos, role_bucket) -> beta concentration of the share, and a
    # pooled fallback. None when dispersion was not estimated, in which
    # case the model behaves exactly as it did before.
    role_concentration: Optional[dict] = None
    global_concentration: Optional[float] = None
    # (role_pos, role_bucket) -> (grid, weights, role_mean). Present only
    # for positions listed in SHARE_MIXTURE_POSITIONS.
    role_mixture: Optional[dict] = None
    # (team, player_id) -> how many times his ROLE'S rate he earns. None
    # unless role_relative=True, in which case it replaces the raw rate.
    multipliers: Optional[dict] = None
    # player_id -> how many times his OLD role's rate he earned. Used only
    # when (team, player_id) is absent, i.e. he changed teams.
    elsewhere: Optional[dict] = None
    # player_id -> his OLD ABSOLUTE rate, in opportunities per game. The
    # level, not the ratio. Present alongside `elsewhere`.
    elsewhere_level: Optional[dict] = None
    # How much weight a mover's own LEVEL gets against the role-scaled
    # estimate. 0.0 reproduces the role-only behaviour exactly.
    cross_team_w: float = 0.0
    # Deepest role bucket the blend applies to. A WR7's chart position is
    # the model SAYING he is buried, and that is usually right; his own
    # history is stale evidence from a job he no longer holds. Blending
    # there resurrects deep reserves and steals share from the starters
    # books actually price.
    cross_team_max_bucket: int = 3

    def mixture_for(self, role: Optional[tuple]) -> Optional[tuple]:
        """The role's empirical share distribution, or None to fall back."""
        if self.role_mixture is None or role is None:
            return None
        return self.role_mixture.get((role[0], int(role[1])))

    def concentration_for(self, role: Optional[tuple]) -> Optional[float]:
        """
        How locked-in this role's share is, as a beta concentration. Lower
        means more volatile: Var(p) = p(1-p)/(c+1).

        Keyed on ROLE rather than on the player, deliberately. A player's
        own share volatility needs many games to estimate and the estimate
        is dominated by whether he happened to get hurt; the volatility of
        "being a WR2" is a property of the role and pools across the league.
        The same argument decided the share PRIOR, and it is the same
        argument: estimate the thing that has enough data.
        """
        if self.role_concentration is None:
            return self.global_concentration
        if role is not None:
            c = self.role_concentration.get((role[0], int(role[1])))
            if c is not None:
                return c
        return self.global_concentration

    @classmethod
    def fit(cls, player_games: pl.DataFrame, value_col: str = "opportunities",
            trailing: int = 8, roles: Optional[pl.DataFrame] = None,
            share_dispersion: bool = False,
            mixture_positions: Sequence[str] = (),
            role_relative: Sequence[str] = (),
            cross_team: bool = False,
            cross_team_w: float = 0.0,
            cross_team_max_bucket: int = 3,
            prior_k: Optional[float] = None) -> "ShareModel":
        """
        roles: optional frame with team, player_id, role_pos, role_bucket --
        one row per player, the most recent depth-chart standing the caller
        is allowed to see. When absent the prior falls back to position,
        which measured six times worse and exists only so the model still
        runs on seasons with no usable depth chart.

        prior_k: override the empirical-Bayes shrinkage strength, in GAMES.
        None keeps the estimated value. See SHARE_PRIOR_K in model_flags for
        why an override exists at all -- the EB estimate is the right answer
        only when the shrinkage TARGET is unbiased, and the depth-chart role
        prior is not.
        """
        need = {"season", "week", "team", "player_id", "position", value_col}
        missing = need - set(player_games.columns)
        if missing:
            raise ValueError(f"player_games missing columns: {sorted(missing)}")

        df = player_games.sort(["player_id", "season", "week"])
        agg = (
            df.group_by(["team", "player_id", "position"], maintain_order=True)
              .agg([
                  pl.col(value_col).tail(trailing).sum().alias("_sum"),
                  pl.col(value_col).tail(trailing).len().alias("_n"),
                  pl.col(value_col).tail(trailing).var().alias("_var"),
              ])
              .filter(pl.col("_n") > 0)
        )
        global_rate = float(agg["_sum"].sum() / agg["_n"].sum())

        position_priors = {
            r["position"]: (r["rate"] if r["n_players"] >= 3 else global_rate)
            for r in agg.group_by("position").agg([
                (pl.col("_sum").sum() / pl.col("_n").sum()).alias("rate"),
                pl.len().alias("n_players"),
            ]).iter_rows(named=True)
        }

        role_priors = {}
        # PREFER THE CONTEMPORANEOUS ROLE. If player_games carries the role
        # the player held in each game, estimate each bucket's prior from the
        # games actually played at that rank.
        #
        # The alternative -- joining a player's CURRENT rank to all of his
        # history -- mixes roles badly. It put the WR7 prior at 1.18 targets
        # a game when a real WR7 sees 0.004 of a team's targets, because
        # today's WR7 was often yesterday's WR2. Every no-history deep
        # reserve then got 1.18, ~16 of them per team, and the real
        # contributors were diluted by the difference: the model's mean
        # targets came to 0.55x actual, uniformly across the distribution.
        if {"role_pos", "role_bucket"}.issubset(set(player_games.columns)):
            cells = (df.group_by(["role_pos", "role_bucket"])
                       .agg([pl.col(value_col).mean().alias("rate"),
                             pl.len().alias("n_games")]))
            role_priors = {
                (r["role_pos"], int(r["role_bucket"])): r["rate"]
                for r in cells.iter_rows(named=True)
                if r["n_games"] >= 30
            }
        elif roles is not None and roles.height:
            need_r = {"team", "player_id", "role_pos", "role_bucket"}
            missing_r = need_r - set(roles.columns)
            if missing_r:
                raise ValueError(f"roles missing columns: {sorted(missing_r)}")
            joined = agg.join(roles, on=["team", "player_id"], how="inner")
            role_priors = {
                (r["role_pos"], int(r["role_bucket"])): r["rate"]
                for r in joined.group_by(["role_pos", "role_bucket"]).agg([
                    (pl.col("_sum").sum() / pl.col("_n").sum()).alias("rate"),
                    pl.len().alias("n_players"),
                ]).iter_rows(named=True)
                if r["n_players"] >= 5
            }

        k = estimate_prior_strength_counts(
            agg["_sum"].to_numpy(),
            agg["_n"].to_numpy(),
            np.nan_to_num(agg["_var"].to_numpy(), nan=0.0),
            min_trials=2,
        )
        if prior_k is not None:
            k = float(prior_k)

        model = cls(rates={}, counts={}, prior_k=float(k), role_priors=role_priors,
                    position_priors=position_priors, global_rate=global_rate)

        role_lookup = {}
        if roles is not None and roles.height:
            for r in roles.iter_rows(named=True):
                role_lookup[(r["team"], r["player_id"])] = (r["role_pos"], int(r["role_bucket"]))

        priors = np.array([
            model._prior_for(role_lookup.get((t, p)), pos)
            for t, p, pos in zip(agg["team"].to_list(), agg["player_id"].to_list(),
                                 agg["position"].to_list())
        ])
        rates = shrink(agg["_sum"].to_numpy(), agg["_n"].to_numpy(), priors, k)

        for t, p, s, n, rate in zip(agg["team"].to_list(), agg["player_id"].to_list(),
                                    agg["_sum"].to_list(), agg["_n"].to_list(), rates.tolist()):
            model.rates[(t, p)] = float(rate)
            model.counts[(t, p)] = (float(s), float(n))

        if role_relative:
            model.multipliers = _fit_multipliers(df, model, value_col, trailing,
                                                 set(role_relative))

        if cross_team:
            model.elsewhere, model.elsewhere_level = _fit_cross_team(
                df, model, value_col, trailing)
            model.cross_team_w = float(cross_team_w or 0.0)
            model.cross_team_max_bucket = int(cross_team_max_bucket)

        if share_dispersion:
            rc, pooled = estimate_share_concentration(
                player_games, value_col=value_col, trailing=trailing)
            model.role_concentration = rc or None
            model.global_concentration = pooled if np.isfinite(pooled) else None

        if mixture_positions:
            mix = estimate_share_mixture(player_games, value_col=value_col,
                                         trailing=trailing)
            keep = {k: v for k, v in mix.items() if k[0] in set(mixture_positions)}
            model.role_mixture = keep or None
        return model

    def _prior_for(self, role: Optional[tuple], position: Optional[str]) -> float:
        if role is not None and role in self.role_priors:
            return self.role_priors[role]
        if position is not None and position in self.position_priors:
            return self.position_priors[position]
        return self.global_rate

    def rate_for(self, team: str, player_id: str, role: Optional[tuple] = None,
                 position: Optional[str] = None) -> float:
        """
        Shrunk opportunities per game. A player with no history returns his
        role prior rather than zero -- which is just `shrink` with n = 0, not
        a special case.
        """
        key = (team, player_id)
        if self.multipliers and key in self.multipliers:
            # ROLE-RELATIVE. His multiplier times the prior for the role he
            # holds NOW -- so a demotion moves the base and keeps the
            # player. Only positions in ROLE_RELATIVE_POSITIONS have an
            # entry; everyone else falls through to the raw rate below.
            return float(self.multipliers[key] * self._prior_for(role, position))
        if key in self.rates:
            return self.rates[key]
        prior = self._prior_for(role, position)
        if self.elsewhere and player_id in self.elsewhere:
            # He has history, just not under this team -- he moved. Treat it
            # as what it is: n games of evidence about HIM, shrunk toward
            # what his new role usually gets. Same `shrink` every other
            # rate in this model uses, so a player who moved is handled
            # like a player who did not, rather than as a stranger.
            role_scaled = float(self.elsewhere[player_id] * prior)
            w = float(self.cross_team_w or 0.0)
            shallow = (role is not None
                       and int(role[1]) <= int(self.cross_team_max_bucket))
            if (w > 0.0 and shallow and self.elsewhere_level
                    and player_id in self.elsewhere_level):
                # HIS LEVEL IS NOT ZERO EVIDENCE. The role-scaled estimate
                # divides his measured rate out and re-inflates it against
                # the new role's prior, so 21 games of history reach the
                # prediction only as a ratio. At w = 0 that is the old
                # behaviour; above 0 his own level is carried too.
                return float(w * self.elsewhere_level[player_id]
                             + (1.0 - w) * role_scaled)
            return role_scaled
        return float(shrink([0.0], [0.0], prior, self.prior_k)[0])

    def share_for(self, team: str, player_id: str, role: Optional[tuple] = None,
                  position: Optional[str] = None) -> float:
        """Unnormalised rate expressed against the league's mean team volume."""
        return self.rate_for(team, player_id, role, position)

    def team_vector(self, team: str, player_ids: Sequence[str],
                    roles: Optional[Sequence[Optional[tuple]]] = None,
                    positions: Optional[Sequence[Optional[str]]] = None) -> np.ndarray:
        """
        Shares for a named set of players, renormalised over that set.

        roles: per-player (role_pos, role_bucket) from the depth chart, or
        None where unknown. positions: per-player position, the weaker
        fallback. Both optional; supplying neither reproduces the old
        behaviour for known players and gives unknown ones the global rate
        rather than zero.
        """
        n = len(player_ids)
        if roles is None:
            roles = [None] * n
        if positions is None:
            positions = [None] * n
        v = np.array([
            self.rate_for(team, p, r, pos)
            for p, r, pos in zip(player_ids, roles, positions)
        ], dtype=float)
        total = v.sum()
        if total <= 0:
            return np.full(n, 1.0 / max(n, 1))
        return v / total


# ---------------------------------------------------------------------
# Compound
# ---------------------------------------------------------------------

def betabinom_matrix(support: np.ndarray, out_support: np.ndarray,
                     share: float, concentration: float) -> np.ndarray:
    """
    Beta-binomial P(T = t | N) for every (N, t), with mean share and the
    given beta concentration.

        p ~ Beta(a, b),   a = c * share,   b = c * (1 - share)
        T | N, p ~ Binomial(N, p)

        P(T = t | N) = C(N, t) * B(t + a, N - t + b) / B(a, b)

    Computed in log space through `betaln` and `gammaln`, because B(a, b)
    underflows for the concentrations this model uses (c in the hundreds
    with share near 1 gives a beta function around 1e-300).

    As c -> infinity the beta collapses to a point mass at `share` and this
    returns the binomial exactly. That limit is asserted in the tests, and
    it is what makes the change safe: the old behaviour is a special case,
    not a branch.
    """
    from scipy.special import betaln, gammaln

    a = max(concentration * share, 1e-9)
    b = max(concentration * (1.0 - share), 1e-9)
    N = support[:, None].astype(float)
    t = out_support[None, :].astype(float)
    ok = t <= N

    log_choose = gammaln(N + 1) - gammaln(t + 1) - gammaln(np.where(ok, N - t, 0) + 1)
    log_p = log_choose + betaln(t + a, np.where(ok, N - t, 0) + b) - betaln(a, b)
    return np.where(ok, np.exp(log_p), 0.0)


def opportunity_pmf(volume_pmf: np.ndarray, support: np.ndarray,
                    share: float, max_opportunities: int = MAX_OPPORTUNITIES,
                    share_concentration: Optional[float] = None,
                    share_mixture: Optional[tuple] = None) -> np.ndarray:
    """
    Mix Binomial(N, share) over the team-volume pmf:

        P(T = t) = sum_N P(N) * Binom(t; N, share)

    Returns a pmf over 0..max_opportunities.

    This is the marginal of the multinomial, which is what a single-player
    prop needs. For anything involving two teammates at once -- a parlay, a
    correlated portfolio, or an honest error bar -- use joint_moments()
    below, because the marginals alone do not carry the dependence.

    THE SHARE IS NOT A CONSTANT, AND PRETENDING IT IS COST MOST OF THE
    MODEL'S WIDTH
    ------------------------------------------------------------------
    `share_concentration` swaps Binomial(N, p) for BetaBinomial(N, p, c),
    which is the same model with p allowed to vary game to game. Pass None
    for the old fixed-share behaviour.

    This was found by measurement, not by taste. Decomposing the predicted
    variance against reality on 2025 (`validate_width.py`) gave:

        stage 1a  team volume               var ratio 1.07 - 1.08   fine
        stage 2   per-opportunity outcome   var ratio 0.99 - 1.02   fine
        stage 1   player opportunities      var ratio 0.20 - 0.57   NOT

    Both ends of the pipeline are calibrated and the step between them is
    short by a factor of two to five. The reason is structural: under the
    multinomial, a player's only source of variance is sampling,

        Var(T) = p^2 Var(N) + p(1-p) E[N]

    and for a QB with p = 0.95 the second term is 0.0475 * E[N], almost
    nothing, so the model says a starter's attempts are nearly certain. In
    reality he throws 41 in a shootout and 19 in a downpour. Letting p vary
    adds the term the multinomial has no way to express:

        Var(T) = p^2 Var(N) + p(1-p) E[N] + (E[N^2] - E[N]) * Var(p)

    with Var(p) = p(1-p)/(c+1) for a Beta of concentration c. Small c means
    a volatile role; large c means a locked-in one. The measured ratios
    above say c must be small enough to roughly double the variance, and
    small c is exactly what the QB case demands.

    Note the sign of the third term's coefficient: E[N^2] - E[N] is large
    and positive, so a modest Var(p) buys a lot of width. This is the same
    law-of-total-variance structure as the shrinkage estimator elsewhere in
    this file, read in the other direction -- there it was used to find out
    how much of a player's spread was real, here to put that spread back.
    """
    from scipy.stats import binom

    if not 0.0 <= share <= 1.0:
        raise ValueError(f"share must be in [0, 1], got {share}")
    out_support = np.arange(0, max_opportunities + 1)
    if share == 0.0:
        pmf = np.zeros(len(out_support))
        pmf[0] = 1.0
        return pmf

    if share_mixture is not None:
        # EMPIRICAL SHARE MIXTURE. `share_mixture` is (grid, weights,
        # role_mean) from estimate_share_mixture. The grid is the role's
        # realised share distribution; it is rescaled so its mean is this
        # player's share, which keeps his level while borrowing the role's
        # SHAPE. Beta and beta-binomial are both unimodal and a
        # quarterback's share is not -- see estimate_share_mixture for the
        # measurement that forced this.
        grid, weights, role_mean = share_mixture
        pts = np.asarray(grid, dtype=float)
        w = np.asarray(weights, dtype=float).copy()
        alpha = share / role_mean if role_mean > 0 else 1.0

        # MOVE MASS, NEVER SCALE THE VALUES. Both branches below mix the
        # role's histogram with a point mass and both preserve the mean
        # EXACTLY -- asserted in the tests across the whole share range.
        #
        # Scaling the grid instead is what the first version did, and it
        # was wrong twice over. It turns "takes every snap in 80% of weeks"
        # into "takes 81% of the snaps every week", the smooth unimodal
        # thing this mixture exists to avoid (measured: QB1 variance ratio
        # 0.707 -> 0.579, worse than the beta-binomial it replaced). And
        # scaling UP a distribution already piled at 1.0 has nowhere to go,
        # so clipping silently ate the mean: at share 1.00, E[T] came back
        # 30.70 against a true 33.00, and passing yards picked up a -4.86
        # bias that had not been there before.
        #
        # The physical reading is the point. A quarterback's share is a
        # probability of starting, not a fraction of a start, so to make
        # him less of a starter you add weeks where he throws nothing, and
        # to make him more of one you add weeks where he takes every snap.
        if share <= role_mean:
            #   alpha * role + (1 - alpha) * delta_0
            #   mean = alpha * role_mean = share                    [exact]
            w *= alpha
            w[0] += 1.0 - alpha
        else:
            #   beta * role + (1 - beta) * delta_1,  beta = (1-share)/(1-role_mean)
            #   mean = beta*role_mean + (1-beta) = 1 - beta(1-role_mean) = share
            if role_mean >= 1.0 - 1e-9:
                w = np.zeros_like(w)
                w[-1] = 1.0
            else:
                beta = (1.0 - share) / (1.0 - role_mean)
                w *= beta
                w[-1] += 1.0 - beta

        keep = w > 1e-9
        weights = w
        pmf = np.zeros(len(out_support))
        for sj, wj in zip(pts[keep], weights[keep]):
            if sj <= 0.0:
                pmf[0] += wj          # a share of exactly zero is zero
                continue
            pmf += wj * (volume_pmf @ binom.pmf(
                out_support[None, :], support[:, None], sj))
    elif share_concentration is None or not np.isfinite(share_concentration):
        # (len(support), len(out_support)) matrix of Binom(t; N, share)
        mat = binom.pmf(out_support[None, :], support[:, None], share)
        pmf = volume_pmf @ mat
    else:
        mat = betabinom_matrix(support, out_support, share,
                               float(share_concentration))
        pmf = volume_pmf @ mat
    total = pmf.sum()
    return pmf / total if total > 0 else pmf


def joint_moments(volume_pmf: np.ndarray, support: np.ndarray,
                  p_i: float, p_j: float) -> dict:
    """
    Exact first and second moments for two teammates under the
    volume-then-multinomial model. No simulation needed.

        E[T_i]       = E[N] p_i
        Var(T_i)     = E[N] p_i(1-p_i) + Var(N) p_i^2
        Cov(T_i,T_j) = p_i p_j ( Var(N) - E[N] )

    The covariance is positive exactly when team volume is overdispersed,
    which it is (league var/mean 1.97). See the module docstring for what
    that does and does not mean for evaluation.
    """
    EN = float(volume_pmf @ support)
    EN2 = float(volume_pmf @ (support ** 2))
    VN = EN2 - EN ** 2
    return {
        "E_i": EN * p_i,
        "E_j": EN * p_j,
        "Var_i": EN * p_i * (1 - p_i) + VN * p_i ** 2,
        "Var_j": EN * p_j * (1 - p_j) + VN * p_j ** 2,
        "Cov_ij": p_i * p_j * (VN - EN),
        "E_N": EN,
        "Var_N": VN,
    }


def teammate_correlation(volume_pmf: np.ndarray, support: np.ndarray,
                         p_i: float, p_j: float) -> float:
    m = joint_moments(volume_pmf, support, p_i, p_j)
    denom = np.sqrt(m["Var_i"] * m["Var_j"])
    return float(m["Cov_ij"] / denom) if denom > 0 else 0.0
