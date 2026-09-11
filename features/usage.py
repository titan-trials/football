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
            trailing: int = 8, max_support: int = MAX_OPPORTUNITIES) -> "TeamVolumeModel":
        """
        team_games: one row per team-game with columns season, week, team,
        and `value_col`. Rows must already be restricted to games the caller
        is allowed to see -- this class does no as-of filtering of its own.
        """
        need = {"season", "week", "team", value_col}
        missing = need - set(team_games.columns)
        if missing:
            raise ValueError(f"team_games missing columns: {sorted(missing)}")

        vals = team_games[value_col].to_numpy().astype(float)
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

@dataclass
class ShareModel:
    """
    Each player's shrunk share of his team's opportunities.

    Shares within a team are renormalised to sum to 1 AFTER shrinkage, so
    the multinomial step downstream is well defined. That renormalisation
    is not cosmetic: shrinking each player independently toward a role
    baseline does not preserve the sum, and an un-normalised vector would
    quietly leak or invent opportunities.
    """
    shares: dict           # (team, player_id) -> share
    prior_k: float
    role_priors: dict      # position -> pooled share

    @classmethod
    def fit(cls, player_games: pl.DataFrame, value_col: str = "opportunities",
            trailing: int = 8) -> "ShareModel":
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

        # Role baseline: pooled per-game opportunities by position. A WR1
        # and a blocking TE are not one population, and with a median 14
        # games of history the prior does most of the work.
        role = (
            agg.group_by("position")
               .agg([(pl.col("_sum").sum() / pl.col("_n").sum()).alias("role_rate"),
                     pl.len().alias("n_players")])
        )
        global_rate = float(agg["_sum"].sum() / agg["_n"].sum())
        role_priors = {
            r["position"]: (r["role_rate"] if r["n_players"] >= 3 else global_rate)
            for r in role.iter_rows(named=True)
        }

        k = estimate_prior_strength_counts(
            agg["_sum"].to_numpy(),
            agg["_n"].to_numpy(),
            np.nan_to_num(agg["_var"].to_numpy(), nan=0.0),
            min_trials=2,
        )
        priors = np.array([role_priors.get(p, global_rate) for p in agg["position"].to_list()])
        rates = shrink(agg["_sum"].to_numpy(), agg["_n"].to_numpy(), priors, k)

        out = agg.with_columns(pl.Series("_rate", rates))
        shares = {}
        for team, grp in out.group_by("team"):
            team_name = team[0] if isinstance(team, tuple) else team
            r = grp["_rate"].to_numpy()
            total = r.sum()
            if total <= 0:
                continue
            for pid, s in zip(grp["player_id"].to_list(), (r / total).tolist()):
                shares[(team_name, pid)] = float(s)

        return cls(shares=shares, prior_k=float(k), role_priors=role_priors)

    def share_for(self, team: str, player_id: str) -> float:
        return float(self.shares.get((team, player_id), 0.0))

    def team_vector(self, team: str, player_ids: Sequence[str]) -> np.ndarray:
        """Shares for a named set of players, renormalised over that set."""
        v = np.array([self.share_for(team, p) for p in player_ids], dtype=float)
        total = v.sum()
        if total <= 0:
            return np.full(len(player_ids), 1.0 / max(len(player_ids), 1))
        return v / total


# ---------------------------------------------------------------------
# Compound
# ---------------------------------------------------------------------

def opportunity_pmf(volume_pmf: np.ndarray, support: np.ndarray,
                    share: float, max_opportunities: int = MAX_OPPORTUNITIES) -> np.ndarray:
    """
    Mix Binomial(N, share) over the team-volume pmf:

        P(T = t) = sum_N P(N) * Binom(t; N, share)

    Returns a pmf over 0..max_opportunities.

    This is the marginal of the multinomial, which is what a single-player
    prop needs. For anything involving two teammates at once -- a parlay, a
    correlated portfolio, or an honest error bar -- use joint_moments()
    below, because the marginals alone do not carry the dependence.
    """
    from scipy.stats import binom

    if not 0.0 <= share <= 1.0:
        raise ValueError(f"share must be in [0, 1], got {share}")
    out_support = np.arange(0, max_opportunities + 1)
    if share == 0.0:
        pmf = np.zeros(len(out_support))
        pmf[0] = 1.0
        return pmf

    # (len(support), len(out_support)) matrix of Binom(t; N, share)
    mat = binom.pmf(out_support[None, :], support[:, None], share)
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
