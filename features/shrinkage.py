"""
Empirical-Bayes shrinkage. Ported unchanged in method from
baseball_predictor/features/rate_features.py -- same beta-binomial method of
moments, same clip band, same "large k is the correct answer, not a
failure" behaviour.

Two things are different, and both come from football having 17 games
instead of 162.

  1. `estimate_prior_strength_counts` matters far more here than it did in
     baseball, where it was a side path for RBI and total bases. In
     football the headline Stage 1 quantity -- targets, carries, attempts --
     IS a count per game, so the count version is the main road.

  2. Shrinkage is toward a ROLE baseline, not a league mean. A league-mean
     prior over all pass-catchers pulls a WR1 and a blocking TE toward the
     same number, and with a median of 14 career games per targeted player
     the prior does most of the work. `group_prior_mean` builds the
     baseline per (position, depth) cell so the pull is toward something
     the player might plausibly be.

The baseball Tier-3 finding this is guarding against: rbi_rate, obp_series
and the two-stage RBI inputs used raw unshrunk means with fillna(0.0), so a
debut hitter got a HR rate below anyone alive. Football's version is a
week-1 rookie WR with three career targets. Everything shrinks.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import polars as pl


def estimate_prior_strength(successes, trials, min_trials: int = 20) -> float:
    """
    Beta-binomial method of moments for a 0/1 rate (catch rate, TD per
    target). Returns k for use in (x + k*prior) / (n + k).

        observed variance between players = true variance + binomial noise
        binomial noise is computable exactly, as mean(p(1-p)/n)
        whatever is left is the true spread, and k = p(1-p)/true_var - 1

    When the leftover is zero or negative -- players differ no more than
    chance would produce -- k is very large, shrinking everyone to the
    population mean. That is the right answer: it says there is no
    demonstrated skill difference here to preserve.

    min_trials defaults to 20 rather than baseball's 50: a full NFL season
    of targets for a WR2 is roughly 70, so a 50-trial floor would discard
    most of the league.
    """
    successes = np.asarray(successes, dtype=float)
    trials = np.asarray(trials, dtype=float)

    keep = np.isfinite(successes) & np.isfinite(trials) & (trials >= min_trials)
    successes, trials = successes[keep], trials[keep]
    if len(trials) < 3:
        return 500.0  # too few players to estimate anything; shrink hard

    prior_mean = successes.sum() / trials.sum()
    rates = successes / trials

    observed_var = rates.var(ddof=1)
    binomial_var = float(np.mean(prior_mean * (1.0 - prior_mean) / trials))
    true_var = observed_var - binomial_var

    if true_var <= 1e-9:
        return 5000.0

    k = prior_mean * (1.0 - prior_mean) / true_var - 1.0
    return float(np.clip(k, 10.0, 5000.0))


def estimate_prior_strength_counts(sums, trials, within_var,
                                   min_trials: int = 4) -> float:
    """
    The same idea for a per-game rate of a COUNT (targets per game, carries
    per game) rather than a 0/1 flag.

        observed variance of per-player means
          = true variance + mean(within-player variance / n)

    `trials` is games played, `sums` is total opportunities, `within_var` is
    each player's own game-to-game variance. Returns k in GAMES.

    min_trials is 4 games. That is low, and deliberately so -- raising it
    to a baseball-like floor would drop every rookie and every player who
    missed time, which in a 17-game sport is most of the interesting ones.
    """
    sums = np.asarray(sums, dtype=float)
    trials = np.asarray(trials, dtype=float)
    within_var = np.asarray(within_var, dtype=float)

    keep = (trials >= min_trials) & np.isfinite(within_var) & np.isfinite(sums)
    sums, trials, within_var = sums[keep], trials[keep], within_var[keep]
    if len(trials) < 3:
        return 500.0

    rates = sums / trials
    observed_var = rates.var(ddof=1)
    noise_var = float(np.mean(within_var / trials))
    true_var = observed_var - noise_var
    if true_var <= 1e-12:
        return 5000.0

    k = float(np.mean(within_var)) / true_var - 1.0
    return float(np.clip(k, 0.5, 200.0))


def shrink(successes, trials, prior_mean, k: float):
    """The empirical-Bayes estimate. Vectorised; prior_mean may be an array."""
    successes = np.asarray(successes, dtype=float)
    trials = np.asarray(trials, dtype=float)
    prior_mean = np.asarray(prior_mean, dtype=float)
    return (successes + k * prior_mean) / (trials + k)


def group_prior_mean(df: pl.DataFrame, value_col: str, trial_col: str,
                     by: Sequence[str], fallback: Optional[float] = None) -> pl.DataFrame:
    """
    Pooled rate within each `by` cell, for use as the shrinkage target.

    Returns the input frame with a `<value_col>_prior` column joined on.
    Cells thinner than 3 players fall back to the global pooled rate, so a
    (position, depth) cell that barely exists cannot invent its own prior.

    This is the part that is NOT a port. Baseball shrank toward one league
    mean because a hitter is a hitter. A WR1 and a blocking TE are not the
    same population, and with 17 games the prior is most of the estimate.
    """
    global_rate = (df[value_col].sum() / df[trial_col].sum()) if fallback is None else fallback

    cells = (
        df.group_by(list(by))
          .agg([
              pl.col(value_col).sum().alias("_v"),
              pl.col(trial_col).sum().alias("_t"),
              pl.len().alias("_n"),
          ])
          .with_columns(
              pl.when(pl.col("_n") >= 3)
                .then(pl.col("_v") / pl.col("_t"))
                .otherwise(pl.lit(global_rate))
                .alias(f"{value_col}_prior")
          )
          .drop(["_v", "_t", "_n"])
    )
    return df.join(cells, on=list(by), how="left").with_columns(
        pl.col(f"{value_col}_prior").fill_null(global_rate)
    )
