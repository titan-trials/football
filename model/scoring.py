"""
Scoring. Two instruments, because football props are continuous and
baseball's were not.

WHY BRIER ALONE IS NOT ENOUGH HERE
----------------------------------
Baseball props were binary or small-count -- "does he homer", "over 5.5 K"
-- and the line barely moved. Brier skill against a base-rate null was the
right and only instrument.

A football prop is a yardage line that is different for every player every
week: receiving yards over 45.5, passing yards over 248.5. Brier can still
score it, but only AFTER collapsing the prediction to a single number at
one posted line, which throws away the rest of the distribution and makes
the score depend on where the book happened to hang the number.

So:
  crps()        -- scores the whole predictive distribution. Proper, needs
                   no line, and is the model-development metric.
  brier_skill() -- collapses to the binary at the posted line, for market
                   comparison only. Same instrument compare_market.py uses.

THE POOLING BUG, WHICH COST A SESSION IN BASEBALL
-------------------------------------------------
From clean-vs-tainted-rows.md: the dashboard reported HR edge +1.79% while
score_slate printed +1.65%, for the same rows. Skill is

    1 - brier / (p_base * (1 - p_base))

and the average of a ratio is not the ratio of the averages. score_slate
averaged per-slate skills; each slate divided by its own base rate, so the
average silently weighted slates by how extreme their base rate happened to
be. The dashboard pooled and was right.

Football will hit this per-week instead of per-slate, and harder, because a
week's base rate for "over the posted line" is pinned near 0.5 by the book
while a base-rate null over a season is not. brier_skill() below pools. It
is the only implementation; there is deliberately no per-week variant to
average by accident.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


# --- Continuous: CRPS --------------------------------------------------

def crps_ensemble(y_true, samples) -> np.ndarray:
    """
    CRPS for a predictive distribution given as an ensemble of samples.

        CRPS = E|X - y| - 0.5 * E|X - X'|

    where X, X' are independent draws from the forecast. Lower is better,
    and it reduces to absolute error when the forecast is a point mass --
    so a deterministic model can be compared to a distributional one on the
    same scale.

    y_true: (n,) observed values
    samples: (n, m) m samples from each row's predictive distribution
    Returns (n,) per-row CRPS.
    """
    y_true = np.asarray(y_true, dtype=float)
    samples = np.asarray(samples, dtype=float)
    if samples.ndim == 1:
        samples = samples[:, None]

    n, m = samples.shape
    if len(y_true) != n:
        raise ValueError(f"y_true has {len(y_true)} rows, samples has {n}")

    s = np.sort(samples, axis=1)
    term1 = np.abs(s - y_true[:, None]).mean(axis=1)

    # E|X - X'| via the sorted-sample identity, which is O(m log m) rather
    # than the O(m^2) double loop:
    #   E|X - X'| = (2 / m^2) * sum_i (2i - m + 1) * s_i     (i zero-based)
    idx = np.arange(m)
    weights = (2.0 * idx - m + 1.0)
    term2 = (2.0 / (m * m)) * (s * weights).sum(axis=1)

    return term1 - 0.5 * term2


def crps_pmf(y_true, support, probs) -> np.ndarray:
    """
    CRPS for a discrete distribution given as a pmf over a shared support
    -- the natural form for counting props (receptions, attempts).

        CRPS = sum_k (F(k) - 1{y <= k})^2 * dk

    support: (m,) ascending values
    probs: (n, m) each row a pmf over `support`
    """
    y_true = np.asarray(y_true, dtype=float)
    support = np.asarray(support, dtype=float)
    probs = np.asarray(probs, dtype=float)

    cdf = np.cumsum(probs, axis=1)
    heaviside = (y_true[:, None] <= support[None, :]).astype(float)
    dk = np.diff(support, prepend=support[0] - (support[1] - support[0]))
    return ((cdf - heaviside) ** 2 * dk).sum(axis=1)


def crps_skill(y_true, samples, baseline_samples) -> float:
    """
    1 - mean(CRPS_model) / mean(CRPS_baseline). Pooled, not averaged.

    The honest baseline is the climatological forecast: the player's own
    shrunk historical distribution, ignoring this week's matchup. Beating
    an unconditional forecast is the football equivalent of beating the
    base rate, and is just as low a bar -- the market is the real one.
    """
    m = float(np.mean(crps_ensemble(y_true, samples)))
    b = float(np.mean(crps_ensemble(y_true, baseline_samples)))
    if b <= 0:
        return float("nan")
    return 1.0 - m / b


# --- Binary at a posted line -------------------------------------------

def brier(y_true, probs) -> float:
    y_true = np.asarray(y_true, dtype=float)
    probs = np.asarray(probs, dtype=float)
    return float(np.mean((probs - y_true) ** 2))


def brier_skill(y_true, probs, base_rate: Optional[float] = None) -> float:
    """
    Pooled Brier skill against a constant base-rate forecast.

        skill = 1 - brier / (p_base * (1 - p_base))

    POOLED. Compute this once over every row you want to summarise. Never
    compute it per week and average the results -- see the module docstring.
    If base_rate is None it is the observed rate over the rows passed in,
    which is the honest null.
    """
    y_true = np.asarray(y_true, dtype=float)
    p = float(np.mean(y_true)) if base_rate is None else float(base_rate)
    denom = p * (1.0 - p)
    if denom <= 0:
        return float("nan")
    return 1.0 - brier(y_true, probs) / denom


def prob_over(support, probs, line: float) -> np.ndarray:
    """
    P(X > line) from a pmf. Football lines are hung at .5 so pushes are
    rare, but a whole-number line (over 1.5 TDs vs over 2) does happen and
    the strict inequality is the correct reading of "over".
    """
    support = np.asarray(support, dtype=float)
    probs = np.asarray(probs, dtype=float)
    mask = (support > line).astype(float)
    return probs @ mask


def paired_bootstrap(y_true, probs_a, probs_b, n_boot: int = 2000,
                     seed: int = 0) -> tuple:
    """
    Bootstrap CI on the Brier DIFFERENCE between two forecasts on the same
    rows. Paired, because the rows are shared and an unpaired interval
    would be far too wide to see a real 0.01 of skill.

    Returns (mean_diff, lo, hi) at 95%, where positive means A is better.

    Football caveat that baseball did not have: every receiver on one team
    draws from the same ~35 pass attempts, so rows within a game are
    strongly dependent. Resampling rows understates the interval. Resample
    GAMES and pass rows grouped, or read the interval as optimistic.
    """
    y_true = np.asarray(y_true, dtype=float)
    a = np.asarray(probs_a, dtype=float)
    b = np.asarray(probs_b, dtype=float)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = (np.mean((b[idx] - y_true[idx]) ** 2)
                    - np.mean((a[idx] - y_true[idx]) ** 2))
    point = float(np.mean((b - y_true) ** 2) - np.mean((a - y_true) ** 2))
    return point, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
