"""
Stage 2: outcome per opportunity.

One structure covers every prop in the project:

    opportunity  ->  outcome        conditioned on an optional SHAPE parameter
    target       ->  yards gained   aDOT
    target       ->  0/1 catch      aDOT
    carry        ->  yards gained   (none yet)
    attempt      ->  yards gained   team aDOT

Stage 1 gives the number of opportunities as a pmf; this file gives the
outcome of one opportunity as a pmf; the compound is an exact discrete
convolution. Nothing is simulated.

WHY A SHAPE PARAMETER AT ALL
----------------------------
Yards per target has split-half reliability **0.323** -- a player's own past
efficiency barely predicts his future. But that is a fact about YPT, not
about Stage 2, because YPT is the wrong quantity to estimate:

    average depth of target (aDOT)   split-half r = 0.885
    targets                          split-half r = 0.863
    catch rate                       split-half r = 0.402
    yards per target                 split-half r = 0.323

aDOT is more stable than target volume -- the most reliable quantity in the
project -- and predicts future efficiency better than past efficiency does
(0.343 vs 0.255). It is *structure*, not a *level*: where a player is thrown
the ball, not how well it went. Baseball's `bases_per_hit` was the same kind
of quantity.

It also explains why YPT looks unstable: aDOT moves the hurdle and the tail
in OPPOSITE directions, so the product partly cancels.

    aDOT band   P(catch)   yards|catch   P(>=20|catch)    YPT
    < 5           0.772        7.77          0.067        6.00
    8-11          0.662       12.15          0.166        8.04
    14+           0.547       15.93          0.262        8.72

YPT spans 1.45x; the tail spans 3.9x.

AND WHY IT BUYS LESS THAN THAT SUGGESTS
---------------------------------------
Measured end to end, conditioning on aDOT is worth **under 1%** of CRPS on
game totals. The reason is in this file's own arithmetic: a game total is a
SUM, and sums forget shape. Max |CDF difference| between a deep and a
shallow per-target distribution shifted to a common mean:

    k=1  0.438     k=4  0.189     k=8  0.168     k=12  0.148

That is the central limit theorem. Only the mean survives the convolution,
and aDOT's effect on the mean is the 1.45x column. Keep the shape parameter
-- it is free, significant and correct -- but expect Stage 1 to carry the
model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import polars as pl

from features.shrinkage import estimate_prior_strength_counts, shrink

# Default per-opportunity support: receiving/rushing yards on one play.
# Observed range over 2022-2025 is -24 to 98 for targets.
PT_MIN, PT_MAX = -25, 105
PT_SUPPORT = np.arange(PT_MIN, PT_MAX + 1)

# Default total support. team-run-model.md's MAX_RUNS lesson: a cap tuned to
# the cases you tested is a cap that fails on the case you did not, so it is
# set well clear of the single-game record and `retained_mass()` measures
# what truncation actually costs rather than leaving it assumed.
TOTAL_MIN, TOTAL_MAX = -40, 500
TOTAL_SUPPORT = np.arange(TOTAL_MIN, TOTAL_MAX + 1)

# aDOT bands. Empirical pmfs per band, interpolated between centres -- there
# is no parametric form here to be wrong about. Finer low-end bands were
# tested (select on 2024, verify on 2025) and moved CRPS by 0.0004: noise.
ADOT_EDGES = np.array([-np.inf, 5.0, 8.0, 11.0, 14.0, np.inf])
ADOT_CENTRES = np.array([3.0, 6.5, 9.5, 12.5, 16.0])

MIN_ROWS_PER_BAND = 300


def band_of(value: float, edges: np.ndarray, n_bands: int) -> int:
    return int(np.clip(np.searchsorted(edges[1:-1], value, side="right"), 0, n_bands - 1))


def adot_band(adot: float) -> int:
    """Band index under the default aDOT banding."""
    return band_of(adot, ADOT_EDGES, len(ADOT_CENTRES))


@dataclass
class PerOpportunityModel:
    """
    Distribution of the outcome of ONE opportunity, optionally conditioned on
    a shape parameter.

    Failed opportunities are carried as zeros, so this is the unconditional
    per-opportunity distribution and any hurdle is already inside it. For
    receiving that means incompletes are zeros -- a hurdle, not zero
    inflation, because the zero comes from the catch failing rather than
    from the yardage distribution having extra mass at zero. Measured
    P(complete | target) = 0.6728 over 68,751 targets.

    Band pmfs are pooled across the league. Per-player pmfs are not
    estimated and should not be: with a median 14 career games and a YPT
    reliability of 0.32, a per-player outcome shape would be almost entirely
    noise. What the player contributes is his shape parameter, and that is
    shrunk too.
    """
    band_pmfs: np.ndarray
    band_counts: np.ndarray
    support: np.ndarray
    centres: np.ndarray
    edges: np.ndarray
    league_shape: float

    @classmethod
    def fit(cls, rows: pl.DataFrame, outcome_col: str = "outcome",
            shape_col: Optional[str] = None,
            support: np.ndarray = PT_SUPPORT,
            edges: np.ndarray = ADOT_EDGES,
            centres: np.ndarray = ADOT_CENTRES,
            min_rows_per_band: int = MIN_ROWS_PER_BAND) -> "PerOpportunityModel":
        """
        rows: one row per opportunity, with the outcome and (optionally) the
        player's shape parameter *as known before that game*.

        shape_col=None fits a single pooled distribution -- the right thing
        when no stable structural parameter has been found for that prop,
        and an honest default rather than inventing one.

        The caller owns the as-of discipline. Labelling rows with a
        season-long shape parameter that includes the game being predicted
        is the train/serve skew that broke `_project_pa` in baseball.
        """
        if outcome_col not in rows.columns:
            raise ValueError(f"rows missing outcome column {outcome_col!r}")
        if shape_col is not None and shape_col not in rows.columns:
            raise ValueError(f"rows missing shape column {shape_col!r}")

        lo, hi = int(support[0]), int(support[-1])
        if shape_col is None:
            d = rows.drop_nulls([outcome_col])
            shape_vals = np.zeros(d.height)
            centres = np.array([0.0])
            edges = np.array([-np.inf, np.inf])
        else:
            d = rows.drop_nulls([outcome_col, shape_col])
            shape_vals = d[shape_col].to_numpy().astype(float)

        out = np.clip(np.round(d[outcome_col].to_numpy().astype(float)), lo, hi)
        n_bands = len(centres)
        bands = np.array([band_of(v, edges, n_bands) for v in shape_vals]) if n_bands > 1 \
            else np.zeros(len(out), dtype=int)

        pmfs = np.zeros((n_bands, len(support)))
        counts = np.zeros(n_bands)
        idx = (out - lo).astype(int)
        for b in range(n_bands):
            m = bands == b
            counts[b] = m.sum()
            if m.sum():
                np.add.at(pmfs[b], idx[m], 1.0)

        pooled = pmfs.sum(axis=0)
        pooled = pooled / pooled.sum() if pooled.sum() > 0 else pooled
        for b in range(n_bands):
            # A thin band borrows the pooled distribution rather than fitting
            # its own -- the same refusal as ShareModel's 5-player floor.
            if counts[b] < min_rows_per_band or pmfs[b].sum() == 0:
                pmfs[b] = pooled
            else:
                pmfs[b] = pmfs[b] / pmfs[b].sum()

        return cls(band_pmfs=pmfs, band_counts=counts, support=support,
                   centres=centres, edges=edges,
                   league_shape=float(np.mean(shape_vals)) if len(shape_vals) else 0.0)

    def pmf_for(self, shape: Optional[float] = None) -> np.ndarray:
        """
        Per-opportunity pmf at an arbitrary shape value, linearly
        interpolated between neighbouring band centres.

        Interpolating rather than snapping avoids a discontinuity at the
        edges, where a receiver at 10.9 and one at 11.1 would otherwise get
        materially different tails.
        """
        if len(self.centres) == 1 or shape is None:
            return self.band_pmfs[0]
        c = self.centres
        a = float(np.clip(shape, c[0], c[-1]))
        j = int(np.clip(np.searchsorted(c, a), 1, len(c) - 1))
        lo, hi = c[j - 1], c[j]
        w = 0.0 if hi == lo else (a - lo) / (hi - lo)
        pmf = (1 - w) * self.band_pmfs[j - 1] + w * self.band_pmfs[j]
        total = pmf.sum()
        return pmf / total if total > 0 else pmf

    def summary(self) -> pl.DataFrame:
        rows = []
        for b, centre in enumerate(self.centres):
            pmf = self.band_pmfs[b]
            rows.append({
                "band": b, "centre": float(centre), "n": int(self.band_counts[b]),
                "p_zero": float(pmf[self.support == 0].sum()),
                "mean": float(pmf @ self.support),
                "p_20plus": float(pmf[self.support >= 20].sum()),
            })
        return pl.DataFrame(rows)


# Backwards-compatible alias: the receiving-specific name this started as.
PerTargetModel = PerOpportunityModel


# ---------------------------------------------------------------------
# Shrunk per-player shape parameter
# ---------------------------------------------------------------------

@dataclass
class ShapeModel:
    """
    Each player's shrunk shape parameter (aDOT, for receiving).

    Shrinkage is light on purpose: at split-half 0.885 a player's own aDOT
    is close to the most reliable quantity in the model. But it is still
    shrunk, and toward a ROLE baseline, because a rookie deep threat with
    four targets should not be handed a 17-yard aDOT on that evidence. Same
    rule as everywhere: everything shrinks, nothing is `fillna(0.0)`.
    """
    player_value: dict
    counts: dict
    league_value: float
    role_value: dict
    prior_k: float

    @classmethod
    def fit(cls, player_games: pl.DataFrame, roles: Optional[pl.DataFrame] = None,
            sum_col: str = "shape_sum", n_col: str = "shape_n",
            trailing: int = 8) -> "ShapeModel":
        need = {"season", "week", "team", "player_id", sum_col, n_col}
        missing = need - set(player_games.columns)
        if missing:
            raise ValueError(f"player_games missing columns: {sorted(missing)}")

        df = player_games.sort(["player_id", "season", "week"])
        agg = (
            df.group_by(["team", "player_id"], maintain_order=True)
              .agg([
                  pl.col(sum_col).tail(trailing).sum().alias("_sum"),
                  pl.col(n_col).tail(trailing).sum().alias("_n"),
                  pl.col(sum_col).tail(trailing).var().alias("_var"),
              ])
              .filter(pl.col("_n") > 0)
        )
        league = float(agg["_sum"].sum() / agg["_n"].sum())

        role_value = {}
        if roles is not None and roles.height:
            joined = agg.join(roles, on=["team", "player_id"], how="inner")
            role_value = {
                (r["role_pos"], int(r["role_bucket"])): r["v"]
                for r in joined.group_by(["role_pos", "role_bucket"]).agg([
                    (pl.col("_sum").sum() / pl.col("_n").sum()).alias("v"),
                    pl.len().alias("n_players"),
                ]).iter_rows(named=True)
                if r["n_players"] >= 5
            }

        k = estimate_prior_strength_counts(
            agg["_sum"].to_numpy(), agg["_n"].to_numpy(),
            np.nan_to_num(agg["_var"].to_numpy(), nan=0.0), min_trials=5,
        )
        lookup = {}
        if roles is not None and roles.height:
            for r in roles.iter_rows(named=True):
                lookup[(r["team"], r["player_id"])] = (r["role_pos"], int(r["role_bucket"]))

        model = cls(player_value={}, counts={}, league_value=league,
                    role_value=role_value, prior_k=float(k))
        priors = np.array([model._prior_for(lookup.get((t, p)))
                           for t, p in zip(agg["team"].to_list(), agg["player_id"].to_list())])
        vals = shrink(agg["_sum"].to_numpy(), agg["_n"].to_numpy(), priors, k)
        for t, p, s, n, v in zip(agg["team"].to_list(), agg["player_id"].to_list(),
                                 agg["_sum"].to_list(), agg["_n"].to_list(), vals.tolist()):
            model.player_value[(t, p)] = float(v)
            model.counts[(t, p)] = (float(s), float(n))
        return model

    def _prior_for(self, role: Optional[tuple]) -> float:
        if role is not None and role in self.role_value:
            return self.role_value[role]
        return self.league_value

    def value_for(self, team: str, player_id: str, role: Optional[tuple] = None) -> float:
        key = (team, player_id)
        if key in self.player_value:
            return self.player_value[key]
        return float(shrink([0.0], [0.0], self._prior_for(role), self.prior_k)[0])


AdotModel = ShapeModel  # the receiving-specific name this started as


# ---------------------------------------------------------------------
# Compound: Stage 1 x Stage 2
# ---------------------------------------------------------------------

def _convolve_k(per_opp: np.ndarray, k: int, cache: dict,
                pt_support: np.ndarray, total_support: np.ndarray) -> np.ndarray:
    if k in cache:
        return cache[k]
    t_min, t_max = int(total_support[0]), int(total_support[-1])
    if k == 0:
        out = np.zeros(len(total_support))
        out[total_support == 0] = 1.0
    else:
        prev = _convolve_k(per_opp, k - 1, cache, pt_support, total_support)
        full = np.convolve(prev, per_opp)
        start = t_min + int(pt_support[0])
        idx = np.arange(len(full)) + start
        out = np.zeros(len(total_support))
        keep = (idx >= t_min) & (idx <= t_max)
        np.add.at(out, (idx[keep] - t_min).astype(int), full[keep])
    cache[k] = out
    return out


def total_pmf(opp_pmf: np.ndarray, opp_support: np.ndarray, per_opp: np.ndarray,
              max_opportunities: int = 25,
              pt_support: np.ndarray = PT_SUPPORT,
              total_support: np.ndarray = TOTAL_SUPPORT) -> np.ndarray:
    """
    Total pmf = sum over k of P(k opportunities) x (per-opportunity conv k).

    Exact, not simulated. Verified against closed forms: the mean is exactly
    k x the per-opportunity mean and the variance exactly k x its variance.
    """
    cache: dict = {}
    out = np.zeros(len(total_support))
    for k, p in zip(opp_support, opp_pmf):
        if p <= 0:
            continue
        kk = int(min(k, max_opportunities))
        out += p * _convolve_k(per_opp, kk, cache, pt_support, total_support)
    s = out.sum()
    return out / s if s > 0 else out


def retained_mass(opp_pmf: np.ndarray, opp_support: np.ndarray, per_opp: np.ndarray,
                  max_opportunities: int = 25,
                  pt_support: np.ndarray = PT_SUPPORT,
                  total_support: np.ndarray = TOTAL_SUPPORT) -> float:
    """
    Mass surviving truncation, before renormalising. Call it on the most
    extreme player you have before trusting the cap -- truncating and
    renormalising pulls the mean DOWN, invisibly at small means and
    materially at large ones.
    """
    cache: dict = {}
    tot = 0.0
    for k, p in zip(opp_support, opp_pmf):
        if p <= 0:
            continue
        tot += p * _convolve_k(per_opp, int(min(k, max_opportunities)), cache,
                               pt_support, total_support).sum()
    return float(tot)


def prob_over(pmf: np.ndarray, line: float,
              total_support: np.ndarray = TOTAL_SUPPORT) -> float:
    """P(total > line). Strict, so a whole-number line reads correctly."""
    return float(pmf[total_support > line].sum())


# The receiving-specific names this started as.
yards_pmf = total_pmf
