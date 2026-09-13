"""
Is the predicted DISTRIBUTION the right width, and if not, which stage is
too narrow?

    python validate_width.py --score-season 2025

WHY WIDTH IS THE WHOLE GAME FOR A PROP BET
------------------------------------------
A prop settles on P(X > L), not on E[X]. Those are different functionals of
the same distribution, and a model can be perfect on one while being wrong
on every value of the other.

    X ~ Normal(mu, sigma^2),  P(X > L) = 1 - Phi((L - mu) / sigma)

Hold mu right and shrink sigma and every probability away from the median
moves toward 0 or 1. Worked example with the real numbers for a QB1:

    actual   mu = 200, sigma = 70   ->  P(X > 264.5) = 1 - Phi(0.921) = 0.179
    model    mu = 200, sigma = 55   ->  P(X > 264.5) = 1 - Phi(1.173) = 0.120

Same mean, 21% narrower, and the over is priced at 0.120 when it should be
0.179. In odds that is +733 against a fair +459: the model would decline a
bet with a 6-point edge and would happily sell it. And the mean-calibration
check -- ratio 1.011 for QB1 -- passes cleanly the whole time, because the
mean is right. That is why "on average correct" is not the finish line.

Measured on 2025 (see CONTEXT.md), ranks 1-3:

    passing_yards  over 224.5   model 0.112   actual 0.176
    passing_yards  over 174.5   model 0.198   actual 0.257

Every sign the same, and only above the median. That is the signature of a
distribution that is too narrow rather than a mean that is too low.

THE DECOMPOSITION THIS SCRIPT MEASURES
--------------------------------------
Stage 1 produces opportunities T, Stage 2 an outcome Y per opportunity, and
the prop is X = Y_1 + ... + Y_T. For independent Y,

    Var(X) = E[T] * Var(Y)  +  Var(T) * E[Y]^2
             \_____________/    \______________/
              Stage 2 width      Stage 1 width

and Stage 1's own variance comes from team volume N and share p:

    T ~ Multinomial(N, p)  ->  Var(T) = p^2 * Var(N) + p(1-p) * E[N]

So there are three places width can go missing, and a fix aimed at the
wrong one is wasted work. Each is measured against reality separately
below, using the SAME ratio in every case:

    calibration ratio  =  mean predicted variance / mean squared error

    1.0   the spread is right
    <1.0  overconfident -- the model's distribution is too narrow
    >1.0  underconfident

MSE is used rather than the sample variance of the actuals on purpose:
E[(X - mu_i)^2] is the quantity the model's variance is a forecast OF, and
the sample variance of a heterogeneous population also contains the
between-player spread of mu_i, which the model is supposed to predict, not
absorb.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import polars as pl

from data.depth import role_bucket, role_ranks
from data.nflverse import load_pbp, load_player_stats, load_rosters_weekly
from features.efficiency import PerOpportunityModel
from features.props import (
    PROPS, opportunity_rows, player_games, roster_player_games, team_games,
)
from features.usage import ShareModel, TeamVolumeModel, opportunity_pmf
from compare_props import served_rows
from model_flags import (CROSS_TEAM_HISTORY, ROLE_RELATIVE_POSITIONS,
                         SHARE_DISPERSION, SHARE_MIXTURE_POSITIONS)


def moments(pmf: np.ndarray, support: np.ndarray) -> tuple[float, float]:
    m = float(pmf @ support)
    return m, float(pmf @ (support ** 2) - m ** 2)


def ratio_line(label: str, pred_var: np.ndarray, err2: np.ndarray, n: int) -> None:
    pv, e2 = float(np.mean(pred_var)), float(np.mean(err2))
    r = pv / e2 if e2 > 0 else float("nan")
    # sigma ratio is the one to quote, because it is what moves a
    # probability: a variance ratio of 0.62 is a 21% narrow sigma.
    print(f"  {label:34} n={n:6}  pred var {pv:9.3f}  MSE {e2:9.3f}  "
          f"var ratio {r:6.3f}  sigma ratio {np.sqrt(r):6.3f}")


# ---------------------------------------------------------------------
# Stage 1a: team volume
# ---------------------------------------------------------------------

def team_volume_width(spec, ps, score_season, min_week, trailing, prior_seasons):
    """
    Is the negative binomial on team volume the right width?

    Worth checking first because `TeamVolumeModel.pmf` scales the variance
    by the LEAGUE var/mean ratio, and `league_var` is the variance of the
    pooled team-game distribution -- which contains between-team spread that
    `team_means` already models separately. That double-counts, so the
    plausible error here is too WIDE, not too narrow.
    """
    tg = team_games(player_games(spec, ps))
    weeks = sorted(tg.filter(pl.col("season") == score_season)["week"].unique().to_list())
    pv, e2 = [], []
    for week in weeks:
        if week < min_week:
            continue
        before = (pl.col("season") < score_season) | (
            (pl.col("season") == score_season) & (pl.col("week") < week))
        past = tg.filter(before)
        if past.height < 150:
            continue
        vol = TeamVolumeModel.fit(past, trailing=trailing, prior_seasons=prior_seasons)
        now = tg.filter((pl.col("season") == score_season) & (pl.col("week") == week))
        for r in now.iter_rows(named=True):
            m, v = moments(vol.pmf(r["team"]), vol.support)
            pv.append(v)
            e2.append((r["N"] - m) ** 2)
    return np.asarray(pv), np.asarray(e2)


# ---------------------------------------------------------------------
# Stage 1b: player opportunities
# ---------------------------------------------------------------------

def opportunity_width(spec, ps, rr, statuses, score_season, min_week,
                      trailing, prior_seasons, share_dispersion=False,
                      mixture_positions=(), role_relative=()):
    """
    Team volume and share compounded, against realised opportunities. This
    is the whole of Stage 1, on the served population.
    """
    stats_pg = player_games(spec, ps)
    pg = roster_player_games(spec, ps, rr, statuses)
    tg = team_games(stats_pg)
    weeks = sorted(stats_pg.filter(pl.col("season") == score_season)["week"]
                   .unique().to_list())
    rows = []
    for week in weeks:
        if week < min_week:
            continue
        before = (pl.col("season") < score_season) | (
            (pl.col("season") == score_season) & (pl.col("week") < week))
        past_pg, past_tg = pg.filter(before), tg.filter(before)
        if past_tg.height < 150:
            continue
        cur = (rr.filter((pl.col("season") == score_season) & (pl.col("week") == week))
                 .select(["team", "player_id", "role_pos", "role_bucket"])
                 .unique(subset=["team", "player_id"]))
        look = {(r["team"], r["player_id"]): (r["role_pos"], int(r["role_bucket"]))
                for r in cur.iter_rows(named=True)}
        vol = TeamVolumeModel.fit(past_tg, trailing=trailing, prior_seasons=prior_seasons)
        shr = ShareModel.fit(past_pg, trailing=trailing, roles=cur,
                             share_dispersion=share_dispersion,
                             mixture_positions=mixture_positions,
                             role_relative=role_relative)

        now = served_rows(spec, stats_pg, rr, statuses, score_season, week)
        if now.is_empty() or "opportunities" not in now.columns:
            continue
        for team, grp in now.group_by("team"):
            tn = team[0] if isinstance(team, tuple) else team
            pids = grp["player_id"].to_list()
            roles = [look.get((tn, p)) for p in pids]
            shares = shr.team_vector(tn, pids, roles=roles,
                                     positions=grp["position"].to_list())
            vpmf = vol.pmf(tn)
            for pid, role, share, act in zip(pids, roles, shares,
                                             grp["opportunities"].to_list()):
                mix = shr.mixture_for(role)
                conc = (shr.concentration_for(role)
                        if share_dispersion and mix is None else None)
                opmf = opportunity_pmf(vpmf, vol.support, float(share),
                                       share_concentration=conc,
                                       share_mixture=mix)
                m, v = moments(opmf, np.arange(len(opmf)))
                rows.append({"role_pos": role[0] if role else None,
                             "role_bucket": role[1] if role else None,
                             "pred_mean": m, "pred_var": v, "actual": act,
                             "err2": (act - m) ** 2})
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------
# Stage 2: per-opportunity outcome
# ---------------------------------------------------------------------

def per_opportunity_width(spec, pbp, score_season, min_week):
    """
    The per-opportunity pmf against the realised spread of single
    opportunities. No compounding, so this isolates Stage 2.
    """
    rows_all = opportunity_rows(spec, pbp)
    weeks = sorted(rows_all.filter(pl.col("season") == score_season)["week"]
                   .unique().to_list())
    pv, e2 = [], []
    for week in weeks:
        if week < min_week:
            continue
        before = (pl.col("season") < score_season) | (
            (pl.col("season") == score_season) & (pl.col("week") < week))
        past = rows_all.filter(before)
        if past.height < 2000:
            continue
        po = PerOpportunityModel.fit(past, outcome_col="outcome",
                                     shape_col=None, support=spec.pt_support)
        m, v = moments(po.pmf_for(None), spec.pt_support)
        now = rows_all.filter((pl.col("season") == score_season)
                              & (pl.col("week") == week))["outcome"].to_numpy()
        pv.append(np.full(len(now), v))
        e2.append((now - m) ** 2)
    if not pv:
        return np.array([]), np.array([])
    return np.concatenate(pv), np.concatenate(e2)


# ---------------------------------------------------------------------
# End to end, from the harness output
# ---------------------------------------------------------------------

def end_to_end(prop: str, tag: str, top: int) -> None:
    path = f"cache/compare_{prop}{tag}.parquet"
    if not os.path.exists(path):
        print(f"  (no {path}; run compare_props.py --tag {tag})")
        return
    df = pl.read_parquet(path)
    if "var" not in df.columns:
        print(f"  ({path} predates the variance column; rerun compare_props.py)")
        return
    df = df.with_columns(((pl.col("actual") - pl.col("exp")) ** 2).alias("err2"))
    ratio_line("end to end, all rows",
               df["var"].to_numpy(), df["err2"].to_numpy(), df.height)
    sub = df.filter(pl.col("role_bucket") <= top)
    if not sub.is_empty():
        ratio_line(f"end to end, ranks 1-{top} (priced)",
                   sub["var"].to_numpy(), sub["err2"].to_numpy(), sub.height)
    for r in (df.group_by(["role_pos", "role_bucket"])
                .agg([pl.len().alias("n"), pl.col("var").mean().alias("pv"),
                      pl.col("err2").mean().alias("e2")])
                .filter((pl.col("n") >= 100) & (pl.col("role_bucket") <= top))
                .sort(["role_pos", "role_bucket"]).iter_rows(named=True)):
        rr_ = r["pv"] / r["e2"] if r["e2"] > 0 else float("nan")
        print(f"      {r['role_pos']}{r['role_bucket']:<3} n={r['n']:5}  "
              f"pred var {r['pv']:9.3f}  MSE {r['e2']:9.3f}  "
              f"var ratio {rr_:6.3f}  sigma ratio {np.sqrt(rr_):6.3f}")


def main(a) -> None:
    pbp = load_pbp(a.seasons)
    ps = load_player_stats(a.seasons)
    rr = role_ranks(a.seasons).with_columns(
        role_bucket(pl.col("role_rank")).alias("role_bucket"))
    try:
        statuses = load_rosters_weekly(a.seasons)
    except Exception as e:  # noqa: BLE001
        print(f"  rosters_weekly unavailable ({str(e)[:60]})")
        statuses = None

    for name in a.props:
        spec = PROPS[name]
        print(f"\n{'=' * 92}\n{name}\n{'=' * 92}")

        end_to_end(name, a.tag, a.top)

        pv, e2 = team_volume_width(spec, ps, a.score_season, a.min_week,
                                   a.trailing, a.prior_seasons)
        if len(pv):
            ratio_line("stage 1a: team volume", pv, e2, len(pv))

        ow = opportunity_width(spec, ps, rr, statuses, a.score_season,
                               a.min_week, a.trailing, a.prior_seasons,
                               a.share_dispersion,
                               tuple(a.mixture_positions),
                               tuple(a.role_relative))
        if not ow.is_empty():
            ratio_line("stage 1: player opportunities",
                       ow["pred_var"].to_numpy(), ow["err2"].to_numpy(), ow.height)
            sub = ow.filter(pl.col("role_bucket") <= a.top)
            if not sub.is_empty():
                ratio_line(f"stage 1: opportunities, ranks 1-{a.top}",
                           sub["pred_var"].to_numpy(), sub["err2"].to_numpy(),
                           sub.height)

        pv, e2 = per_opportunity_width(spec, pbp, a.score_season, a.min_week)
        if len(pv):
            ratio_line("stage 2: per-opportunity outcome", pv, e2, len(pv))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--props", nargs="+",
                    default=["passing_yards", "receiving_yards", "receptions",
                             "rushing_yards"])
    ap.add_argument("--seasons", nargs="+", type=int,
                    default=[2022, 2023, 2024, 2025])
    ap.add_argument("--score-season", type=int, default=2025)
    ap.add_argument("--min-week", type=int, default=5)
    ap.add_argument("--trailing", type=int, default=8)
    ap.add_argument("--prior-seasons", type=int, default=2)
    ap.add_argument("--tag", default="_roster")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--share-dispersion", action="store_true",
                    default=SHARE_DISPERSION)
    ap.add_argument("--no-share-dispersion", dest="share_dispersion",
                    action="store_false")
    ap.add_argument("--mixture-positions", nargs="*",
                    default=list(SHARE_MIXTURE_POSITIONS))
    ap.add_argument("--role-relative", nargs="*",
                    default=list(ROLE_RELATIVE_POSITIONS))
    main(ap.parse_args())
