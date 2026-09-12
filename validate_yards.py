"""
Rolling-origin validation of the full receiving-yards model.

    python validate_yards.py --seasons 2022 2023 2024 2025 --score-season 2025

Stage 1 (targets) x Stage 2 (yards per target) -> total receiving yards.
Same discipline as validate_usage.py: fits on weeks strictly before the one
being scored, and calls the same functions the predictor will.

AS-OF DISCIPLINE FOR aDOT
-------------------------
The per-target bands are fitted on historical targets labelled with the
receiver's aDOT *as it was known before that game* -- an expanding mean
shifted by one game. Labelling them with a season-long aDOT that includes
the game being predicted is the train/serve skew that broke `_project_pa`
in baseball, and it would flatter the bands badly, because aDOT computed
with hindsight separates the outcomes it is supposed to be predicting.

BASELINES
---------
  climatology  -- the player's own trailing receiving-yard distribution.
                  The honest null; beating it is necessary, not impressive.
  mean-only    -- the model's own mean, delivered as a league-shaped
                  distribution scaled to it. Isolates how much of the skill
                  comes from getting the SHAPE right versus the level.
"""
from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from data.depth import role_bucket, role_ranks
from data.nflverse import load_pbp, load_player_stats
from features.efficiency import (
    PT_SUPPORT, TOTAL_SUPPORT, AdotModel, PerTargetModel, prob_over, yards_pmf,
)
from features.usage import (
    MAX_OPPORTUNITIES, ShareModel, TeamVolumeModel, opportunity_pmf,
)
from model.scoring import brier_skill, crps_pmf

POSITIONS = ["WR", "TE", "RB"]
OPP_SUPPORT = np.arange(0, MAX_OPPORTUNITIES + 1)
LINES = [24.5, 39.5, 54.5, 69.5]


def build(seasons):
    """Target-level rows with as-of aDOT, plus player-game and team-game tables."""
    pbp = load_pbp(seasons)
    tgt = (
        pbp.filter((pl.col("pass_attempt") == 1) & pl.col("receiver_player_id").is_not_null())
           .select(["season", "week", "posteam", "receiver_player_id",
                    "yards_gained", "air_yards"])
           .rename({"posteam": "team", "receiver_player_id": "player_id"})
           .with_columns(pl.col("yards_gained").fill_null(0.0))
    )

    # Per player-game air-yard totals, for the as-of expanding aDOT.
    pergame = (
        tgt.group_by(["season", "week", "team", "player_id"])
           .agg([pl.col("air_yards").sum().alias("air_yards_sum"),
                 pl.col("air_yards").is_not_null().sum().alias("ay_targets"),
                 pl.len().alias("targets")])
           .sort(["player_id", "season", "week"])
    )
    # Expanding, SHIFTED by one game: what was known before kickoff.
    pergame = pergame.with_columns([
        pl.col("air_yards_sum").cum_sum().over("player_id").shift(1).over("player_id").alias("_ay"),
        pl.col("ay_targets").cum_sum().over("player_id").shift(1).over("player_id").alias("_n"),
    ]).with_columns(
        pl.when(pl.col("_n") > 0).then(pl.col("_ay") / pl.col("_n"))
          .otherwise(None).alias("asof_adot")
    )

    tgt = tgt.join(
        pergame.select(["season", "week", "team", "player_id", "asof_adot"]),
        on=["season", "week", "team", "player_id"], how="left",
    ).rename({"asof_adot": "player_adot"})

    ps = load_player_stats(seasons)
    pg = (
        ps.filter(pl.col("position").is_in(POSITIONS))
          .select(["season", "week", "team", "player_id", "position",
                   "targets", "receiving_yards"])
          .with_columns([pl.col("targets").fill_null(0).alias("opportunities"),
                         pl.col("receiving_yards").fill_null(0.0).alias("rec_yards")])
          .drop(["targets", "receiving_yards"])
    )
    tg = pg.group_by(["season", "week", "team"]).agg(pl.col("opportunities").sum().alias("N"))
    return tgt, pergame, pg, tg


def climatology_pmf(history: np.ndarray) -> np.ndarray:
    pmf = np.ones(len(TOTAL_SUPPORT)) * 1e-7
    for h in history:
        idx = int(np.clip(round(h), TOTAL_SUPPORT[0], TOTAL_SUPPORT[-1])) - TOTAL_SUPPORT[0]
        pmf[idx] += 1.0
    return pmf / pmf.sum()


def run(seasons, score_season, min_week, trailing, prior_seasons):
    tgt, pergame, pg, tg = build(seasons)
    rr = role_ranks(seasons).with_columns(role_bucket(pl.col("role_rank")).alias("role_bucket"))
    rows = []

    weeks = sorted(pg.filter(pl.col("season") == score_season)["week"].unique().to_list())
    for week in weeks:
        if week < min_week:
            continue
        before = (pl.col("season") < score_season) | (
            (pl.col("season") == score_season) & (pl.col("week") < week))
        past_pg, past_tg, past_tgt = pg.filter(before), tg.filter(before), tgt.filter(before)
        past_ag = pergame.filter(before)
        if past_tg.height < 200:
            continue

        cur = (rr.filter((pl.col("season") == score_season) & (pl.col("week") == week))
                 .select(["team", "player_id", "role_pos", "role_bucket"])
                 .unique(subset=["team", "player_id"]))
        look = {(r["team"], r["player_id"]): (r["role_pos"], int(r["role_bucket"]))
                for r in cur.iter_rows(named=True)}

        vol = TeamVolumeModel.fit(past_tg, trailing=trailing, prior_seasons=prior_seasons)
        shr = ShareModel.fit(past_pg, trailing=trailing, roles=cur)
        per_target = PerTargetModel.fit(past_tgt.drop_nulls("player_adot"))
        adot = AdotModel.fit(
            past_ag.rename({"ay_targets": "n_ay"}).with_columns(pl.col("n_ay").alias("targets")),
            roles=cur, trailing=trailing)

        now = pg.filter((pl.col("season") == score_season) & (pl.col("week") == week))
        hist = {}
        for pid, y in zip(past_pg["player_id"].to_list(), past_pg["rec_yards"].to_list()):
            hist.setdefault(pid, []).append(y)

        for team, grp in now.group_by("team"):
            tn = team[0] if isinstance(team, tuple) else team
            pids = grp["player_id"].to_list()
            positions = grp["position"].to_list()
            actual = np.array(grp["rec_yards"].to_list(), dtype=float)
            roles = [look.get((tn, p)) for p in pids]
            shares = shr.team_vector(tn, pids, roles=roles, positions=positions)
            vpmf = vol.pmf(tn)

            for pid, share, role, act in zip(pids, shares, roles, actual):
                tpmf = opportunity_pmf(vpmf, vol.support, float(share))
                a = adot.adot_for(tn, pid, role)
                pt = per_target.pmf_for_adot(a)
                ypmf = yards_pmf(tpmf, vol.support, pt)

                # mean-only baseline: league-shaped, scaled to the same mean
                league_pt = per_target.pmf_for_adot(per_target.league_adot)
                ypmf_mean = yards_pmf(tpmf, vol.support, league_pt)

                h = np.array(hist.get(pid, [])[-trailing:])
                clim = climatology_pmf(h) if len(h) else climatology_pmf(np.array([0.0]))

                row = {"season": score_season, "week": week, "team": tn, "player_id": pid,
                       "actual": act, "adot": a, "share": float(share),
                       "exp_yards": float(ypmf @ TOTAL_SUPPORT),
                       "crps_model": float(crps_pmf([act], TOTAL_SUPPORT, ypmf[None, :])[0]),
                       "crps_shapeless": float(crps_pmf([act], TOTAL_SUPPORT, ypmf_mean[None, :])[0]),
                       "crps_clim": float(crps_pmf([act], TOTAL_SUPPORT, clim[None, :])[0])}
                for L in LINES:
                    row[f"p_over_{L}"] = prob_over(ypmf, L)
                    row[f"clim_over_{L}"] = prob_over(clim, L)
                rows.append(row)

    return pl.DataFrame(rows)


def report(df: pl.DataFrame):
    print(f"\n=== Receiving yards, Stage 1 x Stage 2, rolling origin ===")
    print(f"rows: {df.height}   weeks: {df['week'].n_unique()}   season: {df['season'][0]}")
    cm, cs, cc = df["crps_model"].mean(), df["crps_shapeless"].mean(), df["crps_clim"].mean()
    print(f"\nmean CRPS (lower is better)")
    print(f"  model                 {cm:.3f}")
    print(f"  league-shape baseline {cs:.3f}    aDOT shape is worth {1 - cm / cs:+.4f}")
    print(f"  climatology           {cc:.3f}    model skill vs clim {1 - cm / cc:+.4f}")

    print(f"\nBrier skill at yardage lines (pooled)")
    for L in LINES:
        y = (df["actual"] > L).to_numpy().astype(float)
        if y.mean() <= 0 or y.mean() >= 1:
            continue
        bm = brier_skill(y, df[f"p_over_{L}"].to_numpy())
        bc = brier_skill(y, df[f"clim_over_{L}"].to_numpy())
        print(f"  over {L:5}:  base {y.mean():.3f}   model {bm:+.4f}   climatology {bc:+.4f}")

    L = 39.5
    y = (df["actual"] > L).to_numpy().astype(float)
    p = df[f"p_over_{L}"].to_numpy()
    print(f"\ncalibration, over {L} yards")
    edges = np.quantile(p, np.linspace(0, 1, 6))
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() < 20:
            continue
        print(f"  pred {p[m].mean():.3f}   actual {y[m].mean():.3f}   n={m.sum():5}   gap {y[m].mean() - p[m].mean():+.3f}")

    print(f"\nmean predicted yards {df['exp_yards'].mean():.2f}   mean actual {df['actual'].mean():.2f}   bias {df['exp_yards'].mean() - df['actual'].mean():+.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int, default=[2022, 2023, 2024, 2025])
    ap.add_argument("--score-season", type=int, default=2025)
    ap.add_argument("--min-week", type=int, default=5)
    ap.add_argument("--trailing", type=int, default=8)
    ap.add_argument("--prior-seasons", type=int, default=2)
    ap.add_argument("--out", default="cache/yards_validation.parquet")
    a = ap.parse_args()

    df = run(a.seasons, a.score_season, a.min_week, a.trailing, a.prior_seasons)
    report(df)
    df.write_parquet(a.out)
    print(f"\nwrote {df.height} scored rows to {a.out}")
