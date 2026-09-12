"""
Rolling-origin validation of the Stage 1 usage model.

    python validate_usage.py [--seasons 2021 2022 2023 2024 2025] [--min-week 5]

WHAT IT DOES
------------
Walks forward one week at a time. At each week w it fits TeamVolumeModel and
ShareModel on games strictly before w, predicts a target pmf for every
active pass-catcher, and scores it against what actually happened.

The rule this enforces is the one model-review-2026-09-05.md says the K
backtest could not be checked against: this script calls the same fit and
predict functions the predictor will call. There is no separate backtest
code path to drift out of sync.

BASELINES
---------
Two, because they answer different questions.

  climatology  -- the player's own trailing target distribution, ignoring
                  team and matchup entirely. This is the football analogue
                  of "quote the base rate". Beating it is necessary, not
                  impressive.
  team-blind   -- league-average share applied to the team's volume. Isolates
                  how much of the model's skill is knowing WHO gets targets
                  versus knowing how many the team throws.

Scored with CRPS (proper, no line needed) and, separately, Brier at the
half-integer lines a book would actually hang.
"""
from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from data.depth import role_bucket, role_ranks
from data.nflverse import load_player_stats
from features.usage import (
    MAX_OPPORTUNITIES, ShareModel, TeamVolumeModel, opportunity_pmf,
)
from model.scoring import brier_skill, crps_pmf

POSITIONS = ["WR", "TE", "RB"]
SUPPORT = np.arange(0, MAX_OPPORTUNITIES + 1)


def build_tables(seasons):
    ps = load_player_stats(seasons)
    pg = (
        ps.filter(pl.col("position").is_in(POSITIONS))
          .select(["season", "week", "team", "player_id", "position", "targets"])
          .with_columns(pl.col("targets").fill_null(0).alias("opportunities"))
          .drop("targets")
    )
    tg = (
        pg.group_by(["season", "week", "team"])
          .agg(pl.col("opportunities").sum().alias("N"))
    )
    return pg, tg


def climatology_pmf(history: np.ndarray) -> np.ndarray:
    """Empirical pmf of a player's own prior games, smoothed by one pseudo-count."""
    pmf = np.ones(len(SUPPORT)) * 1e-6
    if len(history):
        for h in history:
            h = int(min(max(h, 0), MAX_OPPORTUNITIES))
            pmf[h] += 1.0
    return pmf / pmf.sum()


def run(seasons, min_week, trailing, use_roles=True, prior_seasons=2, score_season=None):
    pg, tg = build_tables(seasons)
    rows = []

    rr = None
    if use_roles:
        rr = role_ranks(seasons).with_columns(
            role_bucket(pl.col("role_rank")).alias("role_bucket"))

    ordered = sorted(set(zip(pg["season"].to_list(), pg["week"].to_list())))
    for season, week in ordered:
        if week < min_week:
            continue
        if score_season is not None and season != score_season:
            continue
        past_pg = pg.filter((pl.col("season") < season) |
                            ((pl.col("season") == season) & (pl.col("week") < week)))
        past_tg = tg.filter((pl.col("season") < season) |
                            ((pl.col("season") == season) & (pl.col("week") < week)))
        if past_tg.height < 200:
            continue

        roles_now = None
        role_lookup = {}
        if rr is not None:
            # Depth charts are a SERVE-TIME feed: the chart for the upcoming
            # game is legitimately visible, unlike snap counts. So this week's
            # rows are allowed, and are what a live predictor would hold.
            cur = rr.filter((pl.col("season") == season) & (pl.col("week") == week))
            roles_now = cur.select(["team", "player_id", "role_pos", "role_bucket"]).unique(
                subset=["team", "player_id"])
            for r in roles_now.iter_rows(named=True):
                role_lookup[(r["team"], r["player_id"])] = (r["role_pos"], int(r["role_bucket"]))

        vol = TeamVolumeModel.fit(past_tg, trailing=trailing, prior_seasons=prior_seasons)
        shr = ShareModel.fit(past_pg, trailing=trailing, roles=roles_now)

        now = pg.filter((pl.col("season") == season) & (pl.col("week") == week))
        hist = {}
        for pid, opp in zip(past_pg["player_id"].to_list(), past_pg["opportunities"].to_list()):
            hist.setdefault(pid, []).append(opp)

        for team, grp in now.group_by("team"):
            team_name = team[0] if isinstance(team, tuple) else team
            pids = grp["player_id"].to_list()
            positions = grp["position"].to_list()
            actual = np.array(grp["opportunities"].to_list(), dtype=float)
            roles = [role_lookup.get((team_name, p)) for p in pids]
            shares = shr.team_vector(team_name, pids, roles=roles, positions=positions)
            vpmf = vol.pmf(team_name)
            n_players = max(len(pids), 1)

            for pid, share, act in zip(pids, shares, actual):
                model = opportunity_pmf(vpmf, vol.support, float(share))
                blind = opportunity_pmf(vpmf, vol.support, 1.0 / n_players)
                clim = climatology_pmf(np.array(hist.get(pid, [])[-trailing:]))
                rows.append({
                    "season": season, "week": week, "team": team_name,
                    "player_id": pid, "actual": act,
                    "crps_model": float(crps_pmf([act], SUPPORT, model[None, :])[0]),
                    "crps_blind": float(crps_pmf([act], SUPPORT, blind[None, :])[0]),
                    "crps_clim": float(crps_pmf([act], SUPPORT, clim[None, :])[0]),
                    "p_over_2.5": float(model[SUPPORT > 2.5].sum()),
                    "p_over_4.5": float(model[SUPPORT > 4.5].sum()),
                    "p_over_6.5": float(model[SUPPORT > 6.5].sum()),
                    "clim_over_4.5": float(clim[SUPPORT > 4.5].sum()),
                })

    return pl.DataFrame(rows)


def report(df: pl.DataFrame):
    n = df.height
    cm, cb, cc = df["crps_model"].mean(), df["crps_blind"].mean(), df["crps_clim"].mean()
    print(f"\n=== Stage 1 usage model, rolling origin ===")
    print(f"rows: {n}   weeks: {df['week'].n_unique()}   seasons: {sorted(df['season'].unique().to_list())}")
    print(f"\nmean CRPS (lower is better)")
    print(f"  model                  {cm:.4f}")
    print(f"  team-blind (equal share) {cb:.4f}    model skill vs blind  {1 - cm / cb:+.4f}")
    print(f"  climatology            {cc:.4f}    model skill vs clim   {1 - cm / cc:+.4f}")

    print(f"\nBrier skill at posted-style lines (pooled, vs that line's own base rate)")
    for line in ["2.5", "4.5", "6.5"]:
        y = (df["actual"] > float(line)).to_numpy().astype(float)
        p = df[f"p_over_{line}"].to_numpy()
        print(f"  over {line}:  base rate {y.mean():.3f}   model skill {brier_skill(y, p):+.4f}")

    y = (df["actual"] > 4.5).to_numpy().astype(float)
    print(f"  over 4.5, climatology for comparison:      {brier_skill(y, df['clim_over_4.5'].to_numpy()):+.4f}")

    print(f"\ncalibration, over 4.5 targets")
    p = df["p_over_4.5"].to_numpy()
    edges = np.quantile(p, np.linspace(0, 1, 6))
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() < 20:
            continue
        print(f"  pred {p[m].mean():.3f}   actual {y[m].mean():.3f}   n={m.sum():5}   gap {y[m].mean() - p[m].mean():+.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int, default=list(range(2019, 2026)))
    ap.add_argument("--min-week", type=int, default=5)
    ap.add_argument("--trailing", type=int, default=8)
    ap.add_argument("--prior-seasons", type=int, default=2,
                    help="seasons in the league prior; 0 = pool all history")
    ap.add_argument("--score-season", type=int, default=None,
                    help="only score this season, but train on everything prior")
    ap.add_argument("--no-roles", action="store_true",
                    help="disable the depth-chart role prior, for the A/B")
    ap.add_argument("--out", default="cache/usage_validation.parquet")
    a = ap.parse_args()

    df = run(a.seasons, a.min_week, a.trailing, use_roles=not a.no_roles,
             prior_seasons=a.prior_seasons, score_season=a.score_season)
    print(f"\nrole prior: {'OFF' if a.no_roles else 'ON'}")
    report(df)
    df.write_parquet(a.out)
    print(f"\nwrote {df.height} scored rows to {a.out}")
    print("(lab output is saved -- the 'measured but not reported anywhere'")
    print(" problem in state-check-2026-09-06.md cost a re-run.)")
