"""
Choose the shrinkage strength k for ShareModel.

    python k_sweep.py

WHY THIS SCRIPT EXISTS
----------------------
    rate_i = (x_i + k * p_role) / (n_i + k)

k is how many games of ROLE evidence get blended into each player's own
record. It is the dial between two sources with different properties:

  * personal history  -- specific to the player, obeys no team constraint
  * role prior        -- generic, but the role priors sum to roughly the
                         team's volume BY CONSTRUCTION (WR1 6.5 + WR2 5.6 +
                         WR3 3.9 + TE1 4.4 + RB1 3.2 + ... ~ 30)

At the current k = 0.5, rates are almost entirely personal history, and
their sum came to 48 against a team volume of 30. `team_vector` normalises
to sum-to-one, so that 1.61 ratio lands as a uniform 0.62x multiplier on
every player -- measured shortfall 0.44/0.47/0.56/0.59 across quartiles,
i.e. flat, which is the signature of a scale error rather than a ranking
error.

Raising k pulls rates toward the constrained source. It also drags genuinely
exceptional players toward the average for their slot, which is the cost.
This sweep measures both.

PROTOCOL
--------
Select on 2023-2024, verify on 2025. NEVER on the live market: this week's
596 prop rows are correlated within game and are the thing the model will
be judged against, so tuning on them is fitting to the scoreboard.

An earlier version of this sweep found more shrinkage strictly worse -- but
it ran on the old stats-table roster construction, which no longer exists.
That is why it is being re-run rather than cited.
"""
from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from data.depth import role_bucket, role_ranks
from data.nflverse import load_player_stats, load_rosters_weekly
from features.props import PROPS, player_games, roster_player_games, team_games
from features.usage import ShareModel, TeamVolumeModel, opportunity_pmf
from model.scoring import brier_skill, crps_pmf


def evaluate(spec, pg, tg, rr, statuses, seasons_to_score, kmult,
             trailing=8, prior_seasons=2, min_week=5):
    opp_sup = np.arange(0, spec.max_opportunities + 1)
    rows, ratios = [], []

    for season in seasons_to_score:
        weeks = sorted(pg.filter(pl.col("season") == season)["week"].unique().to_list())
        for week in weeks:
            if week < min_week:
                continue
            before = (pl.col("season") < season) | (
                (pl.col("season") == season) & (pl.col("week") < week))
            past_pg, past_tg = pg.filter(before), tg.filter(before)
            if past_tg.height < 150:
                continue

            cur = (rr.filter((pl.col("season") == season) & (pl.col("week") == week))
                     .select(["team", "player_id", "role_pos", "role_bucket"])
                     .unique(subset=["team", "player_id"]))
            look = {(r["team"], r["player_id"]): (r["role_pos"], int(r["role_bucket"]))
                    for r in cur.iter_rows(named=True)}

            vol = TeamVolumeModel.fit(past_tg, trailing=trailing,
                                      prior_seasons=prior_seasons)
            shr = ShareModel.fit(past_pg, trailing=trailing, roles=cur)
            # Refit every rate at the swept k.
            shr.prior_k = shr.prior_k * kmult
            shr.rates = {
                key: (s + shr.prior_k * shr._prior_for(look.get(key), None))
                     / (n + shr.prior_k)
                for key, (s, n) in shr.counts.items()
            }

            # Diagnostic: do the unnormalised rates sum to the team's volume?
            for team, g in cur.group_by("team"):
                tn = team[0] if isinstance(team, tuple) else team
                pids = g["player_id"].to_list()
                tot = sum(shr.rate_for(tn, p, look.get((tn, p))) for p in pids)
                N = vol.mean_for(tn)
                if N > 0:
                    ratios.append(tot / N)

            now = pg.filter((pl.col("season") == season) & (pl.col("week") == week))
            for team, g in now.group_by("team"):
                tn = team[0] if isinstance(team, tuple) else team
                pids = g["player_id"].to_list()
                poss = g["position"].to_list()
                actual = np.array(g["opportunities"].to_list(), dtype=float)
                roles = [look.get((tn, p)) for p in pids]
                shares = shr.team_vector(tn, pids, roles=roles, positions=poss)
                vpmf = vol.pmf(tn)
                for share, act in zip(shares, actual):
                    pmf = opportunity_pmf(vpmf, vol.support, float(share))
                    t = np.zeros(len(opp_sup))
                    n = min(len(pmf), len(opp_sup))
                    t[:n] = pmf[:n]
                    if t.sum() > 0:
                        t /= t.sum()
                    rows.append((float(crps_pmf([act], opp_sup, t[None, :])[0]),
                                 float(t[opp_sup > 4.5].sum()), float(act),
                                 float(t @ opp_sup)))

    a = np.array(rows)
    y = (a[:, 2] > 4.5).astype(float)
    return {
        "crps": a[:, 0].mean(),
        "brier_skill": brier_skill(y, a[:, 1]),
        "pred_mean": a[:, 3].mean(),
        "actual_mean": a[:, 2].mean(),
        "bias_ratio": a[:, 3].mean() / max(a[:, 2].mean(), 1e-9),
        "sum_over_N": float(np.mean(ratios)) if ratios else float("nan"),
        "n": len(a),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prop", default="receptions")
    ap.add_argument("--mults", nargs="+", type=float,
                    default=[1, 2, 4, 8, 16, 32, 64])
    a = ap.parse_args()

    spec = PROPS[a.prop]
    seasons = [2022, 2023, 2024, 2025]
    rr = role_ranks(seasons).with_columns(
        role_bucket(pl.col("role_rank")).alias("role_bucket"))
    ps = load_player_stats(seasons)
    statuses = load_rosters_weekly(seasons)
    pg = roster_player_games(spec, ps, rr, statuses)
    tg = team_games(player_games(spec, ps))

    print(f"prop: {a.prop}   roster-based construction, "
          f"{pg.height} player-games\n")
    print("=== SELECT on 2023-2024 ===")
    print(f"{'k x':>6} {'CRPS':>8} {'Brier':>9} {'pred/act':>9} {'sum/N':>8} {'n':>7}")
    sel = {}
    for km in a.mults:
        r = evaluate(spec, pg, tg, rr, statuses, [2023, 2024], km)
        sel[km] = r
        print(f"{km:>6} {r['crps']:8.4f} {r['brier_skill']:+9.4f} "
              f"{r['bias_ratio']:9.3f} {r['sum_over_N']:8.3f} {r['n']:7}")

    best = min(sel, key=lambda k: sel[k]["crps"])
    print(f"\n  selected k multiplier: x{best}  (lowest CRPS on the "
          f"selection seasons)")

    print("\n=== VERIFY on 2025 (untouched by the selection) ===")
    print(f"{'k x':>6} {'CRPS':>8} {'Brier':>9} {'pred/act':>9} {'sum/N':>8} {'n':>7}")
    for km in sorted({1.0, best}):
        r = evaluate(spec, pg, tg, rr, statuses, [2025], km)
        print(f"{km:>6} {r['crps']:8.4f} {r['brier_skill']:+9.4f} "
              f"{r['bias_ratio']:9.3f} {r['sum_over_N']:8.3f} {r['n']:7}")
