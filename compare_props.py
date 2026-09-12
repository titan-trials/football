"""
Run every prop in the registry through the same rolling-origin harness and
print one comparison table.

    python compare_props.py --score-season 2025
    python compare_props.py --props receiving_yards rushing_yards

WHAT THIS IS FOR
----------------
Not to pick a winner to deploy -- all four are the same model with
different data. It is to find out where the architecture holds and where it
does not, because the places it does not are where the next real work is.

A prop with low skill against climatology is telling you either that its
opportunities are unpredictable (a Stage 1 problem) or that its
per-opportunity outcome is (a Stage 2 problem), and the two baselines below
separate those.

THE POPULATION IT SCORES
------------------------
The rows `predict_slate.py` would have produced: the depth chart, minus the
players known days ahead not to be playing. **Not** the weekly stats table,
which is what this harness scored until 2026-09-12 and which contains only
players who recorded something. Those are different populations by a factor
of two, and scoring the smaller one let a model that over-predicts deep
reserves measure unbiased, because the rows it over-predicted were the ones
missing from the sample.

`--population stats` reproduces the old behaviour, and `--fit stats`
reproduces the old fitting construction, so any change in a number can be
attributed rather than guessed at.

BASELINES
---------
  climatology   the player's own trailing distribution, ignoring team and
                matchup. The honest null. Beating it is necessary, not
                impressive -- and every number here is against it, not
                against a posted line, which is the bar that actually
                matters and has not been attempted yet.

                **It is also a measurement instrument and it needs a
                determinism test.** Its history was built by iterating a
                frame in whatever order polars produced, so two runs of the
                identical command printed climatology CRPS 8.124 and 8.068
                while the model column never moved. Every published skill
                figure before 2026-09-12 was against an inflated null. See
                `trailing_history`.
  shapeless     the model with the shape parameter replaced by the league
                value. Isolates what Stage 2's structure is worth. Equal to
                the model by construction for props with shape=None.
"""
from __future__ import annotations

import argparse

import numpy as np
import polars as pl

from data.depth import role_bucket, role_ranks
from data.nflverse import load_pbp, load_player_stats, load_rosters_weekly
from features.efficiency import PerOpportunityModel, ShapeModel, prob_over, total_pmf
from features.props import (
    PROPS, label_rows_with_asof_shape, opportunity_rows, player_games,
    roster_player_games, shape_per_game, team_games,
)
from features.usage import ShareModel, TeamVolumeModel, opportunity_pmf
from model.scoring import brier_skill, crps_pmf
# THE SAME FUNCTION THE PREDICTOR USES, not a copy of it. The whole point of
# this rewrite is that the harness and the predictor cannot disagree about
# who is on the slate, and two implementations of one rule always drift.
from model_flags import (ROLE_RELATIVE_POSITIONS, SHARE_DISPERSION,
                         SHARE_MIXTURE_POSITIONS)
from predict_slate import restrict_to_available


def climatology_pmf(history: np.ndarray, support: np.ndarray) -> np.ndarray:
    pmf = np.ones(len(support)) * 1e-7
    for h in history:
        idx = int(np.clip(round(h), support[0], support[-1])) - support[0]
        pmf[idx] += 1.0
    return pmf / pmf.sum()


def trailing_history(past_pg: pl.DataFrame) -> dict:
    """
    Per-player outcome history in CHRONOLOGICAL order, for climatology.

    THE SORT IS LOAD-BEARING. The caller takes `hist[pid][-trailing:]`, so
    without an explicit order it takes whichever rows the frame happened to
    be arranged in. Two runs of the identical command printed climatology
    CRPS 8.124 and 8.068 for receiving yards: the NULL moved between runs
    while the model did not, because ShareModel and TeamVolumeModel sort
    internally and this loop did not.

    Every skill-against-climatology figure published before 2026-09-12 was
    measured against a baseline built this way and is therefore not
    reproducible -- and was flattering, because a scrambled history is a
    worse predictor than a recent one. A baseline that moves between runs
    is not a baseline.
    """
    hist: dict = {}
    hp = past_pg.sort(["player_id", "season", "week"])
    for pid, y in zip(hp["player_id"].to_list(), hp["outcome"].to_list()):
        hist.setdefault(pid, []).append(y)
    return hist


def served_rows(spec, stats_pg, rr, statuses, season, week) -> pl.DataFrame:
    """
    The rows the PREDICTOR would have produced for this week, with what
    actually happened attached.

    This is the correction that makes the published numbers describe the
    deployed model. The old harness scored `player_games` -- the weekly
    stats table -- which contains only players who recorded something. The
    predictor scores the depth chart minus the unavailable. On 2025 week 10
    those were 10.3 and 19.7 players a team, and the same model measured
    unbiased on the first and -4.30 yards biased on the second.

    A player listed and available who caught nothing IS a row here, with
    actual 0. Dropping him is how a model that under-predicts everybody
    still looks unbiased.
    """
    roster = (rr.filter((pl.col("season") == season) & (pl.col("week") == week))
                .filter(pl.col("role_pos").is_in(list(spec.positions)))
                .select(["team", "player_id", "role_pos", "role_bucket"])
                .unique(subset=["team", "player_id"]))
    if roster.is_empty():
        return roster
    roster = restrict_to_available(roster, statuses, season, week, quiet=True)

    # Only teams that actually played this week -- a bye-week depth chart is
    # not a slate, and zero-filling against one invents rows. This is the
    # same guard roster_player_games needs and for the same reason.
    played = (stats_pg.filter((pl.col("season") == season) & (pl.col("week") == week))
                      .select("team").unique())
    cols = ["team", "player_id", "outcome"]
    # Opportunities come along when present so Stage 1 can be measured on
    # its own. validate_width.py needs them and must not reimplement this
    # population rule to get them.
    if "opportunities" in stats_pg.columns:
        cols.append("opportunities")
    act = (stats_pg.filter((pl.col("season") == season) & (pl.col("week") == week))
                   .select(cols))
    fills = [pl.col("outcome").fill_null(0.0).cast(pl.Float64)]
    if "opportunities" in cols:
        fills.append(pl.col("opportunities").fill_null(0.0).cast(pl.Float64))
    return (roster.join(played, on="team", how="inner")
                  .join(act, on=["team", "player_id"], how="left")
                  .with_columns(fills)
                  .rename({"role_pos": "position"}))


def run_prop(spec, pbp, ps, rr, score_season, min_week, trailing, prior_seasons,
             statuses=None, fit="roster", population="served",
             share_dispersion=SHARE_DISPERSION,
             mixture_positions=SHARE_MIXTURE_POSITIONS,
             role_relative=ROLE_RELATIVE_POSITIONS):
    """
    `population` selects the rows that get SCORED:

      served  the depth chart minus the unavailable -- what predict_slate
              produces, and therefore the only population whose numbers
              describe the deployed model
      stats   the weekly stats table -- what this harness used to score,
              kept ONLY so the old published figures can be reproduced and
              the causes of a change separated

    `fit` selects the construction the SHARE model is fitted on:

      roster  the depth chart with zeros filled -- what predict_slate uses
      stats   the weekly stats table -- what this harness used to use

    Both are scored on the same served rows, so running the two is a clean
    A/B on the fit alone. Without the flag, "the numbers moved" is
    ambiguous between the fit changing and the population changing.
    """
    rows_all = opportunity_rows(spec, pbp)
    per_game = shape_per_game(spec, rows_all)
    rows_all = label_rows_with_asof_shape(rows_all, per_game)
    stats_pg = player_games(spec, ps)
    pg = roster_player_games(spec, ps, rr, statuses) if fit == "roster" else stats_pg
    tg = team_games(stats_pg)                  # team volume from real stats

    opp_sup = np.arange(0, spec.max_opportunities + 1)
    out = []
    weeks = sorted(stats_pg.filter(pl.col("season") == score_season)["week"]
                   .unique().to_list())

    for week in weeks:
        if week < min_week:
            continue
        before = (pl.col("season") < score_season) | (
            (pl.col("season") == score_season) & (pl.col("week") < week))
        past_pg, past_tg, past_rows = pg.filter(before), tg.filter(before), rows_all.filter(before)
        if past_tg.height < 150 or past_rows.height < 2000:
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

        if spec.has_shape:
            fit_rows = past_rows.drop_nulls("asof_shape")
            per_opp = PerOpportunityModel.fit(
                fit_rows, outcome_col="outcome", shape_col="asof_shape",
                support=spec.pt_support, edges=spec.shape_edges,
                centres=spec.shape_centres)
            shape_model = ShapeModel.fit(per_game.filter(before), roles=cur, trailing=trailing)
            league_pmf = per_opp.pmf_for(per_opp.league_shape)
        else:
            per_opp = PerOpportunityModel.fit(
                past_rows, outcome_col="outcome", shape_col=None, support=spec.pt_support)
            shape_model = None
            league_pmf = per_opp.pmf_for(None)

        if population == "served":
            now = served_rows(spec, stats_pg, rr, statuses, score_season, week)
        else:
            now = stats_pg.filter((pl.col("season") == score_season)
                                  & (pl.col("week") == week))
        if now.is_empty():
            continue
        # Climatology from the SAME construction the model is fitted on, so
        # the null and the model see one population. Reading history out of
        # the stats table while scoring the roster would hand climatology a
        # survivorship advantage and flatter the null, not the model.
        hist = trailing_history(past_pg)

        for team, grp in now.group_by("team"):
            tn = team[0] if isinstance(team, tuple) else team
            pids = grp["player_id"].to_list()
            positions = grp["position"].to_list()
            actual = np.array(grp["outcome"].to_list(), dtype=float)
            roles = [look.get((tn, p)) for p in pids]
            shares = shr.team_vector(tn, pids, roles=roles, positions=positions)
            vpmf = vol.pmf(tn)

            for pid, share, role, act in zip(pids, shares, roles, actual):
                mix = shr.mixture_for(role)
                conc = (shr.concentration_for(role)
                        if share_dispersion and mix is None else None)
                opmf = opportunity_pmf(vpmf, vol.support, float(share),
                                       share_concentration=conc,
                                       share_mixture=mix)
                opmf_t = np.zeros(len(opp_sup))
                n = min(len(opmf), len(opp_sup))
                opmf_t[:n] = opmf[:n]
                if opmf_t.sum() > 0:
                    opmf_t /= opmf_t.sum()

                sv = shape_model.value_for(tn, pid, role) if shape_model else None
                pt = per_opp.pmf_for(sv)
                pmf = total_pmf(opmf_t, opp_sup, pt, spec.max_opportunities,
                                spec.pt_support, spec.total_support)
                pmf_flat = total_pmf(opmf_t, opp_sup, league_pmf, spec.max_opportunities,
                                     spec.pt_support, spec.total_support)

                h = np.array(hist.get(pid, [])[-trailing:])
                clim = climatology_pmf(h if len(h) else np.array([0.0]), spec.total_support)

                row = {"week": week, "team": tn, "player_id": pid, "actual": act,
                       # Carried so calibration can be sliced by a label that
                       # is KNOWN BEFORE KICKOFF. Slicing by the prediction
                       # hides compression and slicing by the outcome
                       # conditions on success -- both are recorded mistakes
                       # of mine in this project.
                       "role_pos": role[0] if role else None,
                       "role_bucket": role[1] if role else None,
                       "exp": float(pmf @ spec.total_support),
                       # THE PREDICTED VARIANCE, stored so overconfidence can
                       # be measured. A bet settles on P(X > line), not on
                       # E[X], so a model with the right mean and too little
                       # spread is wrong on every price away from the median
                       # while looking perfectly calibrated on the mean.
                       "var": float(pmf @ (spec.total_support ** 2)
                                    - (pmf @ spec.total_support) ** 2),
                       "p0": float(pmf[0]),
                       "crps_model": float(crps_pmf([act], spec.total_support, pmf[None, :])[0]),
                       "crps_flat": float(crps_pmf([act], spec.total_support, pmf_flat[None, :])[0]),
                       "crps_clim": float(crps_pmf([act], spec.total_support, clim[None, :])[0])}
                for L in spec.lines:
                    row[f"p_{L}"] = prob_over(pmf, L, spec.total_support)
                    row[f"c_{L}"] = prob_over(clim, L, spec.total_support)
                out.append(row)

    return pl.DataFrame(out)


def summarise(spec, df: pl.DataFrame) -> dict:
    cm, cf, cc = df["crps_model"].mean(), df["crps_flat"].mean(), df["crps_clim"].mean()
    best_line, best_gain = None, None
    skills = []
    for L in spec.lines:
        y = (df["actual"] > L).to_numpy().astype(float)
        if y.mean() <= 0.02 or y.mean() >= 0.98:
            continue
        bm = brier_skill(y, df[f"p_{L}"].to_numpy())
        bc = brier_skill(y, df[f"c_{L}"].to_numpy())
        skills.append((L, y.mean(), bm, bc))
        if best_gain is None or bm - bc > best_gain:
            best_line, best_gain = L, bm - bc
    return {
        "prop": spec.name, "n": df.height,
        "crps_model": cm, "crps_clim": cc,
        "crps_skill_vs_clim": 1 - cm / cc,
        "shape_worth": 1 - cm / cf if cf > 0 else 0.0,
        "bias": float(df["exp"].mean() - df["actual"].mean()),
        "mean_actual": float(df["actual"].mean()),
        "lines": skills, "best_line": best_line, "best_gain": best_gain,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--props", nargs="+", default=list(PROPS))
    ap.add_argument("--seasons", nargs="+", type=int, default=[2022, 2023, 2024, 2025])
    ap.add_argument("--score-season", type=int, default=2025)
    ap.add_argument("--min-week", type=int, default=5)
    ap.add_argument("--trailing", type=int, default=8)
    ap.add_argument("--prior-seasons", type=int, default=2)
    ap.add_argument("--fit", choices=["roster", "stats"], default="roster",
                    help="share-model fitting construction; roster is what ships")
    ap.add_argument("--population", choices=["served", "stats"], default="served",
                    help="rows to score; served is what predict_slate produces")
    ap.add_argument("--tag", default="", help="suffix for the cached parquet")
    ap.add_argument("--share-dispersion", action="store_true",
                    default=SHARE_DISPERSION,
                    help="beta-binomial share instead of a fixed one")
    ap.add_argument("--no-share-dispersion", dest="share_dispersion",
                    action="store_false")
    ap.add_argument("--role-relative", nargs="*",
                    default=list(ROLE_RELATIVE_POSITIONS),
                    help="positions using the role-relative multiplier")
    ap.add_argument("--mixture-positions", nargs="*",
                    default=list(SHARE_MIXTURE_POSITIONS),
                    help="positions using the empirical share histogram; "
                         "pass with no values to disable")
    a = ap.parse_args()

    pbp = load_pbp(a.seasons)
    ps = load_player_stats(a.seasons)
    rr = role_ranks(a.seasons).with_columns(role_bucket(pl.col("role_rank")).alias("role_bucket"))
    try:
        statuses = load_rosters_weekly(a.seasons)
    except Exception as e:  # noqa: BLE001
        print(f"  rosters_weekly unavailable ({str(e)[:60]}); "
              f"inactive weeks will count as zeros")
        statuses = None

    results = []
    for name in a.props:
        spec = PROPS[name]
        print(f"running {name} ...", flush=True)
        df = run_prop(spec, pbp, ps, rr, a.score_season, a.min_week,
                      a.trailing, a.prior_seasons, statuses, a.fit, a.population,
                      a.share_dispersion, tuple(a.mixture_positions),
                      tuple(a.role_relative))
        if df.is_empty():
            print(f"  {name}: no rows")
            continue
        df.write_parquet(f"cache/compare_{name}{a.tag}.parquet")
        results.append(summarise(spec, df))

    print(f"\n{'='*86}")
    print(f"PROP COMPARISON — {a.score_season}, rolling origin, scored against "
          f"climatology\n  scored population: {a.population}"
          f"{' (depth chart minus unavailable -- what predict_slate serves)' if a.population == 'served' else ' (weekly stats table -- legacy, players who recorded something)'}"
          f"\n  share model fitted on: {a.fit}"
          f"\n  share dispersion: {'BETA-BINOMIAL' if a.share_dispersion else 'fixed share'}"
          f"\n  empirical share mixture: {', '.join(a.mixture_positions) or 'none'}"
          f"\n  role-relative share: {', '.join(a.role_relative) or 'none'}")
    print(f"{'='*86}")
    print(f"{'prop':18} {'n':>6} {'mean':>8} {'CRPS':>8} {'clim':>8} {'skill':>8} {'shape':>8} {'bias':>8}")
    for r in sorted(results, key=lambda x: -x["crps_skill_vs_clim"]):
        print(f"{r['prop']:18} {r['n']:6} {r['mean_actual']:8.1f} {r['crps_model']:8.3f} "
              f"{r['crps_clim']:8.3f} {r['crps_skill_vs_clim']:+8.4f} {r['shape_worth']:+8.4f} "
              f"{r['bias']:+8.2f}")

    print(f"\nBrier skill at book-style lines (model vs climatology)")
    for r in results:
        print(f"\n  {r['prop']}")
        for L, base, bm, bc in r["lines"]:
            print(f"    over {L:7}  base {base:.3f}   model {bm:+.4f}   clim {bc:+.4f}   gain {bm-bc:+.4f}")
