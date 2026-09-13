"""
Predict the upcoming slate.

    python predict_slate.py                  # current week, auto-detected
    python predict_slate.py --week 3
    python predict_slate.py --season 2026 --week 3

Writes `slates/slate_<season>_wk<week>.parquet` — one row per player per
prop, with a probability at each book-style line, the availability flag, and
a per-row timestamp.

THE THREE RULES THIS ENFORCES
-----------------------------
**1. Predict before anything kicks off.** `clean-vs-tainted-rows.md`: a
cleanliness filter that correlates with start time silently re-weights the
sample. Football has five distinct windows a week — Thursday, Sunday
1pm/4pm, Sunday night, Monday — so "before the slate" is not one moment.
Every row is stamped with `predicted_at` and with its own game's kickoff,
and `clean` is a property of the ROW, not of the slate.

**2. Preserve committed rows on re-run.** Re-running on Sunday afternoon
must not overwrite Thursday's rows with hindsight. `preserve_committed_rows`
keeps any row whose game has already started and refreshes only games still
to come — ported from the baseball version, which worked and is the reason
989 clean rows survived there.

**3. Same code path as the backtest.** This calls the same `TeamVolumeModel`
/ `ShareModel` / `PerOpportunityModel` functions `compare_props.py` does.
`model-review-2026-09-05.md`'s Tier-1 bug was a deployed prop compounding a
rate the backtest never touched, with no way to compare them.

WHAT IS NOT IN HERE
-------------------
No market comparison. `data/odds.py` captures lines separately, and the
scoring joins them. Keeping the predictor market-free is what makes the
comparison honest — and MARKET_IN_VOLUME being off means nothing in this
file has seen a price.
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import numpy as np
import polars as pl

from data.availability import attach, flag_frame, report_published
from data.depth import role_bucket, role_ranks, with_player_names
from data.nflverse import (load_pbp, load_player_stats, load_rosters_weekly,
                           load_schedules, upcoming_week)
from features.efficiency import PerOpportunityModel, ShapeModel, prob_over, total_pmf
from features.props import (
    PROPS, anytime_td_probability, label_rows_with_asof_shape, opportunity_rows,
    player_games, roster_player_games, shape_per_game, team_games,
)
from features.usage import ShareModel, TeamVolumeModel, opportunity_pmf
from model_flags import (CROSS_TEAM_HISTORY, ROLE_RELATIVE_POSITIONS,
                         SHARE_DISPERSION, SHARE_MIXTURE_POSITIONS)

SLATE_DIR = "slates"


# Statuses that mean the player is not going to play. DEV is the practice
# squad, RES is injured reserve. INA is a gameday scratch and is not known
# until 90 minutes before kickoff, so it cannot be used at predict time --
# but the other four are known days ahead.
UNAVAILABLE = ("DEV", "RES", "CUT", "RET")


def restrict_to_available(roster: pl.DataFrame, statuses, season: int,
                          week: int, quiet: bool = False) -> pl.DataFrame:
    """
    Drop players who are not going to play, before shares are normalised.

    THE POPULATION MUST MATCH THE ONE THE MODEL WAS VALIDATED ON. This is
    the "same code path" rule applied to the roster rather than to a
    function, and breaking it cost most of a day.

    `roster_player_games` -- what the model is FITTED and VALIDATED on --
    keeps only ACT player-weeks. `predict_slate` was normalising shares
    across the entire depth chart. Measured on 2025 week 10: 705 skill
    players listed, **329 ACT**. Twenty-two players a team in the predictor
    against 10.3 in the validator, a dilution factor of 2.14, and every real
    contributor scaled down to make room for practice-squad bodies.

    Scoring that week through the live path showed it plainly -- receiving
    yards bias **-4.30** on a mean of 21.95, receptions -0.32, passing yards
    -43.25 -- while the same model measured unbiased in the validation
    harness. Same model, two populations.

    Falls back to the previous week's statuses when the upcoming week has
    not published yet, and keeps players with no status row rather than
    silently deleting them.

    `quiet` exists so the backtest harness can call THIS function rather
    than reimplementing it. A second copy of a population rule is how the
    two populations drifted apart in the first place.
    """
    if statuses is None or statuses.is_empty():
        return roster
    st = statuses.filter((pl.col("season") == season) & (pl.col("week") == week))
    if st.is_empty() and week > 1:
        st = statuses.filter((pl.col("season") == season) & (pl.col("week") == week - 1))
        if not st.is_empty() and not quiet:
            print(f"  no week {week} roster statuses yet; using week {week - 1}")
    if st.is_empty():
        if not quiet:
            print("  no roster statuses available; keeping the full depth chart")
        return roster

    st = (st.select(["team", "gsis_id", "status"]).rename({"gsis_id": "player_id"})
            .drop_nulls("player_id")
            .unique(subset=["team", "player_id"], keep="last"))
    before = roster.height
    out = (roster.join(st, on=["team", "player_id"], how="left")
                 .filter(pl.col("status").is_null()
                         | ~pl.col("status").is_in(list(UNAVAILABLE)))
                 .drop("status"))
    if not quiet:
        print(f"  roster {before} -> {out.height} after dropping "
              f"{'/'.join(UNAVAILABLE)} ({out.height / 32:.1f} players/team)")
    return out


def kickoffs(season: int, week: int) -> pl.DataFrame:
    """
    Per-team kickoff time for the week, in UTC. Five windows, not one.

    NFLVERSE `gametime` IS EASTERN, AND `now` IS UTC. Converting is not
    cosmetic: the first version compared a naive Eastern kickoff against a
    naive UTC clock, so every row was marked "not clean" starting four
    hours before its game actually kicked off (five in winter). The error
    was in the safe direction -- it never called a post-kickoff row clean,
    which is the direction that would have let hindsight in -- but it threw
    away four hours of legitimately clean rows and froze them early in
    `preserve_committed_rows`.

    Stored tz-naive in UTC so it compares directly with
    `datetime.now(timezone.utc).replace(tzinfo=None)`, which is what the
    rest of this file uses.
    """
    sch = load_schedules().filter((pl.col("season") == season) & (pl.col("week") == week))
    if sch.is_empty():
        return pl.DataFrame()
    k = sch.with_columns(
        (pl.col("gameday").cast(pl.Utf8) + " " + pl.col("gametime").fill_null("13:00"))
        .str.to_datetime("%Y-%m-%d %H:%M", strict=False)
        .dt.replace_time_zone("America/New_York", ambiguous="earliest")
        .dt.convert_time_zone("UTC")
        .dt.replace_time_zone(None)
        .alias("kickoff"))
    return pl.concat([
        k.select([pl.col("home_team").alias("team"), "kickoff", "game_id"]),
        k.select([pl.col("away_team").alias("team"), "kickoff", "game_id"]),
    ]).drop_nulls("kickoff")


def predict_prop(spec, pbp, ps, rr, season, week, trailing, prior_seasons, roster,
                 statuses=None):
    """One prop, for every eligible player on the slate."""
    rows_all = label_rows_with_asof_shape(
        opportunity_rows(spec, pbp), shape_per_game(spec, opportunity_rows(spec, pbp)))
    # Fit on depth-chart rosters with zeros filled, so the population the
    # model is fitted on is the population it is served on. See
    # roster_player_games for what happens when they differ.
    pg = roster_player_games(spec, ps, rr, statuses)
    tg = team_games(player_games(spec, ps))   # team volume from real stats

    before = (pl.col("season") < season) | (
        (pl.col("season") == season) & (pl.col("week") < week))
    past_pg, past_tg, past_rows = pg.filter(before), tg.filter(before), rows_all.filter(before)
    if past_tg.height < 150:
        return pl.DataFrame()

    cur = (rr.filter((pl.col("season") == season) & (pl.col("week") == week))
             .select(["team", "player_id", "role_pos", "role_bucket"])
             .unique(subset=["team", "player_id"]))
    look = {(r["team"], r["player_id"]): (r["role_pos"], int(r["role_bucket"]))
            for r in cur.iter_rows(named=True)}

    vol = TeamVolumeModel.fit(past_tg, trailing=trailing, prior_seasons=prior_seasons)
    shr = ShareModel.fit(past_pg, trailing=trailing, roles=cur,
                         share_dispersion=SHARE_DISPERSION,
                         mixture_positions=SHARE_MIXTURE_POSITIONS,
                         role_relative=ROLE_RELATIVE_POSITIONS,
                         cross_team=CROSS_TEAM_HISTORY)

    if spec.has_shape:
        per_opp = PerOpportunityModel.fit(
            past_rows.drop_nulls("asof_shape"), outcome_col="outcome",
            shape_col="asof_shape", support=spec.pt_support,
            edges=spec.shape_edges, centres=spec.shape_centres)
        shape_model = ShapeModel.fit(
            shape_per_game(spec, rows_all).filter(before), roles=cur, trailing=trailing)
    else:
        per_opp = PerOpportunityModel.fit(
            past_rows, outcome_col="outcome", shape_col=None, support=spec.pt_support)
        shape_model = None

    opp_sup = np.arange(0, spec.max_opportunities + 1)
    out = []
    # The roster for THIS week comes from the depth chart, not from a stats
    # table -- the stats table for an unplayed game does not exist yet. This
    # is the single biggest difference between predicting and backtesting,
    # and the most likely place for a serve-time bug.
    elig = roster.filter(pl.col("role_pos").is_in(list(spec.positions)))

    for team, grp in elig.group_by("team"):
        tn = team[0] if isinstance(team, tuple) else team
        pids = grp["player_id"].to_list()
        names = grp["player_name"].to_list() if "player_name" in grp.columns else pids
        poss = grp["role_pos"].to_list()
        roles = [look.get((tn, p)) for p in pids]
        shares = shr.team_vector(tn, pids, roles=roles, positions=poss)
        vpmf = vol.pmf(tn)

        for pid, nm, pos, share, role in zip(pids, names, poss, shares, roles):
            mix = shr.mixture_for(role)
            conc = (shr.concentration_for(role)
                    if SHARE_DISPERSION and mix is None else None)
            opmf = opportunity_pmf(vpmf, vol.support, float(share),
                                   share_concentration=conc,
                                   share_mixture=mix)
            opmf_t = np.zeros(len(opp_sup))
            n = min(len(opmf), len(opp_sup))
            opmf_t[:n] = opmf[:n]
            if opmf_t.sum() > 0:
                opmf_t /= opmf_t.sum()

            sv = shape_model.value_for(tn, pid, role) if shape_model else None
            pmf = total_pmf(opmf_t, opp_sup, per_opp.pmf_for(sv), spec.max_opportunities,
                            spec.pt_support, spec.total_support)
            row = {
                "season": season, "week": week, "team": tn, "player_id": pid,
                "player_name": nm, "position": pos, "prop": spec.name,
                "exp_opportunities": float(opmf_t @ opp_sup),
                "share": float(share),
                "shape": float(sv) if sv is not None else None,
                "expected": float(pmf @ spec.total_support),
                "p_zero": float(pmf[spec.total_support == 0].sum()),
            }
            for L in spec.lines:
                row[f"over_{L}"] = prob_over(pmf, L, spec.total_support)
            # The full pmf, so the model can be evaluated at whatever line
            # the book actually hangs. Books post 13.5 and 47.5, not the
            # round numbers in spec.lines, and interpolating between stored
            # columns would put an approximation error inside the edge --
            # which is the one number that has to be exactly right.
            row["_pmf"] = pmf.tolist()
            row["_support_min"] = int(spec.total_support[0])
            out.append(row)
    return pl.DataFrame(out)


def enforce_one_row_per_player(slate: pl.DataFrame, roster: pl.DataFrame) -> pl.DataFrame:
    """
    Exactly one row per (player, prop). Where a player appears twice on
    different teams, keep the one matching the current depth chart.

    WHY THIS EXISTS ON TOP OF THE DEPTH-CHART FIX. `preserve_committed_rows`
    is faithful by design -- it keeps rows written before kickoff so a later
    re-run cannot overwrite them with hindsight. That also means it
    faithfully preserves rows written by a version with a bug. When the
    double-roster bug was fixed upstream, the corrected slate still carried
    Kayshon Boutte on two teams, because one of those rows was committed.

    So preservation needs a companion: fix the data, and also reconcile what
    was already committed against it. Preserving history is not the same as
    preserving mistakes.
    """
    if slate.is_empty() or "player_id" not in slate.columns:
        return slate
    key = ["player_id", "prop"]
    dupes = slate.group_by(key).len().filter(pl.col("len") > 1)
    if dupes.is_empty():
        return slate

    current = roster.select(["player_id", "team"]).unique().with_columns(
        pl.lit(True).alias("_on_current_chart"))
    marked = slate.join(current, on=["player_id", "team"], how="left").with_columns(
        pl.col("_on_current_chart").fill_null(False))
    kept = (marked.sort(["_on_current_chart", "predicted_at"], descending=[True, True])
                  .group_by(key, maintain_order=True).first()
                  .drop("_on_current_chart"))
    print(f"  deduped {slate.height - kept.height} rows for "
          f"{dupes.height} player-props on more than one team")
    return kept


def preserve_committed_rows(fresh: pl.DataFrame, path: str,
                            now: datetime) -> pl.DataFrame:
    """
    Keep any previously written row whose game has already kicked off;
    refresh only games still to come.

    Re-running through the weekend accumulates clean coverage instead of
    trading it away. Without this, a Sunday-evening re-run silently
    replaces Thursday's honest prediction with one made after the fact.
    """
    if not os.path.exists(path):
        return fresh
    old = pl.read_parquet(path)
    if "kickoff" not in old.columns:
        return fresh
    committed = old.filter(pl.col("kickoff") <= now)
    if committed.is_empty():
        return fresh
    still_open = fresh.join(
        committed.select(["team", "player_id", "prop"]).unique(),
        on=["team", "player_id", "prop"], how="anti")
    print(f"  preserved {committed.height} committed rows, refreshed {still_open.height}")
    return pl.concat([committed, still_open], how="diagonal_relaxed")


def main(season, week, trailing, prior_seasons, props):
    os.makedirs(SLATE_DIR, exist_ok=True)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    seasons = list(range(season - 3, season + 1))

    pbp = load_pbp(seasons)
    ps = load_player_stats(seasons)
    rr = role_ranks(seasons).with_columns(role_bucket(pl.col("role_rank")).alias("role_bucket"))
    try:
        statuses = load_rosters_weekly(seasons)
    except Exception as e:  # noqa: BLE001
        print(f"  rosters_weekly unavailable ({str(e)[:60]}); "
              f"inactive weeks will count as zeros")
        statuses = None

    roster = with_player_names(
        rr.filter((pl.col("season") == season) & (pl.col("week") == week))
          .unique(subset=["team", "player_id"]))
    roster = restrict_to_available(roster, statuses, season, week)
    if roster.is_empty():
        raise SystemExit(
            f"No depth-chart rows for {season} week {week}. The chart for an "
            f"upcoming game publishes daily at 07:00 UTC; if the week has not "
            f"been scheduled yet there is nothing to predict.")
    if "player_name" not in roster.columns:
        roster = roster.with_columns(pl.col("player_id").alias("player_name"))
    print(f"slate: {season} week {week} — {roster.height} players on depth charts")

    frames = []
    for name in props:
        spec = PROPS[name]
        df = predict_prop(spec, pbp, ps, rr, season, week, trailing,
                          prior_seasons, roster, statuses)
        print(f"  {name:16} {df.height} rows")
        if not df.is_empty():
            frames.append(df)
    if not frames:
        raise SystemExit("nothing predicted")

    slate = pl.concat(frames, how="diagonal_relaxed")

    ko = kickoffs(season, week)
    if not ko.is_empty():
        slate = slate.join(ko, on="team", how="left")
    slate = slate.with_columns([
        pl.lit(now).alias("predicted_at"),
        (pl.lit(now) < pl.col("kickoff")).alias("clean"),
    ])
    flags = flag_frame([season], through_week=week, current_season=season)
    published = report_published(flags, season, week)
    slate = attach(slate, flags if published else pl.DataFrame())

    path = os.path.join(SLATE_DIR, f"slate_{season}_wk{week}.parquet")
    slate = preserve_committed_rows(slate, path, now)
    slate = enforce_one_row_per_player(slate, roster)

    # The pmf columns live in a companion file: they are ~350 floats a row
    # and would make the slate itself unreadable in a dataframe viewer.
    pmf_path = os.path.join(SLATE_DIR, f"pmf_{season}_wk{week}.parquet")
    if "_pmf" in slate.columns:
        (slate.select(["season", "week", "team", "player_id", "prop",
                       "_pmf", "_support_min"])
              .write_parquet(pmf_path))
        slate = slate.drop(["_pmf", "_support_min"])
    slate.write_parquet(path)

    # anytime TD, combined across the two scoring routes
    td = (slate.filter(pl.col("prop").is_in(["receiving_tds", "rushing_tds"]))
               .select(["team", "player_id", "player_name", "prop", "p_zero"]))
    if not td.is_empty():
        w = td.pivot(values="p_zero", index=["team", "player_id", "player_name"],
                     on="prop", aggregate_function="first")
        for c in ("receiving_tds", "rushing_tds"):
            if c not in w.columns:
                w = w.with_columns(pl.lit(1.0).alias(c))
        w = w.with_columns(
            (1.0 - pl.col("receiving_tds").fill_null(1.0)
                 * pl.col("rushing_tds").fill_null(1.0)).alias("anytime_td"))
        w.write_parquet(os.path.join(SLATE_DIR, f"anytime_td_{season}_wk{week}.parquet"))
        top = w.sort("anytime_td", descending=True).head(8)
        print("\n  top anytime-TD probabilities:")
        for r in top.iter_rows(named=True):
            print(f"    {r['player_name'][:24]:26} {r['team']:4} {r['anytime_td']:.3f}")

    n_clean = slate.filter(pl.col("clean")).height
    print(f"\nwrote {slate.height} rows to {path}")
    print(f"  clean (predicted before kickoff): {n_clean} / {slate.height}")
    if published:
        flagged = slate.filter(pl.col("availability") != "CLEAR")
        print(f"  availability-flagged rows: {flagged.height} "
              f"({flagged['availability'].n_unique()} distinct statuses)")
    else:
        print(f"  availability: NO injury report published yet for "
              f"{season} week {week}. Every row is marked CLEAR because the "
              f"report is ABSENT, not because players are healthy. Re-run "
              f"after Wednesday to pick it up.")
    return slate


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int, default=None)
    ap.add_argument("--trailing", type=int, default=8)
    ap.add_argument("--prior-seasons", type=int, default=2)
    ap.add_argument("--props", nargs="+", default=list(PROPS))
    a = ap.parse_args()
    wk = a.week if a.week is not None else upcoming_week(a.season)
    main(a.season, wk, a.trailing, a.prior_seasons, a.props)
