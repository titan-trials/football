"""
Depth-chart role ranks, normalised across two incompatible schemas.

WHY THIS MODULE EXISTS
----------------------
`features/usage.py` gives a player with no history a share of 0, which is a
point mass at zero. Validation caught it: the bottom calibration bucket
predicted 0.001 against an actual 0.016, so 1.6% of players the model calls
impossible clear 4.5 targets. That is the baseball Tier-3 finding in
football clothing -- raw means with `fillna(0.0)` gave a debut hitter a home
run rate below anyone alive.

The fix is a role prior, and the depth chart is where role lives. Measured
on 493 first-ever games (2022-2024): position alone explains R^2 = 0.053 of
target share, position x depth rank explains **0.311**. Six times better,
and monotone in exactly the way football sense expects.

THE TRAP THIS MODULE HANDLES
----------------------------
nflverse changed the depth-chart schema after 2024, and the two versions do
not mean the same thing by "rank".

    <= 2024   columns: club_code, week, depth_position, depth_team
              depth_team is string depth WITHIN A SLOT, so a team fielding
              three wide receivers has THREE players at depth_team = 1
              (left, right, slot).

    >= 2025   columns: team, dt, pos_abb, pos_slot, pos_rank
              pos_rank orders players ACROSS the whole position group, so
              there is exactly ONE wide receiver at pos_rank = 1, one at 2,
              one at 3. Verified on PHI's 2026-09-10 snapshot: DeVonta
              Smith 1, Dontayvion Wicks 2, Makai Lemon 3, Hollywood Brown 4.

Pooling the two naively puts three starters and one starter in the same
bucket. `role_ranks()` normalises both to the new-schema meaning -- a dense
ordering within (team, week, position) -- and `era_consistency()` checks
that the resulting rank-to-share mapping actually agrees across the change
before anyone pools them.

The 2025+ schema is also keyed by an ISO8601 timestamp rather than a week,
so the week has to be recovered by asking which game the snapshot precedes.
That is done here, strictly: the row used for week w is the latest snapshot
STRICTLY BEFORE that game's kickoff, which is what would have been visible
at prediction time.
"""
from __future__ import annotations

from typing import Iterable, Optional

import nflreadpy as nfl
import polars as pl

from data.cache import DEFAULT_TTL_HOURS, load_cached, save_cache
from data.nflverse import load_rosters_weekly, load_schedules

# QB is here because passing_yards needs a depth-chart row to appear on a
# slate at all. Without it predict_slate produced 0 passing rows and said
# so only in a count nobody would read twice.
SKILL = ("QB", "WR", "TE", "RB", "FB")
NEW_SCHEMA_FIRST_SEASON = 2025


def _old_schema_ranks(season: int) -> pl.DataFrame:
    dc = nfl.load_depth_charts(seasons=[season])
    need = {"club_code", "week", "depth_position", "depth_team", "gsis_id"}
    if not need.issubset(set(dc.columns)):
        raise ValueError(f"{season}: expected pre-2025 schema, got {list(dc.columns)}")

    dc = (
        dc.filter(pl.col("formation") == "Offense")
          .filter(pl.col("depth_position").is_in(SKILL))
          .filter(pl.col("gsis_id").is_not_null())
          .select([
              pl.lit(season).alias("season"),
              pl.col("week").cast(pl.Int64),
              pl.col("club_code").alias("team"),
              pl.col("gsis_id").alias("player_id"),
              pl.col("depth_position").alias("role_pos"),
              pl.col("depth_team").cast(pl.Int64, strict=False).alias("_slot_depth"),
          ])
          .drop_nulls(["week", "_slot_depth"])
    )
    # Dense rank across the position group, ordering by within-slot depth.
    # Ties (the three depth_team=1 receivers) are broken arbitrarily but
    # deterministically -- which is the best this schema supports, and is
    # why era_consistency() exists.
    ranked = (
        dc.sort(["season", "week", "team", "role_pos", "_slot_depth", "player_id"])
          .with_columns(
              pl.col("_slot_depth").rank("ordinal")
                .over(["season", "week", "team", "role_pos"]).cast(pl.Int64).alias("role_rank")
          )
          .drop("_slot_depth")
    )
    # One team per player, as in the new schema. This era has no timestamp to
    # break the tie, so it keeps the shallower depth rank -- the roster where
    # he is used more, which is the better guess at where he actually is.
    return (
        ranked.sort(["season", "week", "player_id", "role_rank"])
              .group_by(["season", "week", "player_id"], maintain_order=True)
              .first()
              .select(["season", "week", "team", "player_id", "role_pos", "role_rank"])
    )


def _new_schema_ranks(season: int) -> pl.DataFrame:
    dc = nfl.load_depth_charts(seasons=[season])
    need = {"team", "dt", "pos_abb", "pos_rank", "gsis_id"}
    if not need.issubset(set(dc.columns)):
        raise ValueError(f"{season}: expected 2025+ schema, got {list(dc.columns)}")

    dc = (
        dc.filter(pl.col("pos_abb").is_in(SKILL))
          .filter(pl.col("gsis_id").is_not_null())
          .select([
              pl.col("team"),
              # Explicit format and UTC: the feed stamps "2026-09-10T12:01:46Z",
              # and polars refuses to infer a format when a zone is present.
              pl.col("dt").cast(pl.Utf8)
                .str.to_datetime("%Y-%m-%dT%H:%M:%SZ", time_zone="UTC", strict=False)
                .dt.replace_time_zone(None).alias("dt"),
              pl.col("gsis_id").alias("player_id"),
              pl.col("pos_abb").alias("role_pos"),
              pl.col("pos_rank").cast(pl.Int64, strict=False).alias("role_rank"),
          ])
          .drop_nulls(["dt", "role_rank"])
    )

    # Attach each snapshot to the game it precedes: latest snapshot strictly
    # before kickoff is what a predictor standing at that moment would see.
    sch = (
        load_schedules()
        .filter(pl.col("season") == season)
        .select([
            pl.col("season"), pl.col("week"), pl.col("gameday"), pl.col("gametime"),
            pl.col("home_team"), pl.col("away_team"),
        ])
    )
    kicks = pl.concat([
        sch.select(["season", "week", "gameday", "gametime",
                    pl.col("home_team").alias("team")]),
        sch.select(["season", "week", "gameday", "gametime",
                    pl.col("away_team").alias("team")]),
    ]).with_columns(
        (pl.col("gameday").cast(pl.Utf8) + " " + pl.col("gametime").fill_null("13:00"))
        .str.to_datetime("%Y-%m-%d %H:%M", strict=False).alias("kickoff")
    ).drop_nulls("kickoff").select(["season", "week", "team", "kickoff"])

    joined = (
        dc.join(kicks, on="team", how="inner")
          .filter(pl.col("dt") < pl.col("kickoff"))
    )
    # For each (week, team, player) keep the latest snapshot before kickoff.
    per_team = (
        joined.sort("dt")
              .group_by(["season", "week", "team", "player_id"], maintain_order=True)
              .agg([pl.col("role_pos").last(), pl.col("role_rank").last(),
                    pl.col("dt").last()])
    )
    # THEN collapse to ONE TEAM PER PLAYER, keeping the most recent chart.
    #
    # A traded or signed player appears on BOTH rosters: his old team's chart
    # still lists him and his new team's already does. Left in, the slate
    # predicted the same player twice with two different shares -- Kayshon
    # Boutte on HOU (share .074) and NE (.044) in 2026 week 1, and
    # Dontayvion Wicks on GB and PHI. He can only play for one of them, so
    # one of those rows is pure invention, and it also makes the name
    # ambiguous when joining book lines.
    #
    # The newer chart wins, because a move is published by the team he joined.
    return (
        per_team.sort("dt")
                .group_by(["season", "week", "player_id"], maintain_order=True)
                .last()
                .select(["season", "week", "team", "player_id", "role_pos", "role_rank"])
    )


def with_player_names(ranks: pl.DataFrame) -> pl.DataFrame:
    """
    Attach display names. Depth charts key on gsis_id, and a slate showing
    `00-0032764` instead of a player name is unreadable -- which is how the
    first live slate came out.
    """
    if ranks.is_empty():
        return ranks
    try:
        players = nfl.load_players().select([
            pl.col("gsis_id").alias("player_id"),
            pl.col("display_name").alias("player_name"),
        ]).drop_nulls("player_id").unique(subset=["player_id"])
    except Exception:
        return ranks.with_columns(pl.col("player_id").alias("player_name"))
    return ranks.join(players, on="player_id", how="left").with_columns(
        pl.col("player_name").fill_null(pl.col("player_id")))


def role_ranks(seasons: Iterable[int], through_week: Optional[int] = None,
               current_season: int = 2026) -> pl.DataFrame:
    """
    One row per (season, week, team, player_id) with role_pos and role_rank,
    normalised to the 2025+ meaning: a dense ordering within the position
    group, so role_rank 1 is the team's top man at that position.

    Depth charts update daily at 07:00 UTC and are a SERVE-TIME feed -- the
    chart for the upcoming game is legitimately available, unlike snap counts
    or participation. `through_week` is therefore inclusive.
    """
    frames = []
    for season in seasons:
        key = f"depth_ranks_{season}"
        ttl = None if season < current_season else DEFAULT_TTL_HOURS
        df = load_cached(key, ttl_hours=ttl)
        if df is None:
            build = _new_schema_ranks if season >= NEW_SCHEMA_FIRST_SEASON else _old_schema_ranks
            df = save_cache(key, build(season))
        frames.append(df)
    if not frames:
        return pl.DataFrame()
    out = pl.concat(frames, how="diagonal_relaxed")
    if through_week is not None:
        out = out.filter(pl.col("week") <= through_week)
    return out


MAX_ROLE_BUCKET = 7


def role_bucket(role_rank: pl.Expr) -> pl.Expr:
    """
    Collapse depth rank to the buckets the role prior is estimated on.

    THE CAP MATTERS AND USED TO BE 4. That was wrong, and it produced the
    largest error this project has had. Measured share of team targets by
    depth rank, 2025:

        WR  r1 .381  r2 .273  r3 .177  r4 .097  r5 .047  r6 .020  r7 .004
        TE  r1 .641  r2 .241  r3 .093  r4 .022  r5 .004
        RB  r1 .582  r2 .298  r3 .108  r4 .010  r5 .003

    The curve does NOT flatten after 4th -- it keeps falling by roughly half
    a step each rank. Collapsing everything from 4th down into one bucket
    handed every WR5, WR6, WR7 and WR8 the WR4 prior of .097, so four deep
    reserves soaked up ~39% of a team's targets and every real contributor
    was diluted by that much.

    That was invisible in backtest, because the backtest's roster came from
    the stats table -- players who actually played -- while the live slate's
    comes from the depth chart, which lists everyone. Same train/serve
    asymmetry that broke `_project_pa` in baseball, arriving through the
    roster rather than through a feature.

    Symptom: the model projected Ja'Marr Chase for 6.4 targets and 51 yards
    against a market line implying ~9 and ~75, and the mean edge across 596
    priced props was **-0.29**.

    Seven buckets, because at r7 the share is .004 and genuinely is
    indistinguishable from zero.
    """
    expr = pl.when(role_rank <= 1).then(pl.lit(1))
    for r in range(2, MAX_ROLE_BUCKET):
        expr = expr.when(role_rank == r).then(pl.lit(r))
    return expr.otherwise(pl.lit(MAX_ROLE_BUCKET))


def era_consistency(shares: pl.DataFrame) -> pl.DataFrame:
    """
    Mean target share by (role_pos, role_bucket), split at the schema change.

    Run this before pooling eras. If the two halves disagree materially, the
    normalisation in role_ranks() is not capturing the same thing on both
    sides and the prior must be fitted on 2025+ only.

    `shares` needs columns: season, role_pos, role_rank, share.
    """
    return (
        shares.with_columns([
            role_bucket(pl.col("role_rank")).alias("bucket"),
            pl.when(pl.col("season") >= NEW_SCHEMA_FIRST_SEASON)
              .then(pl.lit("2025+")).otherwise(pl.lit("<=2024")).alias("era"),
        ])
        .group_by(["role_pos", "bucket", "era"])
        .agg([pl.col("share").mean().alias("mean_share"), pl.len().alias("n")])
        .sort(["role_pos", "bucket", "era"])
    )


def prior_week_snaps(seasons, through_week=None) -> pl.DataFrame:
    """
    Each player's offensive snap share from his team's PREVIOUS game.

    Returns season, week, team, player_id, snap_prev -- where `week` is the
    week the number may be USED for, not the week it was earned. Week 1 has
    no row, by construction.

    WHY LAGGED, AND WHY THAT IS NOT A LIMITATION. `config.SERVE_TIME_FEEDS`
    lists snap_counts as "week N-1": the feed lands after games, so the only
    snap number legitimately visible before kickoff is the previous game's.
    Lagging here is what makes it servable, not a concession.

    The feed is keyed on `pfr_player_id`, not gsis. The crosswalk comes from
    rosters_weekly, which carries both. Rows that fail to cross over are
    dropped rather than guessed -- a wrong join here would move share to the
    wrong player, which is the failure mode this whole module exists to
    avoid.
    """
    from data.nflverse import load_snap_counts

    seasons = list(seasons)
    xw = []
    for s in seasons:
        try:
            r = load_rosters_weekly([s]).select(["pfr_id", "gsis_id"])
            xw.append(r.drop_nulls())
        except Exception:  # noqa: BLE001
            continue
    if not xw:
        return pl.DataFrame(schema={"season": pl.Int64, "week": pl.Int64,
                                    "team": pl.Utf8, "player_id": pl.Utf8,
                                    "snap_prev": pl.Float64})
    xw = pl.concat(xw, how="diagonal_relaxed").unique(subset=["pfr_id"])

    sn = load_snap_counts(seasons, through_week=through_week)
    sn = (sn.select(["season", "week", "team", "pfr_player_id", "offense_pct"])
            .with_columns([pl.col("season").cast(pl.Int64),
                           pl.col("week").cast(pl.Int64)])
            .drop_nulls("offense_pct")
            .join(xw, left_on="pfr_player_id", right_on="pfr_id", how="inner")
            .rename({"gsis_id": "player_id"})
            .select(["season", "week", "team", "player_id", "offense_pct"]))

    # Shift to the week the number is usable in. Within a season and team,
    # a player's next appearance is the next game he is listed for, so a bye
    # carries the last real number forward rather than blanking it.
    sn = sn.sort(["player_id", "season", "week"])
    sn = sn.with_columns((pl.col("week") + 1).alias("week_usable"))
    return (sn.select([pl.col("season"), pl.col("week_usable").alias("week"),
                       pl.col("team"), pl.col("player_id"),
                       pl.col("offense_pct").alias("snap_prev")])
              .unique(subset=["season", "week", "team", "player_id"]))


def apply_snap_ranks(rr: pl.DataFrame, snaps: pl.DataFrame) -> pl.DataFrame:
    """
    Re-rank each position group by PRIOR-WEEK SNAP SHARE where it exists,
    keeping the chart's ordering for anyone without a snap number.

    WHY RANK AND NOT USE THE PERCENTAGE DIRECTLY. Shares are renormalised
    per team, so any common rescaling of the rates is divided straight back
    out -- that is what killed SHARE_PRIOR_K. Only a player's standing
    RELATIVE to his teammates survives normalisation, so the signal has to
    enter as an ordering. It also lets snaps reuse the role-prior machinery
    already validated rather than introducing a second, unvalidated path
    from a number to a share.

    WHY THIS IS NOT THE REFUTED PRODUCTION RE-RANK. Ranking by prior-season
    PRODUCTION measured WORSE than the chart (RMSE 2.7431 vs 2.6974).
    Ranking by prior-week SNAPS measures BETTER, on identical rows:

        chart prior          RMSE 2.7336   bias +0.1344
        snap-ranked prior    RMSE 2.6536   bias -0.0297

    Production is an outcome and carries every confound that goes with it.
    A snap count is a direct observation of whether the coaching staff put
    the player on the field last Sunday, which is the question the depth
    chart is a stale proxy for.

    Players with a snap number always sort above players without one: an
    unmeasured player did not take the field, and the chart's opinion about
    him is the weaker evidence.
    """
    need = {"season", "week", "team", "player_id", "role_pos", "role_rank"}
    missing = need - set(rr.columns)
    if missing:
        raise ValueError(f"rr missing columns: {sorted(missing)}")
    if snaps.is_empty():
        return rr

    out = rr.join(snaps, on=["season", "week", "team", "player_id"], how="left")
    out = out.with_columns(
        pl.when(pl.col("snap_prev").is_not_null())
          .then(pl.lit(0)).otherwise(pl.lit(1)).alias("_tier"))
    out = out.with_columns(
        pl.struct(["_tier", "snap_prev", "role_rank"]).alias("_k"))
    out = out.with_columns(
        pl.col("snap_prev").fill_null(-1.0).alias("_snap"))
    out = (out.sort(["season", "week", "team", "role_pos",
                     "_tier", "_snap", "role_rank"],
                    descending=[False, False, False, False,
                                False, True, False])
              .with_columns(
                  (pl.int_range(pl.len()).over(
                      ["season", "week", "team", "role_pos"]) + 1)
                  .cast(pl.Int64).alias("role_rank")))
    return out.drop(["_tier", "_snap", "_k", "snap_prev"])
