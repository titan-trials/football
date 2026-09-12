"""
Availability: the injury report as a LOGGED FLAG, fed to nothing.

THE RULE THIS FOLLOWS
---------------------
From `context-features-availability.md`: add the flag as a column, show it,
feed it to nothing. Because slates are written before games and scored
after, the column tests itself -- after ~20 weeks, check whether flagged
players fell short of their predicted numbers. If they did, wire it in. If
not, delete the column. Nothing was ever at risk.

That is the discipline this project keeps getting rewarded for. Four of the
last five things that looked like finds (shotgun share, finer aDOT bands,
harder share shrinkage, the market in team volume) measured at or below
zero. A flag costs nothing to be wrong about.

WHAT THE EVIDENCE ALREADY SAYS
------------------------------
Suggestive but far from established, from the Stage 1 top-bucket
investigation on 2025:

    no report listed    n=776   pred 0.774   actual 0.705   gap -0.069
    Questionable        n= 47   pred 0.808   actual 0.660   gap -0.149

Flagged players miss badly, and even "Full Participation in Practice" rows
ran -0.138 -- appearing on the report at all is a negative signal, not just
being limited by it. But they were **5.7%** of the bucket, so removing them
moved the aggregate gap from -0.073 to only -0.069. Real per-player,
negligible in aggregate, n = 47.

WHY THIS IS A SERVE-TIME FEED AND SNAP COUNTS ARE NOT
-----------------------------------------------------
The injury report is published before kickoff -- Wednesday/Thursday/Friday
practice participation and a game-status designation. It is one of the very
few feeds that describes the game about to be played rather than the last
one. Snap counts and participation describe what already happened.

A note the nflverse docs get wrong: they state the injury source ended
after 2024 with no 2025 data. Verified false -- 2025 returns 6,068 rows and
2026 week 1 returns 139.
"""
from __future__ import annotations

from typing import Iterable, Optional

import polars as pl

from data.nflverse import load_injuries

# Ordered worst to best. `status_rank` makes these comparable.
GAME_STATUS = ("Out", "Doubtful", "Questionable", "None")
PRACTICE_STATUS = (
    "Did Not Participate In Practice",
    "Limited Participation in Practice",
    "Full Participation in Practice",
)


def flag_frame(seasons: Iterable[int], through_week: Optional[int] = None,
               current_season: int = 2026) -> pl.DataFrame:
    """
    One row per (season, week, team, player_id) with the availability flag.

    Columns:
        report_status    game designation, or "None" if not listed
        practice_status  practice participation, or "None"
        availability     compact flag: OUT / DOUBTFUL / QUESTIONABLE /
                         LISTED / CLEAR
        status_rank      0 = out, 4 = clear. Ordinal, for sorting and for
                         the eventual self-test.

    `through_week` is INCLUSIVE here, unlike every other feed: Wednesday's
    report for Sunday's game is legitimately available at predict time.
    """
    inj = load_injuries(seasons, through_week=through_week,
                        current_season=current_season)
    if inj.is_empty():
        return pl.DataFrame()

    inj = (
        inj.select(["season", "week", "team", "gsis_id",
                    "report_status", "practice_status"])
           .rename({"gsis_id": "player_id"})
           .filter(pl.col("player_id").is_not_null())
           .unique(subset=["season", "week", "team", "player_id"], keep="last")
           .with_columns([
               pl.col("report_status").fill_null("None"),
               pl.col("practice_status").fill_null("None"),
           ])
    )

    return inj.with_columns([
        pl.when(pl.col("report_status") == "Out").then(pl.lit("OUT"))
          .when(pl.col("report_status") == "Doubtful").then(pl.lit("DOUBTFUL"))
          .when(pl.col("report_status") == "Questionable").then(pl.lit("QUESTIONABLE"))
          .otherwise(pl.lit("LISTED")).alias("availability"),
        pl.when(pl.col("report_status") == "Out").then(pl.lit(0))
          .when(pl.col("report_status") == "Doubtful").then(pl.lit(1))
          .when(pl.col("report_status") == "Questionable").then(pl.lit(2))
          .otherwise(pl.lit(3)).alias("status_rank"),
    ])


def report_published(flags: pl.DataFrame, season: int, week: int) -> bool:
    """
    Has the injury report for this week actually been published?

    THIS MATTERS MORE THAN IT LOOKS. Without it, "no players flagged" is
    indistinguishable from "every player is healthy", and the first live
    slate reported 0 flagged rows for a week whose report simply did not
    exist yet. A silent absence that reads as a confident negative is the
    same failure class as the calibration bug that printed nothing because
    it was looking for the wrong column name.
    """
    if flags.is_empty():
        return False
    return flags.filter((pl.col("season") == season)
                        & (pl.col("week") == week)).height > 0


def attach(slate: pl.DataFrame, flags: pl.DataFrame) -> pl.DataFrame:
    """
    Left-join the flag onto a slate. Players absent from the report are
    CLEAR with rank 4 -- above "listed but no designation", which the
    evidence above says is itself a mild negative.

    Call `report_published` first and say so if it is False. Do NOT fall
    back to a previous week's report: a week-1 "Out" is not evidence about
    week 2, and quietly carrying it forward would be worse than having no
    flag at all.
    """
    if flags.is_empty():
        return slate.with_columns([
            pl.lit("CLEAR").alias("availability"),
            pl.lit(4).alias("status_rank"),
            pl.lit("None").alias("report_status"),
            pl.lit("None").alias("practice_status"),
        ])
    keys = [k for k in ("season", "week", "team", "player_id") if k in slate.columns]
    out = slate.join(flags, on=keys, how="left")
    return out.with_columns([
        pl.col("availability").fill_null("CLEAR"),
        pl.col("status_rank").fill_null(4),
        pl.col("report_status").fill_null("None"),
        pl.col("practice_status").fill_null("None"),
    ])


def self_test(scored: pl.DataFrame, pred_col: str = "exp",
              actual_col: str = "actual") -> pl.DataFrame:
    """
    THE POINT OF THE FLAG. Once ~20 weeks of scored slates exist, call this:
    it reports, per availability bucket, how far actual outcomes fell short
    of predicted ones.

    If flagged players systematically underperform their predictions, the
    flag has earned its way into the model. If they do not, delete the
    column. Either way nothing was ever at risk, which is the whole reason
    to do it this way round.

    `residual` is actual minus predicted, so negative means the model was
    too high for that bucket.
    """
    need = {"availability", pred_col, actual_col}
    missing = need - set(scored.columns)
    if missing:
        raise ValueError(f"scored frame missing: {sorted(missing)}")

    return (
        scored.with_columns((pl.col(actual_col) - pl.col(pred_col)).alias("residual"))
              .group_by("availability")
              .agg([
                  pl.len().alias("n"),
                  pl.col("residual").mean().alias("mean_residual"),
                  (pl.col("residual").std() / pl.len().sqrt()).alias("se"),
                  pl.col(pred_col).mean().alias("mean_pred"),
                  pl.col(actual_col).mean().alias("mean_actual"),
              ])
              .sort("n", descending=True)
    )
