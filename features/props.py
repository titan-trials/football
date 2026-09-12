"""
The prop registry: one declaration per market, so every prop runs through
the same Stage 1 x Stage 2 machinery and is scored the same way.

WHY A REGISTRY RATHER THAN FOUR SCRIPTS
---------------------------------------
`model-review-2026-09-05.md`'s Tier-1 finding was that the deployed pitcher
strikeout prop compounded a rate that had never been backtested, because
the backtest walked a different code path and no one could compare them.
Four hand-written prop scripts is that failure waiting to happen four
times.

Everything that differs between props is DATA in this file -- which
opportunity, which outcome, which positions, which shape parameter, what
support. Everything that is shared is CODE in usage.py and efficiency.py.
So a fix to the compound reaches every prop, and a comparison between props
is a comparison of the same model, not of four cousins.

THE SHAPE PARAMETER IS ON FOR EXACTLY TWO PROPS, AND THAT WAS MEASURED
----------------------------------------------------------------------
End-to-end CRPS gain from conditioning Stage 2 on a shape parameter:

    receptions       aDOT      +0.0133   ON
    receiving_yards  aDOT      +0.0097   ON
    passing_yards    aDOT      +0.0000   off
    rushing_tds      shotgun   -0.0002   off
    receiving_tds    aDOT      -0.0008   off
    rushing_yards    shotgun   -0.0009   off

The rushing result is the interesting one, because shotgun share looks like
a find and is not.

It is reliable -- split-half **0.824**, second only to aDOT's 0.885 -- and
it genuinely reshapes the per-carry distribution: YPC 4.37 -> 5.20 and
P(>=10 yards) 0.106 -> 0.160 across bands. Three other candidates were
tested and were worse: short-yardage share 0.487 (and NEGATIVELY correlated
with future YPC, -0.054), yardline 0.369, goal-line share 0.163 (noise).

So shotgun share was the best available structural parameter for rushing by
a wide margin, and it still buys nothing.

**The test that separates them is predictive, not descriptive.** aDOT BEATS
past efficiency at predicting future efficiency (0.343 vs 0.255). Shotgun
share LOSES to it (0.254 vs 0.367). A parameter that reshapes the
distribution but predicts the mean worse than what you already have will
shift the mean in the wrong direction often enough to cost more than the
shape gains -- especially here, where the convolution washes out shape and
leaves only the mean.

That is the baseball "granular features are not additive when they overlap"
finding arriving from a new direction: the overlap is not with another
feature, it is with the trailing rate the model is already estimating.

**The rule this leaves:** reliability and a visible effect on the
distribution are NOT sufficient evidence to carry a shape parameter. It
must beat the incumbent estimator of the mean. Measure that first; it is
one correlation and it would have predicted every row of the table above.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import polars as pl


@dataclass(frozen=True)
class PropSpec:
    """One market. Everything that makes it different from the others."""
    name: str
    # Stage 1
    opportunity: str            # play-level event being counted
    positions: tuple            # positions eligible for this prop
    stat_opportunity: str       # opportunity column in player_stats
    stat_outcome: str           # outcome column in player_stats
    # Stage 2
    shape: Optional[str]        # per-play shape column, or None to pool
    shape_edges: Optional[np.ndarray]   # band edges for the shape parameter
    shape_centres: Optional[np.ndarray] # band centres, for interpolation
    pt_support: np.ndarray      # per-opportunity outcome support
    total_support: np.ndarray   # game-total support
    max_opportunities: int
    lines: tuple                # book-style lines to score at

    @property
    def has_shape(self) -> bool:
        return self.shape is not None


_YDS_PT = np.arange(-25, 106)
_RUSH_PT = np.arange(-25, 101)
_REC_PT = np.arange(0, 2)

# aDOT bands. Finer low-end bands were tested (select 2024, verify 2025) and
# moved CRPS by 0.0004: noise. Left as they are.
ADOT_EDGES = np.array([-np.inf, 5.0, 8.0, 11.0, 14.0, np.inf])
ADOT_CENTRES = np.array([3.0, 6.5, 9.5, 12.5, 16.0])

# Shotgun-share bands for rushing. Measured 2022-2025 on 58,807 carries:
# split-half reliability 0.824, second only to aDOT's 0.885, and it reshapes
# the per-carry distribution -- YPC 4.37 -> 5.20 and P(>=10 yards)
# 0.106 -> 0.160 from the lowest band to the highest.
#
# Note the difference from receiving. aDOT BEATS past efficiency at
# predicting future efficiency (0.343 vs 0.255); shotgun share does not
# (0.254 vs 0.367). It is complementary rather than superior -- together
# R^2 0.162, both coefficients positive. So it is worth conditioning on,
# but it is not the same kind of find aDOT was.
#
# Short-yardage share (0.487) and goal-line share (0.163) were tested and
# dropped: goal-line is noise, and short-yardage barely moves the
# distribution (YPC 4.47 -> 4.11) and is NEGATIVELY correlated with future
# YPC (-0.054).
SHOTGUN_EDGES = np.array([-np.inf, 0.35, 0.50, 0.65, 0.80, np.inf])
SHOTGUN_CENTRES = np.array([0.20, 0.42, 0.57, 0.72, 0.90])

PROPS = {
    "receiving_yards": PropSpec(
        name="receiving_yards",
        opportunity="target", positions=("WR", "TE", "RB"),
        stat_opportunity="targets", stat_outcome="receiving_yards",
        shape="air_yards", shape_edges=ADOT_EDGES, shape_centres=ADOT_CENTRES,
        pt_support=_YDS_PT, total_support=np.arange(-40, 501),
        max_opportunities=25, lines=(24.5, 39.5, 54.5, 69.5),
    ),
    "receptions": PropSpec(
        name="receptions",
        opportunity="target", positions=("WR", "TE", "RB"),
        stat_opportunity="targets", stat_outcome="receptions",
        shape="air_yards", shape_edges=ADOT_EDGES, shape_centres=ADOT_CENTRES,
        pt_support=_REC_PT, total_support=np.arange(0, 31),
        max_opportunities=25, lines=(1.5, 2.5, 3.5, 4.5, 5.5),
    ),
    "rushing_yards": PropSpec(
        name="rushing_yards",
        opportunity="carry", positions=("RB", "WR", "QB", "FB"),
        stat_opportunity="carries", stat_outcome="rushing_yards",
        shape=None, shape_edges=None, shape_centres=None,   # measured -0.0009, see below
        pt_support=_RUSH_PT, total_support=np.arange(-40, 401),
        max_opportunities=40, lines=(19.5, 39.5, 59.5, 79.5),
    ),
    "passing_yards": PropSpec(
        name="passing_yards",
        opportunity="attempt", positions=("QB",),
        stat_opportunity="attempts", stat_outcome="passing_yards",
        shape=None, shape_edges=None, shape_centres=None,   # measured +0.0000
        pt_support=_YDS_PT, total_support=np.arange(-40, 701),
        max_opportunities=60, lines=(174.5, 224.5, 264.5, 299.5),
    ),
    # --- touchdown props ------------------------------------------------
    # A touchdown prop is a COUNTING prop, not a special case: the
    # per-opportunity outcome is 0/1, the total is the number of scores, and
    # "anytime" is simply over 0.5. No new machinery.
    #
    # These two are modelled separately and combined at slate level, because
    # a player can score either way and the opportunity counts differ:
    #     P(anytime) = 1 - P(0 receiving TD) x P(0 rushing TD)
    # See `anytime_td_probability()` below for the independence caveat.
    "receiving_tds": PropSpec(
        name="receiving_tds",
        opportunity="target", positions=("WR", "TE", "RB"),
        stat_opportunity="targets", stat_outcome="receiving_tds",
        shape=None, shape_edges=None, shape_centres=None,   # measured -0.0008
        # 0..6. The single-game receiving-TD record is 5; the support is set
        # one clear of it, per the MAX_RUNS rule that a cap tuned to the cases
        # you tested is a cap that fails on the case you did not.
        pt_support=np.arange(0, 2), total_support=np.arange(0, 7),
        max_opportunities=25, lines=(0.5, 1.5),
    ),
    "rushing_tds": PropSpec(
        name="rushing_tds",
        opportunity="carry", positions=("RB", "WR", "QB", "FB"),
        stat_opportunity="carries", stat_outcome="rushing_tds",
        shape=None, shape_edges=None, shape_centres=None,   # measured -0.0002
        # 0..6. Six rushing touchdowns in one game has actually happened.
        pt_support=np.arange(0, 2), total_support=np.arange(0, 7),
        max_opportunities=40, lines=(0.5, 1.5),
    ),
}


def anytime_td_probability(p_no_rec_td: float, p_no_rush_td: float) -> float:
    """
    P(scores at least one touchdown, any way).

        P(anytime) = 1 - P(no receiving TD) x P(no rushing TD)

    The multiplication assumes the two are independent GIVEN the player's
    opportunity counts. They are not quite: a goal-line back gets both
    carries and targets inside the 5, so the two are positively correlated
    and this slightly UNDERSTATES P(anytime). The error is small for players
    who do one thing and largest exactly for the goal-line backs the market
    prices most sharply -- so treat this as a first pass and check it
    against the market before trusting it on that population.
    """
    return 1.0 - p_no_rec_td * p_no_rush_td


# ---------------------------------------------------------------------
# Play-level extraction
# ---------------------------------------------------------------------

def opportunity_rows(spec: PropSpec, pbp: pl.DataFrame) -> pl.DataFrame:
    """
    One row per opportunity: season, week, team, player_id, outcome, shape.

    `outcome` is the yards (or 0/1 completion) on that single play, with
    failures carried as zeros so any hurdle lives inside the distribution.
    """
    if spec.opportunity == "target":
        d = pbp.filter((pl.col("pass_attempt") == 1)
                       & pl.col("receiver_player_id").is_not_null())
        if spec.name == "receptions":
            out = pl.col("complete_pass")
        elif spec.name == "receiving_tds":
            out = (pl.col("pass_touchdown").fill_null(0) == 1).cast(pl.Float64)
        else:
            out = pl.col("yards_gained")
        player = "receiver_player_id"
    elif spec.opportunity == "carry":
        d = pbp.filter((pl.col("rush_attempt") == 1)
                       & pl.col("rusher_player_id").is_not_null())
        out = ((pl.col("rush_touchdown").fill_null(0) == 1).cast(pl.Float64)
               if spec.name == "rushing_tds" else pl.col("yards_gained"))
        player = "rusher_player_id"
    elif spec.opportunity == "attempt":
        # SACKS ARE NOT PASS ATTEMPTS. nflfastR sets pass_attempt = 1 on a
        # sack; the official stat does not count it, and its negative yards
        # are charged to team rushing, not passing.
        #
        # Caught by comparing props side by side (2025): pbp gave 18,741
        # attempts and 114,144 yards against an official 17,412 and 122,227
        # -- 1,329 EXTRA attempts, which is the season's sack count, and
        # 8,083 FEWER yards, which is 1,329 x -6.1. Left in, it biased
        # passing yards by -21 per game while every other prop looked fine.
        #
        # qb_spike and qb_kneel are excluded for the same reason: they are
        # clock plays, not attempts.
        d = pbp.filter((pl.col("pass_attempt") == 1)
                       & (pl.col("sack") == 0)
                       & (pl.col("play_type") == "pass")
                       & pl.col("passer_player_id").is_not_null())
        player, out = "passer_player_id", pl.col("yards_gained")
    else:
        raise ValueError(f"unknown opportunity {spec.opportunity!r}")

    cols = [
        pl.col("season"), pl.col("week"), pl.col("posteam").alias("team"),
        pl.col(player).alias("player_id"),
        out.fill_null(0.0).cast(pl.Float64).alias("outcome"),
    ]
    if spec.has_shape:
        cols.append(pl.col(spec.shape).cast(pl.Float64).alias("shape_raw"))
    return d.select(cols)


def shape_per_game(spec: PropSpec, rows: pl.DataFrame) -> pl.DataFrame:
    """
    Per player-game sum and count of the shape parameter, for ShapeModel,
    plus the as-of (expanding, shifted) value for labelling training rows.

    The shift is the whole point: labelling a training row with a shape
    parameter computed including that game is the train/serve skew that
    broke `_project_pa` in baseball, and here it would flatter the bands
    badly because hindsight aDOT separates the outcomes it is meant to
    predict.
    """
    if not spec.has_shape:
        return pl.DataFrame()
    g = (
        rows.group_by(["season", "week", "team", "player_id"])
            .agg([pl.col("shape_raw").sum().alias("shape_sum"),
                  pl.col("shape_raw").is_not_null().sum().alias("shape_n")])
            .sort(["player_id", "season", "week"])
    )
    return g.with_columns([
        pl.col("shape_sum").cum_sum().over("player_id").shift(1).over("player_id").alias("_s"),
        pl.col("shape_n").cum_sum().over("player_id").shift(1).over("player_id").alias("_n"),
    ]).with_columns(
        pl.when(pl.col("_n") > 0).then(pl.col("_s") / pl.col("_n"))
          .otherwise(None).alias("asof_shape")
    ).drop(["_s", "_n"])


def label_rows_with_asof_shape(rows: pl.DataFrame, per_game: pl.DataFrame) -> pl.DataFrame:
    """Attach each opportunity row the player's shape as known BEFORE that game."""
    if per_game.is_empty():
        return rows
    return rows.join(
        per_game.select(["season", "week", "team", "player_id", "asof_shape"]),
        on=["season", "week", "team", "player_id"], how="left",
    )


def player_games(spec: PropSpec, player_stats: pl.DataFrame) -> pl.DataFrame:
    """Per player-game opportunities and the realised prop outcome."""
    return (
        player_stats
        .filter(pl.col("position").is_in(list(spec.positions)))
        .select(["season", "week", "team", "player_id", "position",
                 spec.stat_opportunity, spec.stat_outcome])
        .with_columns([
            pl.col(spec.stat_opportunity).fill_null(0).cast(pl.Float64).alias("opportunities"),
            pl.col(spec.stat_outcome).fill_null(0).cast(pl.Float64).alias("outcome"),
        ])
        .drop([spec.stat_opportunity, spec.stat_outcome])
    )


ACTIVE_STATUS = "ACT"


def roster_player_games(spec: PropSpec, player_stats: pl.DataFrame,
                        ranks: pl.DataFrame,
                        statuses: Optional[pl.DataFrame] = None) -> pl.DataFrame:
    """
    Player-games built from the DEPTH CHART, with zeros filled in.

    WHY THIS REPLACES `player_games` FOR FITTING
    --------------------------------------------
    `player_games` is built from the weekly stats table, so it contains only
    players who recorded something. The live slate's roster comes from the
    depth chart, which lists everyone. Those are different populations, and
    fitting on one while serving the other is the same train/serve asymmetry
    that broke `_project_pa` in baseball -- arriving through the ROSTER this
    time rather than through a feature.

    Two things go wrong when they differ, and both were measured:

    1. **Deep role priors cannot be estimated.** The prior for (WR, rank 6)
       has to come from WR6s, and a WR6 usually records nothing, so he has
       no row. The cell falls below the 5-player floor, falls back to the
       pooled WR rate, and every deep reserve is handed a WR-average share.
       Cincinnati's slate had Jordan Moore, Xavier Johnson and Noah Thomas
       at identical .0324 shares -- the signature of three different players
       collapsing onto one fallback.

    2. **The real contributors are diluted by the difference.** Fourteen
       no-history players soaking ~.03 each is ~35% of a team's targets
       going to players who will get none. Measured against reality:

           actual 2025 CIN   Chase .305  Higgins .161  Brown .145  (top3 .61)
           model             Chase .183  Higgins .114  Brown .091  (top3 .39)

       which produced a mean edge of -0.29 against 596 priced props.

    Filling zeros fixes both: a WR6 who was listed and caught nothing is
    evidence that WR6s catch nothing, and it is the *only* evidence of that.
    """
    need = {"season", "week", "team", "player_id"}
    missing = need - set(ranks.columns)
    if missing:
        raise ValueError(f"ranks missing columns: {sorted(missing)}")

    pos_col = "role_pos" if "role_pos" in ranks.columns else "position"
    keep = ["season", "week", "team", "player_id", pl.col(pos_col).alias("position"),
            pl.col(pos_col).alias("role_pos")]
    if "role_bucket" in ranks.columns:
        # The role the player held IN THAT GAME, carried through so priors can
        # be estimated contemporaneously. Joining a player's CURRENT rank to
        # his HISTORICAL production is how the WR7 prior came out at 1.18
        # targets a game: a WR7 today who was a WR2 two years ago contributed
        # WR2 volume to the WR7 cell, and every no-history deep reserve was
        # then handed it.
        keep.append("role_bucket")
    roster = (ranks.filter(pl.col(pos_col).is_in(list(spec.positions)))
                   .select(keep)
                   .unique(subset=["season", "week", "team", "player_id"]))

    stats = player_games(spec, player_stats).select(
        ["season", "week", "team", "player_id", "opportunities", "outcome"])

    # ONLY TEAM-WEEKS THAT HAVE ACTUALLY BEEN PLAYED.
    #
    # This line is load-bearing and its absence was the worst bug in the
    # project. The 2025+ depth-chart schema is keyed by timestamp, and a
    # snapshot is attached to every week whose kickoff follows it -- so a
    # September chart produces a row for weeks 1 through 18. Zero-filling
    # against that invents a row for every unplayed game.
    #
    # ShareModel then takes the LAST 8 rows per player. For every player in
    # the 2026 slate those were weeks 11-18 of a season that has played one
    # week: eight zeros, all from the future. Every share collapsed to the
    # role prior, Sam LaPorta -- a TE1 with 120 targets in 2023 -- came out
    # at .0063 and 0.2 expected receptions, and the mean edge against 596
    # priced props was -0.17.
    #
    # It is invisible in backtest because a backtest only ever asks for
    # weeks that have already happened.
    played = stats.select(["season", "week", "team"]).unique()
    out = (roster.join(played, on=["season", "week", "team"], how="inner")
                 .join(stats, on=["season", "week", "team", "player_id"], how="left")
                 .with_columns([pl.col("opportunities").fill_null(0.0),
                                pl.col("outcome").fill_null(0.0)]))

    # ONLY WEEKS THE PLAYER WAS ACTIVE.
    #
    # Zero-filling is how a WR6 teaches the model that WR6s get nothing. It
    # is NOT how an injured starter should be treated: he did not fail to
    # get targets, he was absent. Sam LaPorta went on injured reserve for
    # weeks 11-18 of 2025, so a trailing-8 window over those weeks was eight
    # zeros and put a TE1 with 120 career targets at a .0068 share and 0.2
    # expected receptions.
    #
    # `rosters_weekly.status` separates them: ACT is active, RES/INA/DEV/CUT
    # are not there. A row with no status is kept -- absence of the roster
    # feed should not silently delete a player.
    if statuses is not None and not statuses.is_empty():
        st = (statuses.select(["season", "week", "team", "gsis_id", "status"])
                      .rename({"gsis_id": "player_id"})
                      .drop_nulls("player_id")
                      .unique(subset=["season", "week", "team", "player_id"],
                              keep="last"))
        out = (out.join(st, on=["season", "week", "team", "player_id"], how="left")
                  .filter(pl.col("status").is_null()
                          | (pl.col("status") == ACTIVE_STATUS))
                  .drop("status"))
    return out


def team_games(pg: pl.DataFrame) -> pl.DataFrame:
    """Per team-game total opportunities -- Stage 1's N."""
    return pg.group_by(["season", "week", "team"]).agg(
        pl.col("opportunities").sum().alias("N"))
