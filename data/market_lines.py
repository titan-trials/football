"""
Closing lines from nflverse schedules -- the benchmark, free, back to 1999.

From context-features-availability.md: "Recording closing lines is not a
feature -- it is the only way to answer 'is this model good' rather than
'is it better than nothing.'" Baseball had to build that from scratch and
pay Odds API credits for it. Football gets 27 seasons of it in one call.

The implied probabilities here are vig-free. A raw moneyline pair sums to
more than 1; comparing a model to the raw number credits the model with
beating the hold, which it did not do.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from data.nflverse import load_schedules


def american_to_prob(odds) -> np.ndarray:
    """
    American odds -> implied probability, with vig still in it.

    Computed branch-wise rather than with np.where, because np.where
    evaluates BOTH branches: at odds of exactly -100 the positive branch
    divides by zero, emitting a RuntimeWarning and an inf that np.where
    then discards. The value was right and the warning was noise, which is
    the kind of thing that trains you to ignore warnings.
    """
    o = np.asarray(odds, dtype=float)
    out = np.full(o.shape, np.nan, dtype=float)
    neg = o < 0
    pos = ~neg & np.isfinite(o)
    out[neg] = -o[neg] / (-o[neg] + 100.0)
    out[pos] = 100.0 / (o[pos] + 100.0)
    return out


def devig_pair(p_home, p_away) -> tuple:
    """
    Proportional (multiplicative) de-vig. The two implied probabilities are
    scaled to sum to 1.

    This is the simplest of several defensible methods and it is slightly
    wrong in a known direction -- it assumes the hold is spread evenly,
    while books load more of it on the longshot. Shin or power de-vig
    correct for that. At NFL moneyline holds (~4%) the difference is under
    a point, and using one method consistently matters more than which.
    Recorded here so the choice is visible rather than implicit.
    """
    p_home = np.asarray(p_home, dtype=float)
    p_away = np.asarray(p_away, dtype=float)
    total = p_home + p_away
    return p_home / total, p_away / total


def game_lines(seasons=None, through_week=None) -> pl.DataFrame:
    """
    One row per game: closing spread, total, and de-vigged home win
    probability. Games without a posted line are dropped.
    """
    sch = load_schedules()
    if seasons is not None:
        sch = sch.filter(pl.col("season").is_in(list(seasons)))
    if through_week is not None:
        sch = sch.filter(pl.col("week") <= through_week)

    sch = sch.filter(pl.col("spread_line").is_not_null())

    out = sch.select([
        "game_id", "season", "week", "gameday", "home_team", "away_team",
        "spread_line", "total_line", "home_moneyline", "away_moneyline",
        "home_score", "away_score", "result",
    ])

    have_ml = out.filter(pl.col("home_moneyline").is_not_null())
    if have_ml.height:
        ph = american_to_prob(have_ml["home_moneyline"].to_numpy())
        pa = american_to_prob(have_ml["away_moneyline"].to_numpy())
        ph, _ = devig_pair(ph, pa)
        have_ml = have_ml.with_columns(pl.Series("market_home_wp", ph))
        out = out.join(
            have_ml.select(["game_id", "market_home_wp"]), on="game_id", how="left"
        )
    else:
        out = out.with_columns(pl.lit(None, dtype=pl.Float64).alias("market_home_wp"))

    return out


def implied_team_totals(seasons=None, through_week=None) -> pl.DataFrame:
    """
    Split the game total into two team totals using the spread:

        home_total = total/2 + spread/2
        away_total = total/2 - spread/2

    nflverse `spread_line` is positive when the HOME team is favoured, and
    is stated as the home team's expected margin. Verified against 27
    seasons of outcomes: corr(spread_line, home_score - away_score) = 0.426,
    and home margin averages +7.95 when spread_line > 3, -6.95 when it is
    below -3. The first version of this function had the sign inverted and
    every test still passed, because the two columns reconcile to
    total_line either way. test_market_lines.py now checks the sign against
    the favourite, not just the sum.

    This is the single most useful market-derived quantity for a usage
    model. A team projected for 27 points throws and runs more than one
    projected for 17, and both numbers are known before kickoff for free.

    THE TRAP, flagged in nfl-data-recon: feeding this into the usage model
    laundered the market's opinion into predictions that are then compared
    AGAINST the market. Keep a no-market variant and report both, or the
    comparison is circular.
    """
    g = game_lines(seasons=seasons, through_week=through_week)
    return g.with_columns([
        (pl.col("total_line") / 2 + pl.col("spread_line") / 2).alias("home_implied_total"),
        (pl.col("total_line") / 2 - pl.col("spread_line") / 2).alias("away_implied_total"),
    ])
