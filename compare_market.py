"""
Model versus market.

    python compare_market.py --season 2026 --week 1

This is the only comparison that decides whether any of this is worth
anything. Everything measured so far has been against climatology -- the
player's own trailing distribution -- and `context-features-availability.md`
says exactly what that is worth: *"Brier skill against the base rate is a
low bar a sportsbook clears easily."*

WHAT AN "EDGE" IS HERE, AND THE THREE WAYS IT LIES
--------------------------------------------------
    edge = P_model(over line) - P_market(over line)

evaluated at the book's OWN line, from the model's stored pmf. Three things
have to be right before that number means anything, and each has been a real
bug in this project already:

1. **The player must be the right player.** A bad name join compares one
   player's projection to another's line and produces an edge that looks
   entirely normal. `data/player_match.py` scopes by team, records how each
   match was made, and returns unmatched names rather than dropping them.
   Match rate is printed first here, before any edge.

2. **The market probability must be de-vigged.** A raw two-way market sums
   to more than 1 -- measured hold on the first real capture was 6.7% --
   so comparing against raw implied probability credits the model with
   beating the hold, which it did not do.

3. **The line must be the book's line.** Books hang 13.5 and 47.5, not the
   round numbers in `spec.lines`. Interpolating between stored columns would
   put an approximation error inside the one number that has to be exact, so
   the full pmf is stored and evaluated directly.

WHAT A POSITIVE EDGE IS NOT
---------------------------
It is not a bet and it is not proof of skill. With ~200 prop rows a week,
heavy within-game correlation, and a market that prices these for a living,
the honest expectation is that most apparent edge is model error. The number
that matters is not this week's mean edge but the CALIBRATION of edges
across many weeks: when the model says +5%, does it win 5% more often?
That takes most of a season, which is why the log starts now.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import polars as pl

from data.player_match import match_players, match_report
from features.props import PROPS

MARKET_TO_PROP = {
    "player_reception_yds": "receiving_yards",
    "player_receptions": "receptions",
    "player_rush_yds": "rushing_yards",
    "player_pass_yds": "passing_yards",
}


def model_prob_over(pmf: list, support_min: int, line: float) -> float:
    """P(total > line) from a stored pmf. Strict, as a book's 'over' is."""
    p = np.asarray(pmf, dtype=float)
    sup = np.arange(support_min, support_min + len(p))
    return float(p[sup > line].sum())


def build(season: int, week: int, slate_dir: str = "slates",
          market_log: str = "cache/market_log.parquet") -> pl.DataFrame:
    slate_path = os.path.join(slate_dir, f"slate_{season}_wk{week}.parquet")
    pmf_path = os.path.join(slate_dir, f"pmf_{season}_wk{week}.parquet")
    for p in (slate_path, pmf_path, market_log):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")

    slate = pl.read_parquet(slate_path)
    pmfs = pl.read_parquet(pmf_path)
    market = pl.read_parquet(market_log)

    roster = slate.select(["player_id", "player_name", "team"]).unique()
    ev = None
    if {"home_team", "away_team"}.issubset(set(market.columns)):
        ev = (market.select(["event_id", "home_team", "away_team"]).unique()
                    .drop_nulls("home_team"))

    matched, unmatched = match_players(market, roster, ev)
    print(match_report(matched, unmatched, n_names=market["player"].n_unique()))
    if matched.is_empty():
        raise SystemExit("nothing matched; not computing edges")

    matched = matched.with_columns(
        pl.col("market").replace_strict(MARKET_TO_PROP, default=None).alias("prop")
    ).drop_nulls("prop")

    # Drop overlapping columns before the join; polars suffixes them and a
    # duplicate `team`/`line` then breaks every later group_by.
    pmfs = pmfs.select(["player_id", "prop", "_pmf", "_support_min"])
    joined = matched.join(pmfs, on=["player_id", "prop"], how="inner")
    if joined.is_empty():
        raise SystemExit("no market row joined a modelled prop")

    probs = [
        model_prob_over(r["_pmf"], r["_support_min"], r["line"])
        for r in joined.iter_rows(named=True)
    ]
    out = joined.with_columns(pl.Series("model_prob", probs)).with_columns(
        (pl.col("model_prob") - pl.col("market_prob")).alias("edge")
    ).drop(["_pmf", "_support_min"])

    avail = slate.select(["player_id", "prop", "availability"]).unique()
    return out.join(avail, on=["player_id", "prop"], how="left")


def report(df: pl.DataFrame, top: int = 15):
    print(f"\n=== model vs market: {df.height} priced props ===")
    hold = f"   mean hold {df['hold'].mean():.4f}" if "hold" in df.columns else ""
    print(f"  mean |edge| {df['edge'].abs().mean():.4f}   "
          f"mean edge {df['edge'].mean():+.4f}{hold}")

    by = (df.group_by("market").agg([
        pl.len().alias("n"),
        pl.col("edge").mean().alias("mean_edge"),
        pl.col("edge").abs().mean().alias("mean_abs_edge"),
        # NOT aliased to "market" -- that is the group key, and polars
        # raises a duplicate-column error that points at the frame rather
        # than at the alias.
        pl.col("model_prob").mean().alias("mean_model_p"),
        pl.col("market_prob").mean().alias("mean_market_p"),
    ]).sort("n", descending=True))
    print()
    print(by.to_pandas().round(4).to_string(index=False))

    print(f"\n  A mean edge near zero is the GOOD outcome for a first look: it "
          f"says the model\n  and the market agree on average, and disagreements "
          f"are two-sided rather than\n  a systematic bias. A large mean edge "
          f"almost always means a units or scope bug,\n  not free money.")

    print(f"\n=== largest disagreements ===")
    print("  These are where to look for model bugs first, not where to bet.")
    cols = ["player", "team", "market", "line", "model_prob", "market_prob",
            "edge", "n_books"]
    cols = [c for c in cols if c in df.columns]
    big = df.sort(pl.col("edge").abs(), descending=True).head(top)
    print(big.select(cols).to_pandas().round(4).to_string(index=False))

    if "availability" in df.columns:
        flagged = df.filter(pl.col("availability") != "CLEAR")
        if flagged.height:
            print(f"\n  {flagged.height} of these carry an availability flag "
                  f"(mean edge {flagged['edge'].mean():+.4f} vs "
                  f"{df.filter(pl.col('availability') == 'CLEAR')['edge'].mean():+.4f} "
                  f"for clear players).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int, default=None)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.week is None:
        from data.nflverse import upcoming_week
        a.week = upcoming_week(a.season)

    df = build(a.season, a.week)
    report(df, a.top)
    out = a.out or f"cache/edges_{a.season}_wk{a.week}.parquet"
    df.write_parquet(out)
    print(f"\nwrote {df.height} rows to {out}")
