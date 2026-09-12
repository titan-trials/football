"""
Score a slate after the games are played.

    python score_slate.py --season 2026 --week 1

Joins the committed slate to what actually happened, appends to a running
log, and reports skill and calibration. When a market log exists for the
same week it also scores the MARKET on identical rows, which is the only
comparison that decides whether the model is worth anything.

WHY THIS SCRIPT IS THE ANSWER TO AN OPEN QUESTION
-------------------------------------------------
As of 2026-09-12 the model and the market disagree about the prominent
players by a wide margin, and reasoning has not settled which is right:

    on 2025 mid-season history the model is essentially perfect --
      the player's own trailing average  6.269
      the model predicted                6.038   (96.3%)
      what actually happened             6.106   (97.4%)

    on the 2026 week 1 slate it is far below the market --
      their 2025 average                 4.70
      the book prices at                 ~98% of it
      the model projects                 ~75% of it

Week 1 does regress harder than mid-season -- players with a 6.335 average
in 2024 actually averaged 5.830 in 2025 week 1, a ratio of 0.920 -- so the
truth is between the two and nearer the market. But that is an argument,
and both sides have a timestamped, committed prediction sitting on disk.

**Scoring them is an experiment, not an argument.** Run it and stop
guessing.

THE RULES IT KEEPS
------------------
* **Pooled, never averaged.** Skill is `1 - brier/(p(1-p))`, and the average
  of a ratio is not the ratio of averages. Baseball's dashboard and
  score_slate printed two different HR edges for the same rows because one
  averaged per-slate skills. Everything here pools.
* **Clean rows are tracked, not filtered.** `clean-vs-tainted-rows.md`: a
  filter that correlates with kickoff time silently re-weights the sample.
  Both subsets are reported; neither is thrown away.
* **The log is append-only and idempotent.** Re-running a week replaces that
  week's rows rather than duplicating them.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from typing import Optional

import numpy as np
import polars as pl

from data.nflverse import current_season, load_player_stats, load_schedules
from features.props import PROPS
from model.scoring import brier_skill, crps_pmf

SLATE_DIR = "slates"
SCORING_LOG = "cache/scoring_log.parquet"


def committed_weeks(season: int) -> list[int]:
    """Weeks with a slate file on disk, ascending."""
    out = []
    for path in glob.glob(os.path.join(SLATE_DIR, f"slate_{season}_wk*.parquet")):
        m = re.search(r"_wk(\d+)\.parquet$", path)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def week_completeness(season: int) -> dict:
    """
    Fraction of each week's scheduled teams that have results yet.

    Existence of stats for a week is NOT the same as the week being over.
    Checked on 2026-09-12: week 1 had stats for 4 of 32 scheduled teams --
    the Thursday opener and one more -- so "week 1 has data" was true and
    "week 1 can be scored" was false. Auto-detection without this guard
    would have scored a 16-game week off two games and written the result
    into the permanent log.
    """
    try:
        ps = load_player_stats([season])
        sch = load_schedules().filter(pl.col("season") == season)
    except Exception:  # noqa: BLE001
        return {}
    if ps.is_empty() or sch.is_empty():
        return {}
    have = (ps.group_by("week").agg(pl.col("team").n_unique().alias("done")))
    want = (sch.group_by("week").agg((pl.len() * 2).alias("scheduled")))
    j = have.join(want, on="week", how="inner")
    return {int(r["week"]): r["done"] / r["scheduled"]
            for r in j.iter_rows(named=True) if r["scheduled"]}


def latest_scorable(season: int, min_complete: float = 0.9) -> Optional[int]:
    """
    The most recent week that has a committed slate AND is actually over.

    This is what makes `python score_slate.py` work with no arguments.
    The rule is "the newest week that can honestly be scored", not "the
    last week with any data" -- see week_completeness for why those differ
    and what it would have cost.

    Both halves are required. A slate with no results is this week's
    upcoming games; results with no slate is a week the predictor never
    ran on, and there is nothing to score it against.
    """
    weeks = committed_weeks(season)
    if not weeks:
        return None
    comp = week_completeness(season)
    have = [w for w in weeks if comp.get(w, 0.0) >= min_complete]
    return have[-1] if have else None


def actuals(season: int, week: int) -> pl.DataFrame:
    """What each player actually did, one row per (player, prop)."""
    ps = load_player_stats([season]).filter(pl.col("week") == week)
    if ps.is_empty():
        raise SystemExit(
            f"No player stats for {season} week {week} yet. Play-by-play and "
            f"weekly stats land nightly after games; Thursday's data is the "
            f"cleanest, after NFL stat corrections.")
    frames = []
    for name, spec in PROPS.items():
        if spec.stat_outcome not in ps.columns:
            continue
        frames.append(
            ps.select([
                pl.lit(season).alias("season"), pl.lit(week).alias("week"),
                pl.col("team"), pl.col("player_id"),
                pl.lit(name).alias("prop"),
                pl.col(spec.stat_outcome).fill_null(0).cast(pl.Float64).alias("actual"),
            ]))
    return pl.concat(frames, how="diagonal_relaxed")


def score(season: int, week: int) -> pl.DataFrame:
    slate_path = os.path.join(SLATE_DIR, f"slate_{season}_wk{week}.parquet")
    pmf_path = os.path.join(SLATE_DIR, f"pmf_{season}_wk{week}.parquet")
    if not os.path.exists(slate_path):
        raise SystemExit(f"missing {slate_path}")

    slate = pl.read_parquet(slate_path)
    got = actuals(season, week)
    df = slate.join(got, on=["season", "week", "team", "player_id", "prop"], how="inner")
    if df.is_empty():
        raise SystemExit("slate and actuals did not join on any row")

    if os.path.exists(pmf_path):
        pmfs = pl.read_parquet(pmf_path).select(
            ["player_id", "prop", "_pmf", "_support_min"])
        df = df.join(pmfs, on=["player_id", "prop"], how="left")
        crps = []
        for r in df.iter_rows(named=True):
            if r.get("_pmf") is None:
                crps.append(None)
                continue
            p = np.asarray(r["_pmf"], dtype=float)
            sup = np.arange(r["_support_min"], r["_support_min"] + len(p))
            crps.append(float(crps_pmf([r["actual"]], sup, p[None, :])[0]))
        df = df.with_columns(pl.Series("crps", crps, dtype=pl.Float64)).drop(
            ["_pmf", "_support_min"])
    return df


def report(df: pl.DataFrame):
    n_clean = int(df["clean"].sum()) if "clean" in df.columns else 0
    print(f"\n=== scored {df.height} rows, {df['prop'].n_unique()} props ===")
    print(f"  clean (predicted before kickoff): {n_clean} / {df.height}")
    print("  both subsets are reported below; neither is discarded -- a filter "
          "that\n  correlates with kickoff time re-weights the sample.")

    for prop in sorted(df["prop"].unique().to_list()):
        spec = PROPS[prop]
        sub = df.filter(pl.col("prop") == prop)
        exp_col = "expected" if "expected" in sub.columns else None
        bias = (float(sub[exp_col].mean() - sub["actual"].mean())
                if exp_col else float("nan"))
        crps = float(sub["crps"].mean()) if "crps" in sub.columns and sub["crps"].null_count() < sub.height else float("nan")
        print(f"\n  {prop}   n={sub.height}  mean actual {float(sub['actual'].mean()):.2f}"
              f"  bias {bias:+.2f}  CRPS {crps:.3f}")
        for L in spec.lines:
            col = f"over_{L}"
            if col not in sub.columns:
                continue
            s2 = sub.filter(pl.col(col).is_not_null())
            if s2.height < 30:
                continue
            y = (s2["actual"] > L).to_numpy().astype(float)
            if y.mean() <= 0 or y.mean() >= 1:
                continue
            print(f"    over {L:6}: base {y.mean():.3f}  "
                  f"skill {brier_skill(y, s2[col].to_numpy()):+.4f}")


def score_market(season: int, week: int, df: pl.DataFrame):
    """
    THE COMPARISON THAT MATTERS. Model and market on identical rows.

    Both predictions were committed before kickoff, so this is a fair test
    and neither side can be adjusted after the fact.
    """
    edges_path = f"cache/edges_{season}_wk{week}.parquet"
    if not os.path.exists(edges_path):
        print(f"\n  (no {edges_path}; run compare_market.py to include the market)")
        return
    e = pl.read_parquet(edges_path)
    m = e.join(df.select(["player_id", "prop", "actual"]).unique(),
               on=["player_id", "prop"], how="inner")
    if m.is_empty():
        print("\n  market rows did not join any scored row")
        return

    print(f"\n=== MODEL vs MARKET on {m.height} identical priced rows ===")
    y = (m["actual"] > m["line"]).to_numpy().astype(float)
    bm = brier_skill(y, m["model_prob"].to_numpy())
    bk = brier_skill(y, m["market_prob"].to_numpy())
    print(f"  base rate (share going over)  {y.mean():.3f}")
    print(f"  model  Brier skill            {bm:+.4f}")
    print(f"  market Brier skill            {bk:+.4f}")
    print(f"  difference                    {bm - bk:+.4f}")
    print("\n  Negative difference = the market is better, which is the "
          "expected\n  result and is not a failure. The question is by how "
          "much, and whether\n  it shrinks as the model improves.")

    # the open question, settled directly
    print(f"\n  mean predicted by model   {m['model_prob'].mean():.3f}")
    print(f"  mean implied by market    {m['market_prob'].mean():.3f}")
    print(f"  actually went over        {y.mean():.3f}")
    print("  ^ whichever of the first two is closer to the third was right "
          "about\n    the prominent players. That is the compression question, "
          "answered.")


def append_log(df: pl.DataFrame, path: str = SCORING_LOG) -> int:
    """Append-only and idempotent: re-running a week replaces its rows."""
    keep = [c for c in ("season", "week", "team", "player_id", "player_name",
                        "prop", "expected", "actual", "crps", "clean",
                        "availability") if c in df.columns]
    keep += [c for c in df.columns if c.startswith("over_")]
    fresh = df.select(keep)
    if os.path.exists(path):
        old = pl.read_parquet(path)
        old = old.filter(~((pl.col("season") == df["season"][0])
                           & (pl.col("week") == df["week"][0])))
        fresh = pl.concat([old, fresh], how="diagonal_relaxed")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fresh.write_parquet(path)
    return fresh.height


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--week", type=int, default=None,
                    help="defaults to the newest week with both a slate and "
                         "actuals on disk")
    ap.add_argument("--all", action="store_true",
                    help="score every scorable week, oldest first")
    a = ap.parse_args()

    season = a.season if a.season is not None else current_season()

    if a.all:
        weeks = [w for w in committed_weeks(season)
                 if w <= (latest_scorable(season) or 0)]
        if not weeks:
            raise SystemExit(f"no scorable week found for {season}")
        for w in weeks:
            print(f"\n{'#' * 70}\n# {season} week {w}\n{'#' * 70}")
            d = score(season, w)
            report(d)
            score_market(season, w, d)
            append_log(d)
        print(f"\nscoring log now holds "
              f"{pl.read_parquet(SCORING_LOG).height} rows ({SCORING_LOG})")
        raise SystemExit(0)

    week = a.week
    if week is None:
        week = latest_scorable(season)
        if week is None:
            cw = committed_weeks(season)
            comp = week_completeness(season)
            detail = ", ".join(f"wk{w} {comp.get(w, 0.0):.0%} complete"
                               for w in cw) or "none"
            raise SystemExit(
                f"Nothing fully played to score for {season}.\n"
                f"  slates on disk: {detail}\n"
                f"  A week needs 90% of its scheduled teams to have results. "
                f"Stats land nightly and Tuesday is cleanest, after stat "
                f"corrections.\n"
                f"  Pass --week N to score a partial week anyway.")
        print(f"scoring {season} week {week} (newest week with both a slate "
              f"and results)")

    df = score(season, week)
    report(df)
    score_market(season, week, df)
    total = append_log(df)
    print(f"\nscoring log now holds {total} rows ({SCORING_LOG})")
