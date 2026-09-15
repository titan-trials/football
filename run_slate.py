"""
The weekly pipeline, one command.

    python run_slate.py                  # everything, for the upcoming week
    python run_slate.py --dry-run        # say what it would do, spend nothing
    python run_slate.py --week 3         # override the week

Three steps in order, each feeding the next:

    1. capture   buy this week's prop lines that are not already on disk
    2. predict   build the slate
    3. compare   join the two and report the disagreements

Scoring is deliberately NOT in here. `score_slate.py` runs days later,
against a different set of facts, and folding it in would invite scoring a
week that is still being played.

WHY ONE FILE
------------
The steps have a real order and real preconditions, and a human holding
that order in their head is the weakest link in the process. Capture must
precede predict, because predict is free and re-runnable while a missed
capture can only be bought back at ten times the price. Compare must
follow both. Every guardrail below used to be a thing you had to remember:

  * the WEEK is resolved from the schedule, not from a flag you might get
    wrong, and not from last-completed-week + 1, which said week 2 on a day
    when week 1 had 14 games still to play;
  * capture SKIPS events already in the market log, so re-running this
    mid-week to pick up Sunday's games costs nothing for Thursday's;
  * a real capture refuses the shared ODDS_API_KEY, which belongs to
    baseball_predictor and spends its quota;
  * spending is hard-capped, and the cap has to be raised on purpose;
  * a capture failure does NOT stop the pipeline. Predicting is free and
    the slate is worth having on its own. Only the compare step needs the
    market, and it says so and skips rather than failing.

The whole thing is re-runnable at any point in the week. That is the
property worth protecting: nothing here is a one-shot you can spoil by
running it twice.
"""
from __future__ import annotations

import argparse
import os
import traceback

import polars as pl

from data.nflverse import current_season, upcoming_week

DEFAULT_MARKETS = ["player_reception_yds", "player_receptions"]


def _rule(n: int, title: str) -> None:
    print(f"\n{'=' * 72}\n[{n}/3] {title}\n{'=' * 72}")


# ---------------------------------------------------------------------
# 1. capture
# ---------------------------------------------------------------------

def step_capture(season: int, week: int, markets: list, max_credits: int,
                 dry_run: bool, allow_shared_key: bool,
                 refresh: bool) -> bool:
    """
    Returns True if the market log holds anything for this week afterwards
    -- whether this run bought it or a previous one did.
    """
    from data.odds import (SharedKeyRefused, append_market_log,
                           captured_event_ids, check_quota, fetch_slate_props)

    already = captured_event_ids(markets)
    if already and not refresh:
        print(f"  {len(already)} events already captured for all "
              f"{len(markets)} markets")

    try:
        q = check_quota()
        print(f"  key from {q.get('source')}: {q.get('remaining')} credits "
              f"remaining, {q.get('used')} used")
    except RuntimeError as e:
        print(f"  NO USABLE KEY -- skipping capture.\n    {str(e).splitlines()[0]}")
        print("  The slate will still be built; only the market comparison "
              "needs lines.")
        return bool(already)

    try:
        rows, cons, remaining = fetch_slate_props(
            markets=markets, dry_run=dry_run, season=season, week=week,
            max_credits=max_credits, allow_shared_key=allow_shared_key,
            skip_captured=not refresh)
    except SharedKeyRefused as e:
        print(f"  REFUSED: {e}")
        print("  ODDS_API_KEY in the environment is baseball_predictor's key "
              "and quota.\n  Put this project's key in .env as "
              "NFL_ODDS_API_KEY, or pass --allow-shared-key.")
        return bool(already)
    except RuntimeError as e:
        print(f"  capture failed: {e}")
        return bool(already)

    if dry_run:
        print("  dry run -- nothing spent")
        return bool(already)

    if cons is None or cons.is_empty():
        if already:
            print("  nothing new to buy; using what is already on disk")
        else:
            print("  no lines returned. Player props post around Wednesday; "
                  "before that\n  the events exist and the markets come back "
                  "empty, which is not an error.")
        return bool(already)

    n = append_market_log(cons)
    print(f"  captured {cons.height} rows, {cons['player'].n_unique()} "
          f"players; log now {n} rows; {remaining} credits left")
    return True


# ---------------------------------------------------------------------
# 2. predict
# ---------------------------------------------------------------------

def check_snap_feed(season: int, week: int) -> None:
    """
    Warn when the snap feed for last week has not fully landed.

    SNAP_DERIVED_ROLE ranks each position group by PRIOR-WEEK snap share,
    so a half-landed feed does not fail -- it silently re-ranks half of
    every roster and leaves the other half on chart order. That is worse
    than not using snaps at all, and it is invisible in the output.

    Measured 2026-09-13: a few hours after week 1 kicked off the feed held
    187 rows against roughly 1,200 for a complete week. It fills in over
    the following day or two (PFR-dependent), which is why Wednesday is
    the run and not Monday night.
    """
    from model_flags import SNAP_DERIVED_ROLE
    if not SNAP_DERIVED_ROLE or week < 2:
        return
    try:
        from data.nflverse import load_snap_counts
        sn = load_snap_counts([season])
        n = sn.filter(pl.col("week") == week - 1).height
    except Exception as e:  # noqa: BLE001
        print(f"  could not check the snap feed ({str(e).splitlines()[0][:60]})")
        return
    if n >= 900:
        print(f"  snap feed for week {week - 1}: {n} rows -- complete")
    else:
        print(f"  WARNING: snap feed for week {week - 1} has only {n} rows "
              f"(expect ~1,200).")
        print("    Roles are ranked on prior-week snaps, so a partial feed "
              "re-ranks")
        print("    some rosters and not others. Wait for it to land, or run "
              "with")
        print("    SNAP_DERIVED_ROLE off, rather than predicting on half of it.")


def step_predict(season: int, week: int, dry_run: bool) -> bool:
    import predict_slate

    path = os.path.join("slates", f"slate_{season}_wk{week}.parquet")
    check_snap_feed(season, week)
    if dry_run:
        print(f"  would write {path}")
        print("  (re-running preserves rows for games already kicked off)")
        return os.path.exists(path)

    predict_slate.main(season=season, week=week, trailing=8, prior_seasons=2,
                       props=list(predict_slate.PROPS))
    return os.path.exists(path)


# ---------------------------------------------------------------------
# 3. compare
# ---------------------------------------------------------------------

def step_compare(season: int, week: int, top: int, dry_run: bool,
                 have_market: bool, have_slate: bool) -> None:
    if not have_slate:
        print("  no slate on disk -- nothing to compare")
        return
    if not have_market:
        print("  no lines captured for this week -- skipping the comparison")
        print("  (the slate is still written and still worth having)")
        return
    if dry_run:
        print(f"  would join the slate to the market and write "
              f"cache/edges_{season}_wk{week}.parquet")
        return

    import compare_market
    df = compare_market.build(season, week)
    if df is None or df.is_empty():
        print("  the slate and the lines did not join on any row")
        return
    compare_market.report(df, top)
    out = f"cache/edges_{season}_wk{week}.parquet"
    os.makedirs("cache", exist_ok=True)
    df.write_parquet(out)
    print(f"\n  wrote {out}")


# ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--week", type=int, default=None)
    ap.add_argument("--markets", nargs="+", default=DEFAULT_MARKETS)
    ap.add_argument("--max-credits", type=int, default=60,
                    help="hard stop on spending; raise it deliberately")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what each step would do; spends nothing")
    ap.add_argument("--refresh", action="store_true",
                    help="re-buy lines already on disk (costs credits, and "
                         "makes the benchmark inconsistent across weeks)")
    ap.add_argument("--allow-shared-key", action="store_true",
                    help="spend baseball_predictor's quota")
    ap.add_argument("--skip-capture", action="store_true",
                    help="predict and compare using lines already on disk")
    a = ap.parse_args()

    season = a.season if a.season is not None else current_season()
    week = a.week if a.week is not None else upcoming_week(season)

    print(f"{'=' * 72}\nWEEKLY PIPELINE -- {season} week {week}"
          f"{'   [DRY RUN]' if a.dry_run else ''}\n{'=' * 72}")
    print("week resolved from the schedule: the week containing the next "
          "kickoff.")

    _rule(1, "capture the market")
    if a.skip_capture:
        from data.odds import captured_event_ids
        have_market = bool(captured_event_ids(a.markets))
        print(f"  skipped by request; {'lines are' if have_market else 'no lines'} "
              f"on disk")
    else:
        have_market = step_capture(season, week, a.markets, a.max_credits,
                                   a.dry_run, a.allow_shared_key, a.refresh)

    _rule(2, "predict the slate")
    try:
        have_slate = step_predict(season, week, a.dry_run)
    except SystemExit as e:
        print(f"  {e}")
        have_slate = False
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        have_slate = False

    _rule(3, "compare to the market")
    step_compare(season, week, a.top, a.dry_run, have_market, have_slate)

    print(f"\n{'=' * 72}")
    if a.dry_run:
        print("dry run complete -- no credits spent, nothing written")
    else:
        print(f"done. slate: slates/slate_{season}_wk{week}.parquet")
    print("score it after the games with:  python score_slate.py")
    print("browse it with:                 streamlit run dashboard.py")


if __name__ == "__main__":
    main()
