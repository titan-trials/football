"""
Capture the week's player-prop lines and append them to the market log.

    python capture_props.py --check                 # quota only, free
    python capture_props.py --dry-run               # cost estimate, free
    python capture_props.py                         # spends credits

WHY THIS IS A SCRIPT AND NOT A ONE-LINER
----------------------------------------
This was a `python -c "from data.odds import ..."` incantation, which is
the wrong shape for the one operation in the project that SPENDS MONEY and
cannot be undone. A retyped one-liner is how the wrong key gets used, the
wrong week gets pulled, or 424 credits get spent on the rest of the season.

Three things are therefore true of this file and should stay true:

  * `--check` and `--dry-run` cost nothing, and `--dry-run` is the default
    recommendation before every real pull.
  * The week is resolved with `upcoming_week`, not with
    `current_week() + 1`. On 2026-09-12 week 1 had 16 games with only two
    played; last-completed-week-plus-one said week 2 and would have
    captured the wrong slate.
  * A real pull refuses the shared `ODDS_API_KEY` unless
    `--allow-shared-key` is passed on purpose. That variable is
    baseball_predictor's key and its own quota.

THE STANDING BUDGET RULES
-------------------------
  * Featured markets (spread, total, moneyline) are FREE from nflverse back
    to 1999. Never buy them here.
  * Live player props cost regions x markets x events. One week of two
    markets is about 28-30 credits against a 500 monthly allowance.
  * HISTORICAL player props cost 10x live and only start 2023-05-03. A week
    that goes uncaptured can only be bought back at ten times the price, so
    a missed Wednesday is expensive in a way a missed model run is not.
"""
from __future__ import annotations

import argparse

from data.nflverse import upcoming_week
from data.odds import (SharedKeyRefused, append_market_log, check_quota,
                       fetch_slate_props)

DEFAULT_MARKETS = ["player_reception_yds", "player_receptions"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--week", type=int, default=None,
                    help="defaults to the week containing the next kickoff")
    ap.add_argument("--markets", nargs="+", default=DEFAULT_MARKETS)
    ap.add_argument("--max-credits", type=int, default=60,
                    help="hard stop, not a warning; raise it deliberately")
    ap.add_argument("--check", action="store_true",
                    help="print the quota and exit; costs nothing")
    ap.add_argument("--dry-run", action="store_true",
                    help="list events and estimate cost; costs nothing")
    ap.add_argument("--allow-shared-key", action="store_true",
                    help="use ODDS_API_KEY (baseball's key and quota)")
    a = ap.parse_args()

    # `--check` answers "do I have a working key", so a missing key is an
    # ANSWER here, not a crash. Everything else still fails loudly.
    try:
        q = check_quota()
        print(f"key source: {q.get('source')}  "
              f"remaining: {q.get('remaining')}  used: {q.get('used')}")
    except RuntimeError as e:
        print(f"no usable key:\n{e}")
        if a.check:
            return
        raise
    if a.check:
        return

    week = a.week if a.week is not None else upcoming_week(a.season)
    print(f"target: {a.season} week {week}   markets: {', '.join(a.markets)}")

    try:
        rows, cons, remaining = fetch_slate_props(
            markets=a.markets, dry_run=a.dry_run, season=a.season, week=week,
            max_credits=a.max_credits, allow_shared_key=a.allow_shared_key)
    except SharedKeyRefused as e:
        raise SystemExit(
            f"{e}\n\nThe process environment's ODDS_API_KEY belongs to "
            f"baseball_predictor. Put this project's key in .env as "
            f"NFL_ODDS_API_KEY, or pass --allow-shared-key if you really "
            f"mean to spend the other project's quota.")

    if a.dry_run:
        print("\ndry run -- nothing was spent. Re-run without --dry-run to "
              "capture.")
        return

    if cons is None or cons.is_empty():
        raise SystemExit(
            "no consensus rows returned. Player props for a week usually "
            "post Wednesday; before that the events exist but the markets "
            "are empty, which is not an error.")

    n = append_market_log(cons)
    print(f"\ncaptured {cons.height} consensus rows "
          f"({cons['player'].n_unique()} players); "
          f"market log now holds {n} rows")
    print(f"credits remaining: {remaining}")
    print("\nnext: python compare_market.py --season "
          f"{a.season} --week {week}")


if __name__ == "__main__":
    main()
