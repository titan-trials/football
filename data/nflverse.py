"""
nflverse loaders, with serve-time discipline enforced at the boundary.

WHY THIS FILE IS SHAPED THIS WAY
--------------------------------
The baseball model's worst structural bug was train/serve skew: two of
_project_pa's five features meant different things at fit time and at
prediction time, and nothing in the code could have caught it, because the
training path and the serving path each pulled their own data.

Football makes that failure mode much easier to hit, because its feeds
genuinely differ in when they land. Route participation exists for 45k
plays a season and updates ONLY after the postseason. A model trained on
route share cannot be served in week 3, and the code will not tell you --
it will just quietly have NaNs, or worse, be backfilled by a later pull and
look fine in a backtest.

So: every loader here takes `through_week`, and returns nothing dated after
it. There is no way to ask this module for "the current state of the
world"; you must say which week you are standing in. Backtests and the live
predictor call the same functions with the same argument, which is the
property `model-review-2026-09-05.md` says the K backtest lacked.

Loaders for POSTSEASON_ONLY_FEEDS raise unless explicitly unlocked with
allow_unservable=True, which exists so a lab script can measure how much a
participation feature would have been worth -- and makes it impossible to
do so by accident.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import nflreadpy as nfl
import polars as pl

from config import POSTSEASON_ONLY_FEEDS, SERVE_TIME_FEEDS
from data.cache import DEFAULT_TTL_HOURS, load_cached, save_cache


class UnservableFeedError(RuntimeError):
    """Raised when code asks for a feed that will not exist at prediction time."""


def _check_servable(feed: str, allow_unservable: bool) -> None:
    if feed in POSTSEASON_ONLY_FEEDS and not allow_unservable:
        raise UnservableFeedError(
            f"'{feed}' does not update during the regular season -- it lands "
            f"after the postseason completes. A feature built on it cannot be "
            f"served for an upcoming slate.\n"
            f"If you are measuring its value in a lab, pass "
            f"allow_unservable=True and record in CONTEXT.md that the result "
            f"is not deployable as-is."
        )
    if feed not in SERVE_TIME_FEEDS and feed not in POSTSEASON_ONLY_FEEDS:
        raise ValueError(f"Unknown feed '{feed}'; add it to config.py first.")


def _ttl_for(season: int, current_season: int) -> Optional[float]:
    """Completed seasons never expire; the live season expires on a timer."""
    return None if season < current_season else DEFAULT_TTL_HOURS


def _cut(df: pl.DataFrame, through_week: Optional[int]) -> pl.DataFrame:
    """Drop anything dated after the week we are standing in."""
    if through_week is None or "week" not in df.columns:
        return df
    return df.filter(pl.col("week") <= through_week)


def _load(feed: str, seasons: Sequence[int], loader, through_week: Optional[int],
          current_season: int, allow_unservable: bool, **kwargs) -> pl.DataFrame:
    _check_servable(feed, allow_unservable)
    frames = []
    for season in seasons:
        suffix = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
        key = f"{feed}_{season}" + (f"_{suffix}" if suffix else "")
        df = load_cached(key, ttl_hours=_ttl_for(season, current_season))
        if df is None:
            df = loader(seasons=[season], **kwargs)
            save_cache(key, df)
        frames.append(df)
    if not frames:
        return pl.DataFrame()
    out = pl.concat(frames, how="diagonal_relaxed")
    return _cut(out, through_week)


# --- The feeds ---------------------------------------------------------

def load_pbp(seasons: Iterable[int], through_week: Optional[int] = None,
             current_season: int = 2026, regular_only: bool = True) -> pl.DataFrame:
    df = _load("pbp", list(seasons), nfl.load_pbp, through_week, current_season, False)
    if regular_only and "season_type" in df.columns:
        df = df.filter(pl.col("season_type") == "REG")
    return df


def load_player_stats(seasons: Iterable[int], through_week: Optional[int] = None,
                      current_season: int = 2026, regular_only: bool = True) -> pl.DataFrame:
    df = _load("player_stats", list(seasons),
               lambda seasons: nfl.load_player_stats(seasons=seasons, summary_level="week"),
               through_week, current_season, False)
    if regular_only and "season_type" in df.columns:
        df = df.filter(pl.col("season_type") == "REG")
    return df


def load_snap_counts(seasons: Iterable[int], through_week: Optional[int] = None,
                     current_season: int = 2026) -> pl.DataFrame:
    return _load("snap_counts", list(seasons), nfl.load_snap_counts,
                 through_week, current_season, False)


def load_injuries(seasons: Iterable[int], through_week: Optional[int] = None,
                  current_season: int = 2026) -> pl.DataFrame:
    """
    Weekly injury + practice report. Available BEFORE kickoff, which makes
    it one of the few feeds that can describe the game about to be played
    rather than the last one.

    Note: through_week here is inclusive of the CURRENT week on purpose --
    Wednesday's report for Sunday's game is legitimately available at
    predict time, unlike every other feed, which only has week N-1.
    """
    return _load("injuries", list(seasons), nfl.load_injuries,
                 through_week, current_season, False)


def load_depth_charts(seasons: Iterable[int], through_week: Optional[int] = None,
                      current_season: int = 2026) -> pl.DataFrame:
    return _load("depth_charts", list(seasons), nfl.load_depth_charts,
                 through_week, current_season, False)


def load_participation(seasons: Iterable[int], through_week: Optional[int] = None,
                       current_season: int = 2026,
                       allow_unservable: bool = False) -> pl.DataFrame:
    """Routes, coverage, personnel. 2016-2025 only, and NOT servable in-season."""
    return _load("participation", list(seasons), nfl.load_participation,
                 through_week, current_season, allow_unservable)


def load_schedules(through_season: Optional[int] = None) -> pl.DataFrame:
    """
    Every game 1999-present with spread_line, total_line (1999+) and
    moneylines (2006+), 100% populated.

    This is the benchmark, and it is free. Baseball spent five weeks
    without one. Do not spend Odds API credits on featured markets.
    """
    df = load_cached("schedules", ttl_hours=DEFAULT_TTL_HOURS)
    if df is None:
        df = save_cache("schedules", nfl.load_schedules())
    if through_season is not None:
        df = df.filter(pl.col("season") <= through_season)
    return df


def current_week(season: int = 2026) -> int:
    """
    The last week with a completed game. This is the week whose OUTCOMES are
    known; features for the upcoming slate are built `through_week=` this.
    """
    sch = load_schedules()
    done = sch.filter((pl.col("season") == season) & pl.col("result").is_not_null())
    if done.height == 0:
        return 0
    return int(done["week"].max())
