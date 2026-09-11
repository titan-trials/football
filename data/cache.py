"""
Disk cache for nflverse pulls.

Same shape as baseball_predictor/data/cache.py -- cache_path / load_cached /
save_cache, keyed by a string, living in cache/ next to the code. Two
deliberate differences:

  1. Parquet, not CSV. nflverse play-by-play is 397 columns; a season is
     ~48k rows. CSV round-trips lose dtypes and cost seconds per read.
  2. A staleness check. Baseball's cache held Statcast history, which never
     changes once written. Several nflverse feeds are REVISED after the
     fact -- NFL stat corrections land midweek and PFR backfills snap
     counts -- so a cached in-season file has to be able to expire.
     Completed prior seasons never expire.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import polars as pl

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# In-season feeds get re-pulled after this long. Prior seasons are frozen.
DEFAULT_TTL_HOURS = 12.0


def cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, f"{key}.parquet")


def cache_age_hours(key: str) -> Optional[float]:
    """Hours since the cached file was written, or None if it is absent."""
    path = cache_path(key)
    if not os.path.exists(path):
        return None
    return (time.time() - os.path.getmtime(path)) / 3600.0


def load_cached(key: str, ttl_hours: Optional[float] = None) -> Optional[pl.DataFrame]:
    """
    Returns the cached frame, or None if it is absent or stale.

    ttl_hours=None means never expire -- use it for completed seasons.
    """
    path = cache_path(key)
    if not os.path.exists(path):
        return None
    if ttl_hours is not None:
        age = cache_age_hours(key)
        if age is not None and age > ttl_hours:
            return None
    return pl.read_parquet(path)


def save_cache(key: str, df: pl.DataFrame) -> pl.DataFrame:
    df.write_parquet(cache_path(key))
    print(f"  Cached {df.shape[0]} rows to cache/{key}.parquet")
    return df


def clear_cache(prefix: str = "") -> int:
    """Delete cached files whose key starts with `prefix`. Returns the count."""
    n = 0
    for name in os.listdir(CACHE_DIR):
        if name.endswith(".parquet") and name.startswith(prefix):
            os.remove(os.path.join(CACHE_DIR, name))
            n += 1
    return n
