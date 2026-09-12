"""
Join sportsbook player names to nflverse player ids.

THE PROBLEM, AND WHY IT IS THE DANGEROUS KIND
---------------------------------------------
The books say "Marvin Harrison Jr."; nflverse says `00-0039337`. Nothing in
either feed carries the other's identifier, so the only bridge is the name.

A bad name join does not raise. It silently compares the model's number for
one player against the market's number for another, and the resulting
"edge" looks exactly like a real one. Every function here is built around
that: matching is scoped as narrowly as possible, every match records HOW it
was made, and unmatched names are returned rather than dropped.

    `match_players` never silently discards a name. If it cannot match one,
    it says so, and the caller decides.

SCOPING: TEAM FIRST, ALWAYS
---------------------------
Names are matched only against the two rosters in that game. This is not an
optimisation, it is the correctness argument. "Josh Allen" is a Buffalo
quarterback and a Jacksonville edge rusher; "Michael Carter" has been two
different players in the same season. Scoped to one game, those collisions
cannot occur. Unscoped, they are invisible.

The event's teams come from the Odds API events endpoint, which is FREE --
it does not count against the credit quota -- so scoping costs nothing.

THE LADDER
----------
Tried in order, most confident first, and the tier is recorded:

    exact     normalised strings identical
    initial   last name + first initial (D.J. Moore vs DJ Moore)
    fuzzy     difflib ratio >= 0.88, and only if the runner-up is clearly
              worse -- an ambiguous best match is reported as unmatched
              rather than guessed

The runner-up rule matters more than the threshold. Two brothers on one
roster will both score highly against a bare surname; the gap is what says
whether the winner is actually the right one.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from typing import Dict, Iterable, Optional, Tuple

import nflreadpy as nfl
import polars as pl

FUZZY_MIN = 0.88
FUZZY_GAP = 0.06     # best must beat runner-up by this much

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize_name(name: str) -> str:
    """
    Casefold, strip accents and punctuation, drop generational suffixes.

        "Marvin Harrison Jr."  -> "marvin harrison"
        "D.J. Moore"           -> "dj moore"
        "Amon-Ra St. Brown"    -> "amonra st brown"

    Suffixes are dropped because the books and nflverse disagree about them
    constantly, and they never distinguish two players on one roster.
    """
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[.’']", "", s)      # D.J. -> DJ, O'Neal -> ONeal
    s = re.sub(r"[^a-z0-9]+", " ", s)
    parts = [p for p in s.split() if p and p not in _SUFFIXES]
    return " ".join(parts)


def name_key(name: str) -> str:
    """Last name + first initial, the cheap disambiguator below exact match."""
    n = normalize_name(name)
    parts = n.split()
    if len(parts) < 2:
        return n
    return f"{parts[-1]} {parts[0][0]}"


def team_abbr_map() -> Dict[str, str]:
    """
    Full team name -> nflverse abbreviation, e.g. "Atlanta Falcons" -> "ATL".

    Built from `load_teams` rather than hardcoded, so relocations and
    rebrandings arrive with the data instead of needing an edit here.
    """
    t = nfl.load_teams()
    out: Dict[str, str] = {}
    for r in t.iter_rows(named=True):
        abbr = r.get("team_abbr")
        if not abbr:
            continue
        for field in ("team_name", "team_nick"):
            v = r.get(field)
            if v:
                out[normalize_name(v)] = abbr
        out[normalize_name(abbr)] = abbr
    return out


def event_teams(events: Iterable[dict],
                abbr: Optional[Dict[str, str]] = None) -> pl.DataFrame:
    """
    event_id -> (home_team, away_team) as nflverse abbreviations.

    Built from the FREE events endpoint, so team-scoped matching costs no
    credits. Unrecognised names are kept as nulls and reported rather than
    dropped -- a team that fails to map would otherwise silently remove a
    whole game from the join.
    """
    abbr = abbr if abbr is not None else team_abbr_map()
    rows = []
    for e in events:
        rows.append({
            "event_id": e.get("id"),
            "home_team": abbr.get(normalize_name(e.get("home_team", ""))),
            "away_team": abbr.get(normalize_name(e.get("away_team", ""))),
            "home_raw": e.get("home_team"),
            "away_raw": e.get("away_team"),
        })
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def _candidates(roster: pl.DataFrame, teams: Tuple[Optional[str], ...]) -> pl.DataFrame:
    teams = tuple(t for t in teams if t)
    if not teams:
        return roster
    return roster.filter(pl.col("team").is_in(list(teams)))


def _match_one(name: str, pool: pl.DataFrame) -> Tuple[Optional[dict], str, float]:
    """Returns (row, tier, score). Ambiguity yields (None, 'ambiguous', gap)."""
    if pool.is_empty():
        return None, "no_candidates", 0.0
    target = normalize_name(name)

    exact = pool.filter(pl.col("_norm") == target)
    if exact.height == 1:
        return exact.to_dicts()[0], "exact", 1.0
    if exact.height > 1:
        return None, "ambiguous_exact", 0.0

    key = name_key(name)
    byinit = pool.filter(pl.col("_key") == key)
    if byinit.height == 1:
        return byinit.to_dicts()[0], "initial", 0.95
    if byinit.height > 1:
        return None, "ambiguous_initial", 0.0

    names = pool["_norm"].to_list()
    scored = sorted(
        ((difflib.SequenceMatcher(None, target, n).ratio(), n) for n in names),
        reverse=True)
    if not scored:
        return None, "no_candidates", 0.0
    best, best_name = scored[0]
    runner = scored[1][0] if len(scored) > 1 else 0.0
    if best < FUZZY_MIN:
        return None, "below_threshold", best
    if best - runner < FUZZY_GAP:
        # Two plausible matches and no clear winner. Guessing here is how a
        # teammate's line gets compared to the wrong player's projection.
        return None, "ambiguous_fuzzy", best - runner
    row = pool.filter(pl.col("_norm") == best_name).to_dicts()[0]
    return row, "fuzzy", best


def match_players(market: pl.DataFrame, roster: pl.DataFrame,
                  events: Optional[pl.DataFrame] = None) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Attach player_id (and team) to market rows.

    market: needs `player`, and `event_id` if `events` is supplied.
    roster: needs `player_id`, `player_name`, `team`.
    events: event_id -> home_team/away_team, to scope each name to one game.

    Returns (matched, unmatched). BOTH are returned on purpose -- an
    unmatched name is information, not an error to swallow, and the match
    rate is the first thing to look at before trusting any edge.
    """
    for col in ("player_id", "player_name", "team"):
        if col not in roster.columns:
            raise ValueError(f"roster missing column {col!r}")
    if "player" not in market.columns:
        raise ValueError("market missing column 'player'")

    roster = roster.with_columns([
        pl.col("player_name").map_elements(normalize_name, return_dtype=pl.Utf8).alias("_norm"),
        pl.col("player_name").map_elements(name_key, return_dtype=pl.Utf8).alias("_key"),
    ]).unique(subset=["player_id", "team"])

    ev: Dict[str, Tuple] = {}
    if events is not None and not events.is_empty():
        for r in events.iter_rows(named=True):
            ev[r["event_id"]] = (r.get("home_team"), r.get("away_team"))

    matched, unmatched = [], []
    # Distinct names only -- 596 market rows are 157 players, and matching
    # each name once keeps this fast and the diagnostics readable.
    pairs = (market.select(["player", "event_id"]).unique()
             if "event_id" in market.columns
             else market.select("player").unique().with_columns(
                 pl.lit(None, dtype=pl.Utf8).alias("event_id")))

    for r in pairs.iter_rows(named=True):
        name, eid = r["player"], r.get("event_id")
        pool = _candidates(roster, ev.get(eid, ())) if eid in ev else roster
        row, tier, score = _match_one(name, pool)
        if row is None:
            unmatched.append({"player": name, "event_id": eid, "reason": tier,
                              "score": round(float(score), 3),
                              "n_candidates": pool.height})
        else:
            matched.append({"player": name, "event_id": eid,
                            "player_id": row["player_id"],
                            "matched_name": row["player_name"],
                            "team": row["team"], "match_tier": tier,
                            "match_score": round(float(score), 3)})

    m = pl.DataFrame(matched) if matched else pl.DataFrame()
    u = pl.DataFrame(unmatched) if unmatched else pl.DataFrame()
    if not m.is_empty():
        join_on = ["player", "event_id"] if "event_id" in market.columns else ["player"]
        m = market.join(m, on=join_on, how="inner")
    return m, u


def match_report(matched: pl.DataFrame, unmatched: pl.DataFrame,
                 n_names: Optional[int] = None) -> str:
    """One-paragraph summary. Print it before trusting any edge."""
    n_m = matched["player"].n_unique() if not matched.is_empty() else 0
    n_u = unmatched["player"].n_unique() if not unmatched.is_empty() else 0
    total = n_names or (n_m + n_u)
    lines = [f"matched {n_m}/{total} names ({n_m / total:.1%})" if total else "no names"]
    if not matched.is_empty() and "match_tier" in matched.columns:
        tiers = (matched.select(["player", "match_tier"]).unique()
                 .group_by("match_tier").len().sort("len", descending=True))
        lines.append("  by tier: " + ", ".join(
            f"{r['match_tier']} {r['len']}" for r in tiers.iter_rows(named=True)))
    if not unmatched.is_empty():
        reasons = unmatched.group_by("reason").len().sort("len", descending=True)
        lines.append("  unmatched: " + ", ".join(
            f"{r['reason']} {r['len']}" for r in reasons.iter_rows(named=True)))
        for r in unmatched.head(15).iter_rows(named=True):
            lines.append(f"    {r['player']}  ({r['reason']})")
    return "\n".join(lines)
