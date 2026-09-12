"""
The Odds API client: player props and closing lines.

Mirrors `baseball_predictor/data/odds_lines.py` in shape and key handling so
the two projects read alike.

THE CREDIT BUDGET, AND WHY IT IS NOT THE BINDING CONSTRAINT HERE
----------------------------------------------------------------
    /odds                     cost = markets x regions       (whole slate)
    /events/{id}/odds         cost = markets x regions       PER EVENT

Featured markets (h2h, spreads, totals) are whole-slate: 1 credit each.
Player props are per-event, so one market across a 16-game slate is 16
credits.

    1 prop market  x 1 region x 16 games  =   16 credits/week
    5 prop markets x 1 region x 16 games  =   80 credits/week
    a full season at 5 markets            = ~1,400 credits

Baseball spends ~30 credits a NIGHT on pitcher strikeouts alone. Football
is roughly a fifth of that per season, because there are ~18 slates instead
of ~180. **Credits are not the constraint here; sample size is.**

DO NOT SPEND CREDITS ON FEATURED MARKETS. `data/market_lines.py` reads
closing spreads and totals free from nflverse schedules back to 1999, and
moneylines back to 2006, 100% populated. Use this module for player props
and for live line movement only.

THE HISTORICAL TRAP
-------------------
Historical player props start 2023-05-03 and cost **10 credits per market
per region per event** -- ten times live. So every week that passes without
capturing props is a week that can only be bought back at 10x, or not at
all. That is the argument for starting the log before the model is
finished, not after.

THE KEY, AND WHY IT IS NOT THE SAME ONE AS BASEBALL
---------------------------------------------------
Each Odds API key carries its own credit quota. `baseball_predictor` uses
the shared `ODDS_API_KEY` variable and spends ~30 credits a night. If this
project silently fell back to that key, football would drain baseball's
quota and the "credits remaining" number printed here would describe a
budget that something else is also spending. You would not find out until a
pull failed mid-slate.

So the lookup is football-specific first, and the SOURCE is always reported:

    1. .env in this project root   <- PREFERRED. Football's by construction;
                                      no shell, no setx, nothing to collide.
    2. NFL_ODDS_API_KEY env var    <- football-specific by name
    3. .odds_api_key file          <- legacy, still read
    4. ODDS_API_KEY env var        <- SHARED. REFUSED for real spends; on
                                      this machine it is baseball's key.

A key named `ODDS_API_KEY` *inside this repo's .env* is football's and is
treated as such. Only the process-level variable of that name is suspect,
because only that one is shared with another project.

Recommended setup, once:

    Set-Content -Path .env -Value "NFL_ODDS_API_KEY=your-key" -Encoding utf8

Never `echo key > .env` -- PowerShell redirection produces UTF-16, which has
now broken two files in this project's lineage. The loader decodes it anyway,
but do not rely on that.

`.env` is gitignored. It does sync to OneDrive with the rest of the repo,
which keeps it off GitHub but is not the same as private; see data/env.py.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Optional, Sequence

import polars as pl

from data.env import env_path, load_env, project_root

SPORT = "americanfootball_nfl"
BASE = "https://api.the-odds-api.com/v4"
KEY_FILE = ".odds_api_key"
DEFAULT_REGIONS = "us"

# Market keys, with the prop they benchmark. Anything not here is either
# unsupported by the registry or not worth the credits yet.
KNOWN_MARKETS = {
    "player_reception_yds": "receiving_yards",
    "player_receptions": "receptions",
    "player_rush_yds": "rushing_yards",
    "player_pass_yds": "passing_yards",
    "player_anytime_td": "anytime_td",
}


# --- key handling ------------------------------------------------------

def _clean_key(raw: bytes) -> str:
    """
    Strip a BOM and whitespace. PowerShell's `>` writes UTF-16 with a BOM,
    which silently produces a key the API rejects as malformed -- the same
    encoding trap that made baseball's requirements.txt unreadable.
    """
    for enc in ("utf-8-sig", "utf-16", "utf-8"):
        try:
            s = raw.decode(enc).strip()
            if s and s.isprintable():
                return s
        except (UnicodeDecodeError, UnicodeError):
            continue
    return raw.decode("utf-8", errors="ignore").strip()


def _validate_key(key: str) -> str:
    if not key:
        raise RuntimeError("empty API key")
    if len(key) < 16 or " " in key:
        raise RuntimeError(
            f"API key looks malformed (length {len(key)}). If you wrote it with "
            f"PowerShell redirection it may be UTF-16; rewrite with:\n"
            f"    $env:NFL_ODDS_API_KEY='yourkey'")
    return key


NFL_ENV = "NFL_ODDS_API_KEY"
SHARED_ENV = "ODDS_API_KEY"


def resolve_api_key(root: Optional[str] = None) -> tuple:
    """
    Returns (key, source). Project-local sources first.

    The source is returned rather than swallowed so that every credit spend
    can say which quota it came out of. Two keys with separate quotas and no
    way to tell which one is in use is how you discover mid-slate that you
    have been spending the other project's budget.
    """
    root = root or project_root()
    dotenv = load_env(root)

    # 1. This repo's .env -- either name in here is football's.
    for name in (NFL_ENV, SHARED_ENV):
        v = (dotenv.get(name) or "").strip()
        if v:
            return _validate_key(v), f".env:{name}"

    # 2. Football-specific process variable.
    v = os.environ.get(NFL_ENV, "").strip()
    if v:
        return _validate_key(v), NFL_ENV

    # 3. Legacy single-purpose key file.
    path = os.path.join(root, KEY_FILE)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return _validate_key(_clean_key(f.read())), KEY_FILE

    # 4. The shared process variable. Found, reported, and refused for spends.
    v = os.environ.get(SHARED_ENV, "").strip()
    if v:
        return _validate_key(v), SHARED_ENV

    raise RuntimeError(
        f"No API key.\n"
        f"  Put it in {env_path(root)} :\n"
        f'    Set-Content -Path .env -Value "{NFL_ENV}=your-key" -Encoding utf8\n'
        f"  {SHARED_ENV} in the process environment would also be found, but "
        f"on this machine that is baseball_predictor's key and its own quota, "
        f"so it is refused for real spends.")


def load_api_key(root: Optional[str] = None, quiet: bool = False) -> str:
    key, source = resolve_api_key(root)
    if not quiet:
        print(f"  key from {source}")
    return key


class SharedKeyRefused(RuntimeError):
    """Raised rather than spending another project's credit quota."""


def require_own_key(root: Optional[str] = None, allow_shared_key: bool = False) -> str:
    """
    Return the key, REFUSING to proceed if it is the shared one.

    This started as a warning and that was wrong. A warning prints, scrolls
    past while you are reading the credit estimate underneath it, and the
    spend happens anyway -- which is exactly what it did on the first real
    capture, against baseball_predictor's quota.

    A cost that lands on the wrong budget is not recoverable: credits do not
    refund and the two projects cannot tell each other apart after the fact.
    So the dangerous path is now opt-IN. `allow_shared_key=True` is there for
    the case where you genuinely have one key and mean to share it.

    Dry runs are unaffected -- they spend nothing and never reach this.
    """
    key, source = resolve_api_key(root)
    if source == SHARED_ENV and not allow_shared_key:
        raise SharedKeyRefused(
            f"Refusing to spend credits on {SHARED_ENV}.\n"
            f"  That variable is baseball_predictor's key on this machine, and "
            f"its quota is shared, not duplicated.\n"
            f'  Fix:   Set-Content -Path .env -Value "{NFL_ENV}=your-key" '
            f"-Encoding utf8    (no new shell needed)\n"
            f"  Check: python -c \"from data.odds import resolve_api_key; "
            f"print(resolve_api_key()[1])\"\n"
            f"  If you really do mean to share one key, pass "
            f"allow_shared_key=True.")
    print(f"  key from {source}")
    return key


def check_quota(root: Optional[str] = None) -> dict:
    """
    Report remaining and used credits WITHOUT spending any.

    The events endpoint is free but still returns the quota headers, so this
    is the honest way to see where a key stands -- including after a spend
    that landed on the wrong key.
    """
    key, source = resolve_api_key(root)
    _, headers = _get(f"{BASE}/sports/{SPORT}/events?apiKey={key}")
    used = None
    for k in ("x-requests-used", "X-Requests-Used"):
        if k in headers:
            try:
                used = int(float(headers[k]))
            except ValueError:
                pass
    out = {"source": source, "remaining": credits_left(headers), "used": used}
    print(f"  key from {source}:  {out['remaining']} remaining, {out['used']} used")
    return out


# --- http --------------------------------------------------------------

def _get(url: str, retries: int = 2) -> tuple:
    """Returns (payload, headers). Headers carry the remaining credit count."""
    last = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.loads(r.read().decode("utf-8")), dict(r.headers)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")[:200]
            if e.code in (401, 422):
                raise RuntimeError(f"Odds API rejected the request ({e.code}): {body}")
            last = RuntimeError(f"HTTP {e.code}: {body}")
        except Exception as e:  # noqa: BLE001
            last = e
        if attempt < retries:
            time.sleep(1.5 * (attempt + 1))
    raise last


def credits_left(headers: dict) -> Optional[int]:
    for k in ("x-requests-remaining", "X-Requests-Remaining"):
        if k in headers:
            try:
                return int(float(headers[k]))
            except ValueError:
                pass
    return None


# --- odds maths --------------------------------------------------------

def american_to_prob(price) -> float:
    p = float(price)
    return -p / (-p + 100.0) if p < 0 else 100.0 / (p + 100.0)


def devig_two_way(over_prob: float, under_prob: float) -> float:
    """
    Proportional de-vig of a two-way market, returning the fair over
    probability.

    Comparing a model to the RAW implied probability credits the model with
    beating the hold, which it did not do. Same method and same caveat as
    `market_lines.devig_pair`: proportional de-vig assumes the hold is
    spread evenly, while books load more on the longshot. Shin or power
    de-vig correct for that; at typical prop holds the difference is small
    and using one method consistently matters more than which.
    """
    total = over_prob + under_prob
    return over_prob / total if total > 0 else float("nan")


# --- endpoints ---------------------------------------------------------

def list_events(api_key: str, days_ahead: Optional[int] = None) -> list:
    """
    Upcoming NFL events. 0 credits -- the events endpoint is free.

    `days_ahead` MATTERS AND DEFAULTS TO ONE WEEK. The endpoint returns every
    scheduled game it has odds for, which in September is most of the season:
    a live dry run returned **212 events**, so two markets would have cost
    **424 credits** instead of the ~32 one slate actually needs. Thirteen
    times the intended spend, on a quota that does not refund.

    Eight days covers a full week's five windows (Thursday through the
    following Monday) with a day of slack for timezone edges.
    """
    payload, _ = _get(f"{BASE}/sports/{SPORT}/events?apiKey={api_key}")
    if days_ahead is None:
        return payload
    cutoff = datetime.now(timezone.utc) + timedelta(days=days_ahead)
    out = []
    for e in payload:
        ct = e.get("commence_time")
        if not ct:
            continue
        try:
            when = datetime.fromisoformat(ct.replace("Z", "+00:00"))
        except ValueError:
            continue
        if when <= cutoff:
            out.append(e)
    return out


def fetch_event_props(api_key: str, event_id: str, markets: str,
                      regions: str = DEFAULT_REGIONS) -> tuple:
    url = (f"{BASE}/sports/{SPORT}/events/{event_id}/odds"
           f"?apiKey={api_key}&regions={regions}&markets={markets}"
           f"&oddsFormat=american")
    return _get(url)


def _rows_from_event(event: dict, payload: dict) -> list:
    """Flatten one event's prop payload into over/under rows per book."""
    rows = []
    captured = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for book in payload.get("bookmakers", []):
        for market in book.get("markets", []):
            for oc in market.get("outcomes", []):
                rows.append({
                    "event_id": event.get("id"),
                    "commence_time": event.get("commence_time"),
                    "home_team": event.get("home_team"),
                    "away_team": event.get("away_team"),
                    "book": book.get("key"),
                    "market": market.get("key"),
                    "player": oc.get("description"),
                    "side": oc.get("name"),
                    "line": oc.get("point"),
                    "price": oc.get("price"),
                    "captured_at": captured,
                })
    return rows


def consensus(rows: pl.DataFrame) -> pl.DataFrame:
    """
    Collapse per-book rows to one de-vigged fair probability per
    (event, market, player, line), averaged across books.

    Averaging fair probabilities across books rather than picking one is
    deliberate: a single book's number is noisier and occasionally stale,
    and the consensus is the closest free proxy for the true closing price.
    """
    if rows.is_empty():
        return rows

    over = rows.filter(pl.col("side").str.to_lowercase() == "over")
    under = rows.filter(pl.col("side").str.to_lowercase() == "under")
    keys = ["event_id", "market", "player", "line", "book"]

    j = over.join(under, on=keys, how="inner", suffix="_u")
    if j.is_empty():
        # One-way markets (anytime TD) have no Under; de-vig across the
        # whole book instead is not possible, so report raw implied.
        o = over.with_columns(
            pl.col("price").map_elements(american_to_prob, return_dtype=pl.Float64)
              .alias("fair_prob"))
        return (o.group_by(["event_id", "market", "player", "line"])
                 .agg([pl.col("fair_prob").mean().alias("market_prob"),
                       pl.len().alias("n_books"),
                       pl.col("home_team").first(), pl.col("away_team").first(),
                       pl.col("commence_time").first(),
                       pl.col("captured_at").first()])
                 .with_columns(pl.lit(False).alias("devigged")))

    j = j.with_columns([
        pl.col("price").map_elements(american_to_prob, return_dtype=pl.Float64).alias("p_o"),
        pl.col("price_u").map_elements(american_to_prob, return_dtype=pl.Float64).alias("p_u"),
    ]).with_columns(
        (pl.col("p_o") / (pl.col("p_o") + pl.col("p_u"))).alias("fair_prob"))

    # home_team/away_team are carried through so a capture is self-sufficient
    # for team-scoped name matching. The first version dropped them, and
    # scoping then needed a second call to the events endpoint.
    return (j.group_by(["event_id", "market", "player", "line"])
             .agg([pl.col("fair_prob").mean().alias("market_prob"),
                   pl.len().alias("n_books"),
                   (pl.col("p_o") + pl.col("p_u") - 1).mean().alias("hold"),
                   pl.col("home_team").first(), pl.col("away_team").first(),
                   pl.col("commence_time").first(),
                   pl.col("captured_at").first()])
             .with_columns(pl.lit(True).alias("devigged")))


MAX_CREDITS_DEFAULT = 60


def fetch_slate_props(markets: Optional[list] = None,
                      regions: str = DEFAULT_REGIONS,
                      dry_run: bool = False,
                      days_ahead: Optional[int] = None,
                      season: Optional[int] = None,
                      week: Optional[int] = None,
                      max_credits: int = MAX_CREDITS_DEFAULT,
                      allow_shared_key: bool = False,
                      skip_captured: bool = True) -> tuple:
    """
    Pull player props for the upcoming slate.

    dry_run=True lists the events and prints the credit cost WITHOUT
    spending anything, which is how you check a market key is right before
    paying to find out it is not.

    `max_credits` is a hard stop, not a warning. The first live dry run
    would have spent 424 credits because the events endpoint returned the
    rest of the season; a guard that has to be raised deliberately is
    cheaper than a quota that does not refund. Raise it when you mean to.

    Returns (rows, consensus, credits_remaining).
    """
    markets = markets or ["player_reception_yds", "player_receptions"]
    for m in markets:
        if m not in KNOWN_MARKETS:
            print(f"  note: '{m}' is not in KNOWN_MARKETS; proceeding anyway")

    # Dry runs resolve the key loosely (they spend nothing); real pulls
    # refuse the shared key outright.
    api_key = (load_api_key() if dry_run
               else require_own_key(allow_shared_key=allow_shared_key))
    all_events = list_events(api_key)

    # Scope to ONE WEEK by default, not a rolling day count. A day count
    # straddles weeks: 8 days from 2026-09-12 caught 14 remaining week-1
    # games plus week 2's Thursday opener, so the capture would have
    # belonged to two slates and joined cleanly to neither.
    if days_ahead is not None:
        events = list_events(api_key, days_ahead=days_ahead)
        scope = f"within {days_ahead} days"
    else:
        from data.nflverse import upcoming_week, week_window
        season = season or datetime.now(timezone.utc).year
        week = week or upcoming_week(season)
        lo, hi = week_window(season, week)
        if lo is None:
            events, scope = all_events, "unscoped (no schedule found)"
        else:
            lo_u = lo.replace(tzinfo=timezone.utc)
            hi_u = (hi + timedelta(hours=6)).replace(tzinfo=timezone.utc)
            events = []
            for e in all_events:
                ct = e.get("commence_time")
                if not ct:
                    continue
                try:
                    when = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                except ValueError:
                    continue
                if lo_u <= when <= hi_u:
                    events.append(e)
            scope = f"in {season} week {week}"

    # IDEMPOTENCE. Events whose lines are already on disk for every
    # requested market are dropped before the cost is computed, so a
    # mid-week re-run of the pipeline spends nothing on what it already
    # has. See captured_event_ids for why this skips rather than refreshes.
    skipped = 0
    if skip_captured:
        have = captured_event_ids(markets)
        if have:
            keep = [e for e in events if e.get("id") not in have]
            skipped = len(events) - len(keep)
            events = keep

    cost = len(events) * len(markets) * len(regions.split(","))
    print(f"  {len(all_events)} events scheduled, {len(events) + skipped} {scope}")
    if skipped:
        saved = skipped * len(markets) * len(regions.split(","))
        print(f"  {skipped} already captured -- skipping (~{saved} credits saved)")
    print(f"  {len(events)} events x {len(markets)} markets = ~{cost} credits")

    if not events:
        print("  nothing left to capture for this week")
        return pl.DataFrame(), pl.DataFrame(), None

    if dry_run:
        for e in events[:8]:
            print(f"    {e.get('commence_time')}  {e.get('away_team')} @ {e.get('home_team')}")
        if len(events) > 8:
            print(f"    ... and {len(events) - 8} more")
        return pl.DataFrame(), pl.DataFrame(), None

    if cost > max_credits:
        raise RuntimeError(
            f"Refusing to spend ~{cost} credits (cap {max_credits}).\n"
            f"  {len(events)} events {scope} x {len(markets)} markets.\n"
            f"  Narrow with season=/week=, drop a market, or raise "
            f"max_credits= if you mean it.")

    market_str = ",".join(markets)
    all_rows, remaining = [], None
    for e in events:
        try:
            payload, headers = fetch_event_props(api_key, e["id"], market_str, regions)
        except RuntimeError as err:
            print(f"    {e.get('id')}: {err}")
            continue
        all_rows.extend(_rows_from_event(e, payload))
        remaining = credits_left(headers) or remaining

    rows = pl.DataFrame(all_rows) if all_rows else pl.DataFrame()
    print(f"  {rows.height} rows, {remaining} credits remaining")
    return rows, consensus(rows), remaining


def captured_event_ids(markets: Sequence[str],
                      path: str = "cache/market_log.parquet") -> set:
    """
    Event ids already in the market log for EVERY market in `markets`.

    Used to make capture idempotent: re-running mid-week must not re-buy
    lines already on disk. An event counts as captured only if all the
    requested markets are present for it, so adding a market to the list
    correctly re-fetches the events that are missing it.

    WHY SKIP RATHER THAN REFRESH, which is not only about credits. The
    market log is the benchmark, and the benchmark has to mean the same
    thing every week. Re-pulling as kickoff approaches would give later
    weeks a closer-to-closing line than earlier ones, so a trend in the
    model-vs-market gap could be a trend in when the capture happened. The
    first capture of the week is a consistent comparator; a mixture of
    first and last captures is not.
    """
    import os
    if not os.path.exists(path):
        return set()
    log = pl.read_parquet(path)
    if log.is_empty() or "event_id" not in log.columns:
        return set()
    want = set(markets)
    have = (log.group_by("event_id")
               .agg(pl.col("market").unique().alias("mkts")))
    return {r["event_id"] for r in have.iter_rows(named=True)
            if want.issubset(set(r["mkts"]))}


def append_market_log(cons: pl.DataFrame, path: str = "cache/market_log.parquet") -> int:
    """
    Append to the running market log, de-duplicating on
    (event, market, player, line, captured_at).

    Append-only and never rewritten. In baseball this log is the reason the
    model can be argued about at all.
    """
    if cons.is_empty():
        return 0
    if os.path.exists(path):
        old = pl.read_parquet(path)
        cons = pl.concat([old, cons], how="diagonal_relaxed").unique(
            subset=["event_id", "market", "player", "line", "captured_at"], keep="last")
    cons.write_parquet(path)
    return cons.height
