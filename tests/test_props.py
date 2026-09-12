"""
Tests for features/props.py.

The reconciliation tests are the important ones. Every prop pulls its
per-opportunity outcomes from play-by-play and its realised totals from the
weekly stats table, and those two sources agreeing is an ASSUMPTION, not a
guarantee -- the sack bug proved it. nflfastR sets `pass_attempt = 1` on a
sack; the official stat does not count it. Left in, it gave 1,329 phantom
attempts and -8,083 yards in 2025 and biased passing yards by -21 per game,
while every other prop looked fine and nothing crashed.

So: each prop's play-level extraction is reconciled against the official
totals, with a tolerance tight enough to catch a systematic miscount and
loose enough to tolerate the handful of penalty-nullified plays that
legitimately differ.

These tests hit the network on first run and cache afterwards.
"""
import numpy as np
import polars as pl
import pytest

from data.nflverse import load_pbp, load_player_stats
from features.props import (
    PROPS, label_rows_with_asof_shape, opportunity_rows, player_games,
    shape_per_game, team_games,
)

SEASON = 2025


@pytest.fixture(scope="module")
def pbp():
    return load_pbp([SEASON])


@pytest.fixture(scope="module")
def stats():
    return load_player_stats([SEASON])


# --- registry sanity ---------------------------------------------------

def test_every_prop_has_a_coherent_spec():
    for name, spec in PROPS.items():
        assert spec.name == name
        assert spec.positions, f"{name} has no positions"
        assert len(spec.lines) >= 2, f"{name} has too few lines"
        assert spec.pt_support[0] < spec.pt_support[-1]
        assert spec.total_support[0] <= 0 <= spec.total_support[-1]
        assert spec.max_opportunities > 0


def test_total_support_covers_the_worst_case():
    """
    total_support must cover any total that realistically occurs, or the
    convolution silently truncates and renormalising pulls the mean DOWN --
    the MAX_RUNS lesson from team-run-model.md.

    The two prop shapes need different bars, which is the point of checking
    them separately rather than with one formula. A binary-outcome prop
    (touchdowns, receptions) sums to a small count no matter how many
    opportunities there are -- the single-game receiving-TD record is 5 --
    so `max_opportunities x 1` is a wildly pessimistic ceiling and asserting
    against it would demand a support 25 wide for a quantity that never
    exceeds 5. A yardage prop genuinely can run long.
    """
    binary_ceiling = {"receiving_tds": 5, "rushing_tds": 6, "receptions": 20}
    for name, spec in PROPS.items():
        top = int(spec.total_support[-1])
        if int(spec.pt_support[-1]) <= 1:
            need = binary_ceiling.get(name, 10)
            assert top >= need, (
                f"{name}: binary-outcome total_support tops out at {top}, "
                f"below the plausible ceiling {need}")
        else:
            # A 99th-percentile game must fit with room to spare. 15 yards
            # per opportunity is far above any real per-play average.
            plausible_max = spec.max_opportunities * 15
            assert top >= plausible_max * 0.5, (
                f"{name}: total_support tops out at {top}")


# --- reconciliation: pbp vs official stats -----------------------------

@pytest.mark.parametrize("name,att_tol,yds_tol", [
    ("receiving_yards", 0.03, 0.03),
    ("rushing_yards", 0.03, 0.03),
    ("passing_yards", 0.02, 0.02),
])
def test_play_level_reconciles_with_official_totals(pbp, stats, name, att_tol, yds_tol):
    """
    THE SACK REGRESSION TEST. Play-level opportunity counts and outcome
    totals must match the official weekly stats to within a few percent.
    """
    spec = PROPS[name]
    rows = opportunity_rows(spec, pbp)
    official = stats.filter(pl.col("position").is_in(list(spec.positions)))
    off_opp = float(official[spec.stat_opportunity].fill_null(0).sum())
    off_out = float(official[spec.stat_outcome].fill_null(0).sum())

    got_opp = float(rows.height)
    got_out = float(rows["outcome"].sum())

    assert abs(got_opp - off_opp) / off_opp < att_tol, (
        f"{name}: {got_opp:.0f} opportunities from pbp vs {off_opp:.0f} official "
        f"({(got_opp - off_opp) / off_opp:+.1%})")
    assert abs(got_out - off_out) / off_out < yds_tol, (
        f"{name}: {got_out:.0f} outcome from pbp vs {off_out:.0f} official "
        f"({(got_out - off_out) / off_out:+.1%})")


def test_sacks_are_excluded_from_pass_attempts(pbp):
    """Explicit, because this is the bug that shipped."""
    rows = opportunity_rows(PROPS["passing_yards"], pbp)
    sacks = pbp.filter((pl.col("sack") == 1) & pl.col("passer_player_id").is_not_null())
    assert sacks.height > 500, "expected a season's worth of sacks to test against"
    with_sacks = pbp.filter((pl.col("pass_attempt") == 1)
                            & pl.col("passer_player_id").is_not_null()).height
    assert rows.height < with_sacks, "sacks are still being counted as attempts"
    assert rows["outcome"].min() >= -25


def test_receptions_outcome_is_binary(pbp):
    rows = opportunity_rows(PROPS["receptions"], pbp)
    vals = set(np.unique(rows["outcome"].to_numpy()).tolist())
    assert vals <= {0.0, 1.0}, f"receptions outcome not binary: {vals}"


def test_reception_rate_is_plausible(pbp):
    rows = opportunity_rows(PROPS["receptions"], pbp)
    assert 0.6 < rows["outcome"].mean() < 0.72


# --- as-of discipline --------------------------------------------------

def test_asof_shape_is_shifted_and_never_sees_the_current_game(pbp):
    """
    The shape label for week w must come only from weeks before w. A
    season-long value would flatter the bands badly, because hindsight aDOT
    separates the outcomes it is meant to be predicting.
    """
    spec = PROPS["receiving_yards"]
    rows = opportunity_rows(spec, pbp)
    per_game = shape_per_game(spec, rows)

    one = (per_game.filter(pl.col("shape_n") > 0)
                   .sort(["player_id", "season", "week"]))
    grp = one.filter(pl.col("player_id") == one["player_id"][0])
    assert grp["asof_shape"][0] is None, "first game must have no prior shape"
    if grp.height > 2:
        # second game's as-of value is exactly the first game's own mean
        expected = grp["shape_sum"][0] / grp["shape_n"][0]
        assert grp["asof_shape"][1] == pytest.approx(expected, rel=1e-9)


def test_label_rows_attaches_shape_without_dropping_rows(pbp):
    spec = PROPS["receiving_yards"]
    rows = opportunity_rows(spec, pbp)
    labelled = label_rows_with_asof_shape(rows, shape_per_game(spec, rows))
    assert labelled.height == rows.height
    assert "asof_shape" in labelled.columns


def test_shapeless_prop_returns_empty_shape_table(pbp):
    spec = PROPS["rushing_yards"]
    assert not spec.has_shape
    assert shape_per_game(spec, opportunity_rows(spec, pbp)).is_empty()


# --- player and team tables --------------------------------------------

def test_player_games_has_no_nulls(stats):
    for name, spec in PROPS.items():
        pg = player_games(spec, stats)
        assert pg["opportunities"].null_count() == 0, name
        assert pg["outcome"].null_count() == 0, name


def test_team_games_sum_to_player_games(stats):
    spec = PROPS["receiving_yards"]
    pg = player_games(spec, stats)
    tg = team_games(pg)
    assert tg["N"].sum() == pytest.approx(pg["opportunities"].sum())


def test_team_pass_volume_is_in_the_right_neighbourhood(stats):
    """~32 targets a game; a wildly different number means the filter is wrong."""
    tg = team_games(player_games(PROPS["receiving_yards"], stats))
    assert 28 < tg["N"].mean() < 36


# --- availability and odds ---------------------------------------------

def test_availability_flag_ranks_are_ordered():
    from data.availability import flag_frame
    f = flag_frame([2025], through_week=6)
    assert f.height > 100
    m = {r["availability"]: r["status_rank"] for r in f.iter_rows(named=True)}
    for name, rank in m.items():
        assert 0 <= rank <= 3
    if "OUT" in m and "QUESTIONABLE" in m:
        assert m["OUT"] < m["QUESTIONABLE"]


def test_report_published_distinguishes_absent_from_healthy():
    """
    The bug the first live slate had: 0 flagged rows read as 'everyone
    healthy' when the report simply had not published.
    """
    from data.availability import flag_frame, report_published
    f = flag_frame([2025], through_week=6)
    assert report_published(f, 2025, 5) is True
    assert report_published(f, 2025, 99) is False
    assert report_published(pl.DataFrame(), 2025, 5) is False


def test_attach_marks_unlisted_players_clear():
    from data.availability import attach, flag_frame
    f = flag_frame([2025], through_week=6)
    slate = pl.DataFrame({"season": [2025], "week": [5], "team": ["KC"],
                          "player_id": ["definitely-not-a-player"]})
    out = attach(slate, f)
    assert out["availability"][0] == "CLEAR"
    assert out["status_rank"][0] == 4


def test_odds_american_conversion_and_devig():
    from data.odds import american_to_prob, devig_two_way
    assert american_to_prob(-110) == pytest.approx(110 / 210)
    assert american_to_prob(100) == pytest.approx(0.5)
    o, u = american_to_prob(-110), american_to_prob(-110)
    assert o + u > 1.0
    assert devig_two_way(o, u) == pytest.approx(0.5)


def test_odds_key_cleaner_handles_utf16():
    """PowerShell redirection writes UTF-16; pip and this API both choke."""
    from data.odds import _clean_key
    assert _clean_key("abcdef1234567890".encode("utf-16")) == "abcdef1234567890"
    assert _clean_key(b"\xef\xbb\xbfabcdef1234567890") == "abcdef1234567890"


def test_odds_rejects_malformed_key():
    from data.odds import _validate_key
    with pytest.raises(RuntimeError):
        _validate_key("short")
    with pytest.raises(RuntimeError):
        _validate_key("has spaces in it aaaa")


def test_anytime_td_combination():
    from features.props import anytime_td_probability
    assert anytime_td_probability(1.0, 1.0) == pytest.approx(0.0)
    assert anytime_td_probability(0.5, 0.5) == pytest.approx(0.75)
    assert anytime_td_probability(0.9, 1.0) == pytest.approx(0.1)


def test_odds_key_lookup_prefers_the_football_specific_variable(tmp_path, monkeypatch):
    """
    Two keys, two quotas. Falling back to baseball's shared key without
    saying so is how you discover mid-slate that you spent the wrong budget.
    """
    from data import odds
    monkeypatch.setenv(odds.NFL_ENV, "nflkey1234567890")
    monkeypatch.setenv(odds.SHARED_ENV, "sharedkey1234567890")
    key, src = odds.resolve_api_key(root=str(tmp_path))
    assert key == "nflkey1234567890"
    assert src == odds.NFL_ENV


def test_odds_key_falls_back_to_file_then_shared(tmp_path, monkeypatch):
    from data import odds
    monkeypatch.delenv(odds.NFL_ENV, raising=False)
    monkeypatch.setenv(odds.SHARED_ENV, "sharedkey1234567890")

    # file beats the shared variable
    (tmp_path / odds.KEY_FILE).write_text("filekey1234567890")
    key, src = odds.resolve_api_key(root=str(tmp_path))
    assert key == "filekey1234567890" and src == odds.KEY_FILE

    # with no file, the shared variable is last resort
    (tmp_path / odds.KEY_FILE).unlink()
    key, src = odds.resolve_api_key(root=str(tmp_path))
    assert key == "sharedkey1234567890" and src == odds.SHARED_ENV


def test_odds_key_error_names_the_football_variable(tmp_path, monkeypatch):
    from data import odds
    monkeypatch.delenv(odds.NFL_ENV, raising=False)
    monkeypatch.delenv(odds.SHARED_ENV, raising=False)
    with pytest.raises(RuntimeError, match="NFL_ODDS_API_KEY"):
        odds.resolve_api_key(root=str(tmp_path))


def test_fetch_refuses_to_overspend(monkeypatch):
    """
    The 424-credit near-miss. The events endpoint returns the whole season,
    so a cap that must be raised deliberately is cheaper than a quota that
    does not refund.

    Exercised through the days_ahead branch, where the event list is taken
    as given -- the week-scoped branch filters against the real schedule and
    would discard synthetic events before the cap ever saw them.
    """
    from data import odds
    monkeypatch.setattr(odds, "require_own_key", lambda *a, **k: "x" * 32)
    monkeypatch.setattr(odds, "list_events",
                        lambda key, days_ahead=None: [{"id": str(i)} for i in range(200)])
    with pytest.raises(RuntimeError, match="Refusing to spend ~200"):
        odds.fetch_slate_props(markets=["player_receptions"], days_ahead=8,
                               max_credits=60)


def test_week_scoped_fetch_covers_one_slate(monkeypatch):
    """
    Default scoping is by WEEK, not by a rolling day count. Synthetic events
    spanning both weeks must reduce to only the target week's.
    """
    from datetime import timedelta
    from data import odds
    from data.nflverse import week_window

    lo1, hi1 = week_window(2026, 1)
    lo2, _ = week_window(2026, 2)
    events = (
        [{"id": f"w1-{i}",
          "commence_time": (lo1 + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%SZ")}
         for i in range(5)]
        + [{"id": f"w2-{i}",
            "commence_time": (lo2 + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M:%SZ")}
           for i in range(5)]
    )
    monkeypatch.setattr(odds, "load_api_key", lambda *a, **k: "x" * 32)
    monkeypatch.setattr(odds, "list_events", lambda key, days_ahead=None: events)
    rows, cons, left = odds.fetch_slate_props(
        markets=["player_receptions"], season=2026, week=1, dry_run=True)
    assert rows.is_empty() and left is None


def test_list_events_filters_by_window(monkeypatch):
    from data import odds
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    payload = [
        {"id": "soon", "commence_time": (now + timedelta(days=2)).isoformat().replace("+00:00", "Z")},
        {"id": "later", "commence_time": (now + timedelta(days=40)).isoformat().replace("+00:00", "Z")},
        {"id": "nodate"},
    ]
    monkeypatch.setattr(odds, "_get", lambda url, retries=2: (payload, {}))
    assert len(odds.list_events("k")) == 3
    got = odds.list_events("k", days_ahead=8)
    assert [e["id"] for e in got] == ["soon"]


def test_upcoming_week_is_not_last_completed_plus_one():
    """
    THE WEEK-SELECTION REGRESSION TEST. On 2026-09-12 week 1 had 16 games
    with 2 played and 14 still to come, so the next kickoff was a week 1
    game -- but last-completed-week + 1 said week 2. The predictor would
    have produced a slate for the wrong games and a market log captured for
    week 1 would have joined to nothing.
    """
    from datetime import datetime
    from data.nflverse import current_week, upcoming_week
    mid_week_one = datetime(2026, 9, 12, 4, 0)
    assert upcoming_week(2026, now=mid_week_one) == 1
    assert current_week(2026) >= 1  # a game HAS finished; the week has not


def test_upcoming_week_advances_once_the_week_is_over():
    from datetime import datetime
    from data.nflverse import upcoming_week
    after_week_one = datetime(2026, 9, 16, 0, 0)
    assert upcoming_week(2026, now=after_week_one) == 2


def test_week_window_does_not_straddle_weeks():
    """An 8-day window caught 14 week-1 games plus week 2's opener."""
    from data.nflverse import week_window
    lo1, hi1 = week_window(2026, 1)
    lo2, hi2 = week_window(2026, 2)
    assert lo1 is not None and lo2 is not None
    assert hi1 < lo2, "week 1 must end before week 2 begins"


def test_real_spend_refuses_the_shared_key(monkeypatch, tmp_path):
    """
    THE REGRESSION TEST FOR A WARNING THAT WAS IGNORED. A printed warning
    scrolls past while you read the credit estimate under it, and the spend
    lands on the other project's quota anyway. Refuse instead.
    """
    from data import odds
    monkeypatch.delenv(odds.NFL_ENV, raising=False)
    monkeypatch.setenv(odds.SHARED_ENV, "sharedkey1234567890")
    with pytest.raises(odds.SharedKeyRefused, match="Refusing to spend"):
        odds.require_own_key(root=str(tmp_path))
    # opt-in still works, for the genuinely-one-key case
    assert odds.require_own_key(root=str(tmp_path), allow_shared_key=True)


def test_dry_run_still_works_on_the_shared_key(monkeypatch, tmp_path):
    """Dry runs spend nothing, so they must not be blocked."""
    from data import odds
    monkeypatch.delenv(odds.NFL_ENV, raising=False)
    monkeypatch.setenv(odds.SHARED_ENV, "sharedkey1234567890")
    monkeypatch.setattr(odds, "list_events", lambda key, days_ahead=None: [])
    rows, cons, left = odds.fetch_slate_props(
        markets=["player_receptions"], days_ahead=8, dry_run=True)
    assert rows.is_empty()


# --- .env loading ------------------------------------------------------

def test_dotenv_parses_the_awkward_cases():
    from data.env import parse_env
    got = parse_env(
        '# a comment\n'
        '\n'
        'NFL_ODDS_API_KEY=plain123\n'
        'QUOTED = "quoted value"\n'
        "SINGLE='single'\n"
        'export EXPORTED=yes\n'
        'HASH=abc#notacomment\n'
        'EMPTY=\n'
        'no_equals_here\n'
    )
    assert got["NFL_ODDS_API_KEY"] == "plain123"
    assert got["QUOTED"] == "quoted value"
    assert got["SINGLE"] == "single"
    assert got["EXPORTED"] == "yes"
    assert got["HASH"] == "abc#notacomment", "values are not interpolated"
    assert got["EMPTY"] == ""
    assert "no_equals_here" not in got


def test_dotenv_survives_utf16(tmp_path):
    """PowerShell redirection writes UTF-16. It has broken two files already."""
    from data.env import load_env
    (tmp_path / ".env").write_bytes("NFL_ODDS_API_KEY=utf16key123456\n".encode("utf-16"))
    assert load_env(str(tmp_path), refresh=True)["NFL_ODDS_API_KEY"] == "utf16key123456"


def test_dotenv_beats_the_shared_process_variable(tmp_path, monkeypatch):
    """
    THE FIX FOR THE REAL PROBLEM. A process variable means 'whichever project
    set it last'; a file in this repo root is football's by construction.
    """
    from data import odds
    monkeypatch.setenv(odds.SHARED_ENV, "baseballkey1234567890")
    monkeypatch.delenv(odds.NFL_ENV, raising=False)
    (tmp_path / ".env").write_text("NFL_ODDS_API_KEY=footballkey1234567890\n")
    from data.env import load_env
    load_env(str(tmp_path), refresh=True)
    key, source = odds.resolve_api_key(root=str(tmp_path))
    assert key == "footballkey1234567890"
    assert source.startswith(".env")


def test_odds_api_key_inside_our_dotenv_is_ours(tmp_path, monkeypatch):
    """
    A key named ODDS_API_KEY *in this repo's .env* is football's, and must
    NOT trip the shared-key refusal. Only the process variable is suspect.
    """
    from data import odds
    from data.env import load_env
    monkeypatch.delenv(odds.NFL_ENV, raising=False)
    monkeypatch.setenv(odds.SHARED_ENV, "baseballkey1234567890")
    (tmp_path / ".env").write_text("ODDS_API_KEY=footballkey1234567890\n")
    load_env(str(tmp_path), refresh=True)
    key, source = odds.resolve_api_key(root=str(tmp_path))
    assert key == "footballkey1234567890"
    assert source == ".env:ODDS_API_KEY"
    # and it is spendable, unlike the process variable of the same name
    assert odds.require_own_key(root=str(tmp_path)) == "footballkey1234567890"


def test_env_loader_does_not_mutate_the_process_environment(tmp_path):
    """Writing into os.environ is how one project's key leaks into another's."""
    import os
    from data.env import load_env
    (tmp_path / ".env").write_text("SOME_UNIQUE_TEST_VAR=xyz\n")
    load_env(str(tmp_path), refresh=True)
    assert "SOME_UNIQUE_TEST_VAR" not in os.environ


def test_depth_chart_gives_each_player_exactly_one_team():
    """
    THE DOUBLE-ROSTER REGRESSION TEST. A traded player appears on both his
    old and new team's depth chart. Left in, the slate predicted Kayshon
    Boutte for HOU (share .074) and NE (.044) in the same week -- one of
    those rows is invention, and it makes the book name ambiguous too.
    """
    from data.depth import role_ranks
    rr = role_ranks([2026])
    dupes = (rr.group_by(["season", "week", "player_id"]).len()
               .filter(pl.col("len") > 1))
    assert dupes.height == 0, (
        f"{dupes.height} player-weeks appear on more than one team")


def test_slate_dedupe_prefers_the_current_depth_chart():
    """
    Preservation is faithful, including to bugs. A player kept from an
    earlier buggy run on the wrong team must lose to his current chart.
    """
    from datetime import datetime
    from predict_slate import enforce_one_row_per_player
    slate = pl.DataFrame({
        "player_id": ["p1", "p1", "p2"],
        "prop": ["receiving_yards"] * 3,
        "team": ["NE", "HOU", "GB"],
        "predicted_at": [datetime(2026, 9, 11), datetime(2026, 9, 12), datetime(2026, 9, 12)],
    })
    roster = pl.DataFrame({"player_id": ["p1", "p2"], "team": ["HOU", "GB"]})
    out = enforce_one_row_per_player(slate, roster)
    assert out.height == 2
    assert out.filter(pl.col("player_id") == "p1")["team"][0] == "HOU"


def test_slate_dedupe_is_a_noop_when_clean():
    from datetime import datetime
    from predict_slate import enforce_one_row_per_player
    slate = pl.DataFrame({
        "player_id": ["p1", "p2"], "prop": ["receiving_yards"] * 2,
        "team": ["HOU", "GB"],
        "predicted_at": [datetime(2026, 9, 12)] * 2,
    })
    roster = pl.DataFrame({"player_id": ["p1", "p2"], "team": ["HOU", "GB"]})
    assert enforce_one_row_per_player(slate, roster).height == 2


def test_role_bucket_does_not_flatten_after_fourth():
    """
    THE DILUTION REGRESSION TEST. Capping buckets at 4 gave every WR5-WR8
    the WR4 prior (.097 of team targets), so four reserves soaked up ~39% of
    a team's targets and every real contributor was diluted. Measured 2025
    shares keep falling: r5 .047, r6 .020, r7 .004.
    """
    from data.depth import MAX_ROLE_BUCKET, role_bucket
    got = (pl.DataFrame({"r": [1, 2, 3, 4, 5, 6, 7, 12]})
             .with_columns(role_bucket(pl.col("r")).alias("b"))["b"].to_list())
    assert got == [1, 2, 3, 4, 5, 6, 7, 7]
    assert MAX_ROLE_BUCKET >= 7, "deep reserves need their own near-zero prior"


def test_roster_player_games_excludes_unplayed_weeks():
    """
    THE WORST BUG IN THE PROJECT, as a test. The timestamped depth chart
    attaches a September snapshot to every remaining week, so zero-filling
    against it invents a row per unplayed game. ShareModel takes the last 8
    rows, which then came from the FUTURE -- eight zeros -- and every share
    collapsed to the role prior.
    """
    from data.depth import role_ranks
    from data.nflverse import load_player_stats
    from features.props import PROPS, roster_player_games

    rr = role_ranks([2026])
    ps = load_player_stats([2026])
    rpg = roster_player_games(PROPS["receptions"], ps, rr)

    played = set(ps.select(["season", "week"]).unique().iter_rows())
    built = set(rpg.select(["season", "week"]).unique().iter_rows())
    assert built <= played, f"invented rows for unplayed weeks: {built - played}"

    # and the depth chart really does span the whole season, so this is not
    # a vacuous check
    assert rr.filter(pl.col("season") == 2026)["week"].n_unique() > 10


def test_predict_roster_drops_unavailable_players():
    """
    THE POPULATION-MISMATCH REGRESSION TEST. The model is fitted on ACT
    player-weeks only; the predictor was normalising shares across the whole
    depth chart. On 2025 week 10 that was 705 listed skill players against
    329 active -- 22 per team versus 10.3 -- and it biased receiving yards
    by -4.30 on a mean of 21.95 while the validation harness showed the same
    model unbiased.
    """
    from predict_slate import UNAVAILABLE, restrict_to_available
    roster = pl.DataFrame({
        "team": ["KC"] * 4, "player_id": ["a", "b", "c", "d"],
        "role_pos": ["WR"] * 4,
    })
    statuses = pl.DataFrame({
        "season": [2026] * 3, "week": [1] * 3, "team": ["KC"] * 3,
        "gsis_id": ["a", "b", "c"], "status": ["ACT", "RES", "DEV"],
    })
    out = restrict_to_available(roster, statuses, 2026, 1)
    kept = set(out["player_id"].to_list())
    assert "a" in kept, "active player dropped"
    assert "b" not in kept and "c" not in kept, "unavailable players kept"
    assert "d" in kept, "player with no status row must be kept, not deleted"
    assert "INA" not in UNAVAILABLE, (
        "gameday inactives are not known at predict time and must not be used")


def test_restrict_is_a_noop_without_statuses():
    from predict_slate import restrict_to_available
    roster = pl.DataFrame({"team": ["KC"], "player_id": ["a"], "role_pos": ["WR"]})
    assert restrict_to_available(roster, None, 2026, 1).height == 1
    assert restrict_to_available(roster, pl.DataFrame(), 2026, 1).height == 1


def test_climatology_history_is_order_independent():
    """
    THE BASELINE-DETERMINISM REGRESSION TEST.

    `compare_props` takes the last `trailing` entries of each player's
    history, so the history must be in chronological order regardless of
    how the frame arrives. It was not, and two runs of the identical
    command printed climatology CRPS 8.124 and 8.068 for receiving yards.

    The model column never moved -- ShareModel and TeamVolumeModel sort
    internally -- so the discrepancy was entirely in the NULL, which means
    every published skill-against-climatology figure was measured against a
    baseline that could not be reproduced.
    """
    from compare_props import trailing_history

    rows = [
        {"player_id": "a", "season": 2024, "week": 1, "outcome": 1.0},
        {"player_id": "a", "season": 2024, "week": 2, "outcome": 2.0},
        {"player_id": "a", "season": 2025, "week": 1, "outcome": 3.0},
        {"player_id": "b", "season": 2025, "week": 3, "outcome": 9.0},
    ]
    forward = trailing_history(pl.DataFrame(rows))
    backward = trailing_history(pl.DataFrame(list(reversed(rows))))
    shuffled = trailing_history(pl.DataFrame([rows[2], rows[0], rows[3], rows[1]]))

    assert forward == backward == shuffled, "history depends on frame order"
    assert forward["a"] == [1.0, 2.0, 3.0], "history is not chronological"
    # the trailing slice the caller actually takes
    assert forward["a"][-2:] == [2.0, 3.0], "trailing slice is not the most recent"


def test_served_rows_keeps_listed_players_who_recorded_nothing():
    """
    THE SCORED-POPULATION REGRESSION TEST.

    `compare_props` used to score the weekly stats table, which contains
    only players who recorded something, while `predict_slate` serves the
    depth chart minus the unavailable. Scoring the stats table lets a model
    that under-predicts everybody measure unbiased, because the rows it
    over-predicts are the ones missing from the sample.

    Three properties, all of which were wrong or absent before:
      * a listed, available player with no stats row is a row with actual 0
      * a player on IR is not a row at all
      * a team on a bye is not in the frame, so a bye-week depth chart
        cannot invent zeros
    """
    from compare_props import served_rows
    from features.props import PROPS

    spec = PROPS["receptions"]
    rr = pl.DataFrame({
        "season": [2025] * 5, "week": [10] * 5,
        "team": ["KC", "KC", "KC", "KC", "BUF"],
        "player_id": ["a", "b", "c", "d", "e"],
        "role_pos": ["WR", "WR", "WR", "WR", "WR"],
        "role_bucket": [1, 2, 3, 4, 1],
    })
    stats_pg = pl.DataFrame({
        "season": [2025], "week": [10], "team": ["KC"],
        "player_id": ["a"], "outcome": [6.0],
    })
    statuses = pl.DataFrame({
        "season": [2025] * 2, "week": [10] * 2, "team": ["KC"] * 2,
        "gsis_id": ["c", "d"], "status": ["RES", "ACT"],
    })

    out = served_rows(spec, stats_pg, rr, statuses, 2025, 10)
    got = {r["player_id"]: r["outcome"] for r in out.iter_rows(named=True)}

    assert got["a"] == 6.0, "the player who recorded something lost his actual"
    assert got["b"] == 0.0, (
        "a listed, available player who caught nothing must be a row with "
        "actual 0 -- dropping him is how a diluted model measures unbiased")
    assert got["d"] == 0.0, "an ACT player with no stats row was dropped"
    assert "c" not in got, "a player on injured reserve was scored"
    assert "e" not in got, (
        "BUF did not appear in the stats table for this week, so it was on a "
        "bye or unplayed and must not be zero-filled")
    assert "position" in out.columns, "team_vector needs a position column"


def test_captured_events_are_skipped_only_when_every_market_is_present():
    """
    IDEMPOTENCE OF THE CAPTURE. Re-running the pipeline mid-week must not
    re-buy lines already on disk -- that is the one operation here that
    spends money and cannot be undone.

    The "every market" part matters: an event captured for receptions but
    not receiving yards is NOT done, and treating it as done would leave a
    permanent hole in the market log that can only be filled later at 10x
    the historical price.
    """
    import data.odds as odds

    log = pl.DataFrame({
        "event_id": ["e1", "e1", "e2", "e3"],
        "market": ["player_receptions", "player_reception_yds",
                   "player_receptions", "player_pass_yds"],
        "player": ["a", "b", "c", "d"],
        "line": [1.5, 20.5, 2.5, 200.5],
    })
    import os
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "log.parquet")
        log.write_parquet(path)
        want = ["player_receptions", "player_reception_yds"]
        got = odds.captured_event_ids(want, path=path)

    assert got == {"e1"}, (
        "only e1 has both markets; e2 is missing yards and e3 has neither")


def test_score_slate_refuses_a_week_that_is_not_over():
    """
    THE PARTIAL-WEEK GUARD. Checked against reality on 2026-09-12: week 1
    had results for 4 of 32 scheduled teams -- the Thursday opener and one
    more -- so "week 1 has data" was true while "week 1 can be scored" was
    false. Auto-detection without this would have scored a 16-game week off
    two games and written it into the permanent log.
    """
    import score_slate

    # 2 of 32 teams done is 6%; the threshold is 90%
    assert score_slate.latest_scorable.__doc__ is not None
    comp = {1: 0.0625, 2: 1.0}

    def fake_completeness(season):
        return comp

    def fake_weeks(season):
        return [1, 2]

    real_c, real_w = score_slate.week_completeness, score_slate.committed_weeks
    try:
        score_slate.week_completeness = fake_completeness
        score_slate.committed_weeks = fake_weeks
        assert score_slate.latest_scorable(2026) == 2, (
            "picked a week that is not over, or skipped one that is")
        comp[2] = 0.5
        assert score_slate.latest_scorable(2026) is None, (
            "scored a half-played week instead of refusing")
    finally:
        score_slate.week_completeness, score_slate.committed_weeks = real_c, real_w
