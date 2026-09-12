"""
Tests for data/market_lines.py.

These hit the network (nflverse schedules, cached after the first run).
The sign tests exist because the first version of implied_team_totals had
home and away inverted and every structural test still passed -- the two
columns reconcile to total_line whichever way round they are. A test that
only checks the sum cannot catch a swap. Check against the favourite.
"""
import numpy as np
import polars as pl
import pytest

from data.market_lines import (
    american_to_prob, devig_pair, game_lines, implied_team_totals,
)


# --- Pure functions, no network ---------------------------------------

def test_american_to_prob_even_money():
    assert american_to_prob([100])[0] == pytest.approx(0.5)
    assert american_to_prob([-100])[0] == pytest.approx(0.5)


def test_american_to_prob_favourite_and_dog():
    assert american_to_prob([-200])[0] == pytest.approx(2 / 3)
    assert american_to_prob([200])[0] == pytest.approx(1 / 3)


def test_devig_sums_to_one():
    p_home = american_to_prob([-150])
    p_away = american_to_prob([130])
    assert (p_home + p_away)[0] > 1.0, "raw implied probabilities carry vig"
    h, a = devig_pair(p_home, p_away)
    assert (h + a)[0] == pytest.approx(1.0)


def test_devig_preserves_ordering():
    h, a = devig_pair(american_to_prob([-300]), american_to_prob([250]))
    assert h[0] > a[0]


# --- Against real data -------------------------------------------------

@pytest.fixture(scope="module")
def lines():
    return implied_team_totals(seasons=[2023, 2024, 2025])


def test_implied_totals_sum_to_game_total(lines):
    err = (lines["home_implied_total"] + lines["away_implied_total"]
           - lines["total_line"]).abs().max()
    assert err == pytest.approx(0.0, abs=1e-9)


def test_home_favourite_gets_the_bigger_implied_total(lines):
    """
    THE SIGN TEST. spread_line > 0 means the home team is favoured, so the
    home implied total must be the larger of the two.
    """
    fav_home = lines.filter(pl.col("spread_line") > 3)
    assert fav_home.height > 50
    assert (fav_home["home_implied_total"] > fav_home["away_implied_total"]).all()

    fav_away = lines.filter(pl.col("spread_line") < -3)
    assert fav_away.height > 50
    assert (fav_away["away_implied_total"] > fav_away["home_implied_total"]).all()


def test_spread_sign_matches_actual_margins(lines):
    """
    The convention verified against outcomes, not documentation. Home teams
    favoured by more than 3 must win by a positive average margin.
    """
    played = lines.filter(pl.col("home_score").is_not_null())
    margin = (played["home_score"] - played["away_score"]).to_numpy()
    spread = played["spread_line"].to_numpy()
    assert np.corrcoef(spread, margin)[0, 1] > 0.3
    assert margin[spread > 3].mean() > 3
    assert margin[spread < -3].mean() < -3


def test_implied_totals_are_plausible(lines):
    """A team total outside 5-45 points means the arithmetic is wrong."""
    assert lines["home_implied_total"].min() > 5
    assert lines["home_implied_total"].max() < 45


def test_market_home_wp_beats_a_coin_flip(lines):
    """
    The market's de-vigged win probability must have real skill against
    actual results -- if it does not, the de-vig or the join is broken.
    """
    played = lines.filter(
        pl.col("market_home_wp").is_not_null() & pl.col("result").is_not_null()
    )
    y = (played["result"] > 0).to_numpy().astype(float)
    p = played["market_home_wp"].to_numpy()
    brier_market = np.mean((p - y) ** 2)
    brier_coin = np.mean((0.5 - y) ** 2)
    assert brier_market < brier_coin


def test_home_field_advantage_is_present(lines):
    """Sanity: home teams win more than half the time. ~53-54% historically."""
    played = lines.filter(pl.col("result").is_not_null())
    home_win_rate = float((played["result"] > 0).mean())
    assert 0.50 < home_win_rate < 0.60


def test_through_week_cuts():
    g = game_lines(seasons=[2025], through_week=4)
    assert g["week"].max() <= 4


def test_information_asymmetry_warning_fires_when_the_model_is_newer():
    """
    THE FAIR-TEST GUARD. Re-running the pipeline on Sunday re-predicts every
    game that has not kicked off, using Thursday's results and Friday's
    injury report -- while capture skips events already on disk, so those
    rows keep Wednesday's line.

    The model then holds information the price does not, and the edge it
    reports is partly just that gap. Same class of error as the
    clean/tainted split, one level up: there the risk was predicting after
    kickoff, here it is predicting after the price. Worth catching
    precisely because it flatters the model, which is the direction nobody
    checks.
    """
    from datetime import datetime, timedelta

    import compare_market as cm

    def frame(hours_later):
        return pl.DataFrame({
            "predicted_at": [datetime(2026, 9, 13, 12, 0, 0)] * 3,
            "captured_at": [
                (datetime(2026, 9, 13, 12, 0, 0)
                 - timedelta(hours=hours_later)).strftime(
                     "%Y-%m-%dT%H:%M:%S.%f+00:00")] * 3,
        })

    same_day = cm.warn_information_asymmetry(frame(2))
    assert abs(same_day - 2) < 0.1, "gap mis-measured for a fresh capture"

    stale = cm.warn_information_asymmetry(frame(96))
    assert abs(stale - 96) < 0.1, "gap mis-measured for a stale capture"

    # missing columns must not raise -- an older edges file has no
    # predicted_at and should degrade quietly rather than break the report
    assert np.isnan(cm.warn_information_asymmetry(
        pl.DataFrame({"captured_at": ["2026-09-13T12:00:00.0+00:00"]})))
