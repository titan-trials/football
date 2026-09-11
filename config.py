"""
Shared configuration for the NFL prop model.

Mirrors baseball_predictor/config.py in role. The important difference is
AVAILABILITY: baseball's feeds all landed before first pitch, so there was
no reason for a config to say which ones. Football's do not. See
SERVE_TIME_FEEDS below -- that list is load-bearing, not documentation.
"""

# --- Seasons -----------------------------------------------------------
# nflverse play-by-play starts 1999. Participation starts 2016 and route
# data is only complete from 2023. Training on 2019+ keeps the passing-era
# drift manageable without throwing away most of the sample.
FIRST_SEASON = 2019
CURRENT_SEASON = 2026
TRAIN_SEASONS = list(range(FIRST_SEASON, CURRENT_SEASON))

# --- Prop targets ------------------------------------------------------
# Stage 1 quantity (the opportunity) -> Stage 2 quantity (per-opportunity).
# The measured reliabilities that justify this split are in CONTEXT.md.
PROP_TARGETS = {
    "receiving_yards": ("targets", "yards_per_target"),
    "receptions":      ("targets", "catch_rate"),
    "rushing_yards":   ("carries", "yards_per_carry"),
    "passing_yards":   ("attempts", "yards_per_attempt"),
    "passing_tds":     ("attempts", "td_per_attempt"),
}

# Positions that draw enough opportunity to model at all.
SKILL_POSITIONS = ("QB", "RB", "WR", "TE", "FB")

# --- Availability tiers ------------------------------------------------
# Verified 2026-09-10 by calling each loader. A feed in SERVE_TIME_FEEDS is
# usable for the upcoming slate; a feed in POSTSEASON_ONLY_FEEDS is NOT and
# may only ever appear in a training set if the model also has a way to
# predict it from serve-time feeds.
#
# Building a feature on a POSTSEASON_ONLY feed and discovering in week 3
# that it cannot be served is exactly the train/serve skew that broke
# _project_pa in baseball. The check is enforced in data/nflverse.py.
SERVE_TIME_FEEDS = (
    "pbp",              # nightly after games
    "player_stats",     # nightly after games
    "schedules",        # continuous, carries closing lines
    "injuries",         # weekly, pre-game -- the whole point
    "depth_charts",     # daily 07:00 UTC
    "rosters_weekly",   # daily 07:00 UTC
    "snap_counts",      # 0/6/12/18 UTC, PFR-dependent, week N-1
    "nextgen_stats",    # nightly 3-5am ET, week N-1
    "ftn_charting",     # 0/6/12/18 UTC, week N-1
)

POSTSEASON_ONLY_FEEDS = (
    "participation",    # routes, coverage, personnel -- lands after the
                        # postseason completes. NOT servable in-season.
    "ff_opportunity",   # rebuilt per season
)

# --- Shrinkage ---------------------------------------------------------
# Trailing window for the league prior. The baseball era-drift finding was
# +6.8% on all-history versus 12 months; passing volume drifts at least as
# fast, so the prior is trailing, never full-cache.
PRIOR_TRAILING_SEASONS = 3

# Minimum opportunities before a player's own rate is allowed to move his
# estimate at all. Below this he is the position/role baseline.
MIN_OPPORTUNITIES = 20

# --- Market ------------------------------------------------------------
# Featured markets (spread/total/ML) come free from load_schedules back to
# 1999. Only player props ever cost credits.
ODDS_API_REGION = "us"
PROP_MARKETS = (
    "player_pass_yds",
    "player_rush_yds",
    "player_reception_yds",
    "player_receptions",
    "player_anytime_td",
)

# --- Teams -------------------------------------------------------------
ALL_TEAMS = (
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN",
    "DET", "GB", "HOU", "IND", "JAX", "KC", "LA", "LAC", "LV", "MIA",
    "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB",
    "TEN", "WAS",
)
