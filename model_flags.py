"""
Feature flags, mirroring baseball_predictor/model_flags.py.

A flag exists when something is built, measured, and NOT yet trusted enough
to be on by default. Each one carries the measurement that justifies its
current state, so nobody has to re-derive it -- and so that turning one on
is a decision with a number attached rather than a hunch.

Flipping a flag CHANGES THE MODEL. Record the changeover date in CONTEXT.md
when you do, or pooled scores will silently mix two models across it -- the
same bookkeeping problem as the clean/tainted split in baseball.
"""

# ---------------------------------------------------------------------
# MARKET_IN_VOLUME
#
# Feed the closing implied team total and spread into the team pass-volume
# model, on top of the team's own trailing volume.
#
# MEASURED 2026-09-11, rolling origin by season (test 2022-2025, train on
# everything prior), 2,174 held-out team-games:
#
#     all games          RMSE 7.450 -> 7.387   (+0.062)  95% CI on MSE gain [+0.220, +1.665]
#     weeks 1-4          RMSE 8.082 -> 7.982   (+0.100)  95% CI [+0.286, +2.909]
#     weeks 5+           RMSE 7.244 -> 7.194   (+0.050)  95% CI [-0.198, +1.557]
#     |spread| >= 7      RMSE 7.253 -> 7.180   (+0.073)  95% CI [-0.324, +2.400]
#
# Coefficients are stable and sensible across all four test seasons:
# +0.495 pass attempts per point of implied team total, -0.194 per point of
# own expected margin. Favourites throw less. The signal is REAL.
#
# It is also tiny: 0.06 attempts of RMSE on a base of 7.45, under 1%. And
# it is only unambiguously non-zero in weeks 1-4, where the team has no
# trailing history to speak of.
#
# WHY IT IS OFF ANYWAY, despite being real:
# the model's output is compared against the market's prop lines. Putting
# the market's game total into the model means part of what is being
# scored is the market's own opinion fed back to it. A 1% RMSE gain does
# not buy a contaminated benchmark.
#
# Note what the data does NOT say: corr(trailing volume, implied total) is
# 0.081, so the market is NOT redundant with the trailing rate -- it is
# nearly orthogonal to it. The reason the gain is small is that team pass
# volume is barely predictable from anything: R^2 of N on the trailing mean
# alone is 0.0896, and the market alone gets corr 0.117. This is not the
# baseball "level feature that overlaps an existing one" finding. It is a
# different and more uncomfortable one -- both predictors are weak.
#
# Turn it on only for a no-market-vs-market A/B that reports both, or for
# an early-season variant scoped to weeks 1-4.
MARKET_IN_VOLUME = False

# ---------------------------------------------------------------------
# MARKET_IN_SHARE
#
# Never measured. Listed so it is not confused with the flag above.
# There is no obvious mechanism -- the game total should move team volume,
# not who gets the targets -- so this would need a hypothesis before a test.
MARKET_IN_SHARE = False

# ---------------------------------------------------------------------
# ROLE_CHANGE_FLAG
#
# Log a marker when a player's snap share or depth-chart position moved
# after a teammate's injury. A CHANGE, not a level -- which is the category
# the baseball project's five negative results says is worth testing.
#
# Not built. When it is, it goes in as a logged column fed to nothing, and
# is checked after ~20 weeks. Nothing is at risk that way.
ROLE_CHANGE_FLAG = False
