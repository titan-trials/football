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

# ---------------------------------------------------------------------
# SHARE_DISPERSION
#
# Treat a player's share of team opportunities as a random variable rather
# than a constant: BetaBinomial(N, p, c) in place of Binomial(N, p), with
# the concentration c estimated per depth-chart role.
#
# WHY IT EXISTS. Decomposing the predicted variance against reality on the
# 2025 served population (validate_width.py) gave, as var(pred)/MSE:
#
#     stage 1a  team volume               1.07 - 1.08   calibrated
#     stage 2   per-opportunity outcome   0.99 - 1.02   calibrated
#     stage 1   player opportunities      0.20 - 0.57   NOT
#
#     end to end, ranks 1-3:  passing 0.372   rushing 0.531
#                             receptions 0.659   receiving 0.773
#
# Both ends of the pipeline are right and the step between them is short by
# a factor of two to five. Under a fixed share the only variance available
# to a player is sampling, p(1-p)E[N] + p^2 Var(N), and for a QB at p=0.95
# that is almost nothing -- the model asserts a starter's attempt count is
# nearly certain when in reality it swings from 19 to 41.
#
# The consequence is not a bias, which is what makes it easy to miss: mean
# calibration on ranks 1-3 is 0.997 / 0.992 / 0.973 / 1.011 across the four
# props, essentially perfect. But a prop settles on P(X > L), and a
# too-narrow distribution with the right mean under-prices every over above
# the median. Measured: QB1 passing over 224.5 priced at 0.112 against a
# realised 0.176.
#
# WHAT TO MEASURE BEFORE TURNING IT ON. The opportunity variance ratio
# should move toward 1.0 without the mean moving at all (the beta-binomial
# has the same mean as the binomial by construction -- assert it), CRPS
# should fall, and mean predicted probability at the high lines should
# approach the realised base rate. If CRPS rises while the ratio improves,
# the concentration estimate is wrong and not the structure.
#
# TURNED ON 2026-09-12. Every one of those checks passed, on 2025 rolling
# origin, served population, fixed share -> beta-binomial:
#
#     CRPS         passing 30.574 -> 26.803   (-12.3%)
#                  rushing  3.542 ->  3.464   ( -2.2%)
#                  receptions 0.536 -> 0.529  ( -1.3%)
#                  receiving  6.674 -> 6.625  ( -0.7%)
#
#     opportunity variance ratio   passing 0.199 -> 0.649
#                                  receiving/receptions 0.570 -> 0.868
#                                  rushing 0.534 -> 0.720
#
#     mean UNMOVED, as the structure requires: QB ranks 1-3 ratio
#     1.011 -> 1.010, receptions 0.992 -> 0.992, bias +1.07 -> +1.03
#
#     tail calibration, mean predicted p vs realised base rate, ranks 1-3
#         passing over 174.5   0.198 -> 0.229  against 0.257
#         passing over 224.5   0.112 -> 0.141  against 0.176
#         passing over 264.5   0.063 -> 0.084  against 0.096
#         receptions over 4.5  0.125 -> 0.135  against 0.141
#         receptions over 5.5  0.073 -> 0.084  against 0.089
#
#     Brier skill rose at EVERY line of every prop, and skill against
#     climatology for passing yards went +0.0216 -> +0.1423.
#
# CHANGEOVER BOOKKEEPING. Slates predicted before this date used the fixed
# share. slates/slate_2026_wk1.parquet is a MIXED slate: rows for games
# that had already kicked off were preserved by design and come from the
# old model. Do not pool week 1 with later weeks without splitting on it.
#
# WHAT IS STILL WRONG. The ratio is 0.65-0.87, not 1.0, and the residual is
# worst for QBs. A beta is unimodal and a starter's attempt count is not: he
# plays the whole game or he leaves it. That needs a mixture, not a wider
# beta, and it is a separate piece of work.
SHARE_DISPERSION = True
