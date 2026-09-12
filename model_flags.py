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

# ---------------------------------------------------------------------
# SHARE_MIXTURE_POSITIONS
#
# Positions whose share is drawn from the ROLE'S EMPIRICAL DISTRIBUTION on
# a 51-point grid, instead of from a beta with that role's concentration.
# Everything not listed here keeps the beta-binomial from SHARE_DISPERSION.
#
# WHY QB AND ONLY QB. SHARE_DISPERSION fixed most of the missing width but
# left quarterbacks at a variance ratio of 0.649 against 0.868 for
# receivers. The residual is not a beta that is too narrow -- it is that a
# beta is UNIMODAL and a quarterback's share is not. Share of team pass
# attempts, 2022-2025, ACT weeks only:
#
#     QB1  n=1994  mean 0.922   <0.10: 4.4%   0.10-0.80: 5.7%   >=0.80: 90.0%
#     QB2  n=2073  mean 0.127   <0.10: 82.1%  0.10-0.80: 8.4%   >=0.80: 9.5%
#     QB3  n= 367  mean 0.120   <0.10: 84.2%  0.10-0.80: 5.7%   >=0.80: 10.1%
#     QB4  n= 137  mean 0.000   <0.10: 100%   (sd 0.000 -- never thrown a pass)
#
# QB2's mean is 0.127 and he is at 0.127 essentially never. He throws
# nothing four weeks in five and the whole game one week in ten. Any
# unimodal distribution centred on his mean describes a week that does not
# occur: right on average, wrong at every line. Receivers have no such
# structure -- a WR2's share is genuinely unimodal around its mean -- which
# is why this is scoped by position rather than turned on everywhere.
#
# It also attacks the OTHER open QB problem at the same time. Per team-week
# the old model allocated passing yards QB1 82.2% / QB2+ 17.8% against an
# actual 88.6% / 11.4%, so QB1 came out 5.8% low. A distribution that puts
# 82% of QB2's mass at exactly zero cannot leak that.
#
# COST: the role's SHAPE is pooled, so a player with an unusual pattern
# gets the role's. His LEVEL is kept exactly -- the role histogram is mixed
# with a point mass at 0 (below the role mean) or at 1 (above it), and both
# preserve E[T] = E[N] * share to 1e-3 at every share. For a position where
# the player's own history is the signal (target share reliability 0.863)
# pooling the level would be a bad trade, which is the second reason for
# the position gate.
#
# TURNED ON 2026-09-12, after SHARE_DISPERSION, measured on 2025:
#
#     opportunity variance ratio, by depth-chart rank
#         QB1   0.707 -> 1.031        QB2   0.464 -> 0.804
#     (QB3 goes 1.18 -> 2.18, over-dispersed, but its problem is its MEAN:
#      2.21 predicted attempts against 0.49 actual. That is the deep-bucket
#      share prior, which this change does not touch and was not meant to.)
#
#     passing_yards   CRPS 26.803 -> 26.055   skill vs clim +0.142 -> +0.166
#                     CRPS on ranks 1-3  34.393 -> 29.365  (-14.6%)
#                     bias +1.03 -> +0.99, mean ratio 1.010 -- untouched
#     rushing_yards   flat (QBs are a small part of it)
#     receivers       IDENTICAL, as the position gate requires
#
#     tail calibration, ranks 1-3, mean predicted p vs realised base rate
#         over 174.5   0.198 -> 0.229 -> 0.256   against 0.257
#         over 224.5   0.112 -> 0.141 -> 0.166   against 0.176
#         over 264.5   0.063 -> 0.084 -> 0.102   against 0.096
#         over 299.5   0.035 -> 0.049 -> 0.060   against 0.044
#     (three columns: fixed share, beta-binomial, and this. The systematic
#      under-pricing of QB overs is gone -- the signs are now mixed rather
#      than all negative, and the two most-traded lines are within a point.)
SHARE_MIXTURE_POSITIONS = ("QB",)

# ---------------------------------------------------------------------
# ROLE_RELATIVE_SHARE
#
# Express a player's usage as a MULTIPLE of the role he held at the time,
# and apply that multiple to the role he holds now:
#
#     expected_i = sum over trailing games of role_prior(role held THEN)
#     mu_i       = (actual_i + k) / (expected_i + k)
#     rate       = mu_i * role_prior(role held NOW)
#
# WHY. Every deep bucket over-predicts, same direction, every position:
# WR7 2.1x, RB4 3.2x, TE4 1.8x, QB3 4.5x. The role PRIORS are not the
# problem -- against what those roles actually produce they are exact to
# two decimals. The player's own history is.
#
#     current WR7s, 2025 wk10   mean depth rank over trailing 8:  3.56
#                               what they averaged there:         2.631
#                               what a WR7 averages:              1.131
#                                                                 2.33x
#
# A player at WR7 today was a WR3 a month ago and carries a WR3 average.
# prior_k sits on its floor of 0.500 -- correctly, because the estimator
# reports that players within a role genuinely differ -- so the role prior
# gets 6% of the weight at n=8 and the stale rate carries the rest.
# Raising k is the wrong fix: it flattens the real differences the
# estimator is detecting. What transfers across a demotion is not his
# rate, it is how good he was RELATIVE to the role he held.
#
# This is the same error that put the WR7 prior at 1.18 targets a game,
# fixed a day earlier, reappearing one level down -- there in the prior,
# here in the player's own rate. Stated generally: a rate earned in one
# role does not transfer to another, and a depth chart is a thing players
# move around on.
#
# WHAT HAPPENED, 2026-09-12. The mechanism works and the hypothesis was
# still wrong, which is worth recording in that order.
#
# The multiplier does exactly what it was built to do. Current WR7s, 2025
# week 10, mean RAW rate:
#
#     current model        0.4553 targets/game
#     role-relative        0.1436          (a 3.2x reduction, on target:
#                                           a WR7 actually gets 0.154)
#
# And the SERVED prediction for WR7 still got WORSE, 2.51x -> 3.44x. So
# the stale rate was never the binding constraint. What the multiplier
# does is replace "his old rank's production" with "full credit for the
# rank he holds now" -- and for deep receivers the rank he holds now is
# not worth full credit, because ranks below about four turn over
# constantly and a newly-arrived TE4 does not get what a settled TE4 gets
# (TE4 actual 0.239 against a TE4 prior of 0.647). Promotions moved UP
# more than demotions moved down, and the net was worse.
#
# End to end, all positions on:
#
#     passing_yards    26.055 -> 25.399   better
#     rushing_yards     3.465 ->  3.562   worse
#     receiving_yards   6.625 ->  6.736   worse
#
# It helps exactly where depth rank is a real distinction and hurts where
# it is close to arbitrary. QB1 versus QB2 is a fact about the team; WR6
# versus WR7 is a line on a chart that updates slowly. Gated to QB:
#
#     passing_yards    26.055 -> 25.399   (-2.5%)  skill +0.166 -> +0.187
#     rushing_yards     3.465 ->  3.442   (-0.7%)  skill +0.097 -> +0.103
#     bias unchanged on both; receivers untouched by construction
#
# Note that QB-only beats all-positions on rushing (3.442 vs 3.562) as
# well as matching it on passing, so this is not a compromise -- the
# non-QB part of the change was pure cost.
#
# WHAT REMAINS UNFIXED. WR7 2.5x, RB4 2.5x, TE4 1.5x. The deep-bucket
# over-prediction survives, and this rules out the explanation I expected.
# The next hypothesis has to be about the depth chart itself rather than
# about how its history is used.
ROLE_RELATIVE_POSITIONS = ("QB",)
