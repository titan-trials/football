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

# ---------------------------------------------------------------------
# CROSS_TEAM_HISTORY
#
# When a player's (team, player_id) key is missing but he has history
# under a DIFFERENT team, carry it across: his rate relative to the role he
# held then, applied to the role he holds now. Without it he falls through
# to a generic role prior as though he had never played.
#
# MEASURED ON THE 2026 WEEK 1 SLATE:
#
#     Darren Waller, TE, Miami -> Carolina
#       his shrunk rate under MIA      3.622 targets/game
#       ('CAR', waller) in rates?      False
#       so he gets the TE3 prior       0.784        a 4.6x cut
#       model P(over 1.5 receptions)   11%  against the book's 60%
#       his actual 2025 average        4.4 targets, 3.2 receptions
#
#     own history found              323 players
#     HISTORY UNDER AN OLD TEAM      108 players   (21% of the slate)
#     genuinely no history            91 players
#
# The model was not saying those 108 players got worse. It had no idea who
# they were, and priced them as generic depth pieces. That is most of why
# the week 1 board was wall-to-wall UNDER at 35-49 point gaps against the
# market -- amnesia, not an opinion.
#
# WHERE IT CAN AND CANNOT BE MEASURED. The bug bites in the weeks right
# after roster turnover and fades as players accumulate games with their
# new team, so a backtest scoring weeks 5+ barely sees it. Validate on
# EARLY weeks: `compare_props.py --min-week 1`.
#
# MEASURED, 2025 weeks 1-18, served population:
#
#   version                 CRPS recv  CRPS rec  wk1 mean edge   Waller
#   off                       6.635      0.537      -0.1363      0.73 targets
#   role-scaled (shipped)     6.629      0.536      -0.1305      0.67 targets
#   full history, shrunk      6.638      0.537      -0.1451      2.82 targets
#
# AND IT DOES NOT FIX THE EXAMPLE THAT MOTIVATED IT. Waller's own rate
# scaled by his old TE1 role is a 0.853 multiplier; times Carolina's 0.784
# TE3 prior it is still 0.67 targets, and the model still prices his over
# 1.5 receptions at 10% against the book's 60%.
#
# Carrying his FULL old rate instead puts him at 2.82 targets and 53%,
# which is the answer the example demands -- and makes both the backtest
# and the market agreement worse, because shares are normalised: handing
# 108 moved players their old volume takes it from the players who stayed.
#
# So the honest reading is that two separate things were wrong and only one
# of them is this flag. The lost history was real and is now fixed. Waller's
# 49-point gap is the DEPTH CHART calling him a TE3 -- the same binding
# constraint the deep-bucket attempt hit a day earlier. Fixing that needs a
# better read on role than a depth chart provides, not a better estimator.
CROSS_TEAM_HISTORY = True


# ---------------------------------------------------------------------
# SHARE_PRIOR_K
#
# Override the empirical-Bayes shrinkage strength of the SHARE model, in
# games. None keeps the estimated value.
#
# WHY AN OVERRIDE IS LEGITIMATE HERE. `estimate_prior_strength_counts` is
# the right answer when the shrinkage TARGET is unbiased. Ours is not:
#
#   population books price (trailing rate >= 3 targets/game, n = 8,758)
#     the player's own prior-season rate   overshoots by -0.673
#     the depth-chart role prior           undershoots by +0.636
#
# The two bracket the truth. EB does not know the prior is biased, so it
# solves a variance problem and lands at the k = 0.5 clip floor, which is
# w = 0.941 on own history at n = 8. Nearly no shrinkage at all.
#
# WHAT THAT COSTS. With every player carrying essentially his own raw
# trailing rate, the served roster's rates sum to MORE than the predicted
# team volume -- a WR4 whose 4-targets-a-game came from a stretch when the
# WR1 was hurt brings that rate onto a healthy roster. Per-team
# normalisation then squeezes the excess out of everyone uniformly,
# starters included:
#
#   2026 wk1, 145 priced players
#     their own trailing-8 mean            4.602 targets
#     what the model served                3.902          a 15% cut
#
# and zero-filling is not the cause -- only 5.9% of those listed weeks had
# zero targets, and the played-only mean is 4.766.
#
# MEASURED, 2022-2025, depth-chart-joined, zero-filled, causal trailing-8,
# 33,877 player-games (8,758 of them book-like):
#
#     k     w@n=8 | ALL rows RMSE    bias | BOOK-LIKE RMSE    bias
#     0.5   0.941 |         ~2.246  -0.07 |        ~3.62    -0.45
#     2     0.800 |          2.2282 -0.065|         3.5833  -0.256
#     4     0.667 |          2.2192 -0.053|         3.5628  -0.035
#     6     0.571 |          2.2239 -0.045|         3.5651  +0.115
#     8     0.500 |          2.2330 -0.040|         3.5752  +0.224
#    22     0.267 |          2.2941 -0.024|         3.6554  +0.569
#
# k = 4 is optimal on BOTH populations, on RMSE, and it is where the
# book-like bias crosses zero. The optimum is flat from 2 to 8 rather than
# a knife edge, so this is not a fitted parameter.
#
# It should also move share from backups to starters after normalisation:
# shrinking toward a role prior costs a backup proportionally far more
# than a starter (backup rate 2.0 toward prior 0.3 loses 28%; starter 8.0
# toward 6.0 loses 8%), and normalisation hands the difference back to the
# top of the depth chart.
#
# VALIDATED END-TO-END 2026-09-13, AND REJECTED.
#
# compare_props.py --props receptions receiving_yards --min-week 1,
# identical command both runs, n = 9,824 each:
#
#   setting        CRPS recv   CRPS rec   skill recv   skill rec   bias
#   k estimated       6.629      0.536      +0.1143     +0.1069   +0.41/+0.02
#   (lands at 0.5)
#   k = 4             6.678      0.540      +0.1078     +0.1006   +0.42/+0.02
#
# Worse on every measure, and worse at every book-style line in the Brier
# table too. k = 4 is NOT shipped.
#
# WHY THE OFFLINE SWEEP LIED. The sweep scored the shrunk RATE directly
# against realised targets. The model never uses the rate that way: the
# rate becomes a share, and `team_vector` renormalises shares per team.
# Normalisation is scale-invariant, so the absolute level the sweep was
# optimising is divided straight back out. Only the RELATIVE pattern of
# rates within a team survives, and k = 0.5 gives the better one.
#
# The corollary matters more than the flag: the 15% gap between a priced
# player's trailing-8 mean (4.602) and what the model serves (3.902) is
# NOT a leak. It is normalisation pinning the team total, which is
# separately correct at 29.9 vs 30.5 actual. There is nothing to recover
# there.
#
# WHAT IS STILL UNEXPLAINED. On the served backtest population the model
# is essentially unbiased (+0.41 yards on a mean of 12.4, +0.02 receptions
# on 1.1). On the subset books actually price it sits under the book on
# 86.1% of props. An unbiased model that is one-directionally wrong on the
# priced subset is a SELECTION effect, not a level error -- something
# about which players get lines, or which lines get posted, that the
# served population does not contain. That is the next thing to chase,
# and no shrinkage setting will touch it.
#
# The override plumbing is kept: it is cheap, it is now tested, and the
# next person to suspect shrinkage can re-measure in one command.
SHARE_PRIOR_K = None


# ---------------------------------------------------------------------
# CROSS_TEAM_BLEND_W
#
# How much weight a player who CHANGED TEAMS gets on his own absolute
# rate, against the role-scaled estimate CROSS_TEAM_HISTORY produces.
# 0.0 is the old behaviour exactly.
#
# WHAT WAS ACTUALLY WRONG WITH THE OLD MEASUREMENT. CROSS_TEAM_HISTORY
# compared two settings: role-scaled (w = 0) and "full history, shrunk"
# (w ~ 1). Full history lost, so the idea was dropped. Nobody measured
# anything in between, and the optimum is not at either end.
#
# MEASURED 2022-2025, depth-chart-joined, zero-filled, prior-season own
# rate against the contemporaneous role prior, best blend by RMSE:
#
#   regime            population      hist   chart | best w   RMSE   gain
#   week 1            all       n=869 2.4302 2.4599|  0.55  2.3305 +0.1295
#   week 1            movers    n=195 2.2491 2.1376|  0.35  2.0826 +0.0550
#   weeks 2-4         movers    n=601 2.2688 2.1350|  0.35  2.0820 +0.0529
#   weeks 5+          movers   n=3329 2.5233 2.3715|  0.35  2.3054 +0.0661
#
# w = 0.35 for movers in EVERY regime, and it beats chart-only in every
# one. Note the chart still beats history outright for movers (2.14 vs
# 2.25) -- which is why w is well below 0.5 and why "full history" lost.
#
# WHY THIS IS WORTH RE-TESTING AFTER SHARE_PRIOR_K FAILED. That flag
# moved every player toward the role prior with one k, which per-team
# normalisation largely divides back out. This does not: it changes the
# relative standing of movers against the teammates they are normalised
# against, and only movers. Different mechanism, so the earlier negative
# result does not transfer.
#
# AND A REASON TO DISTRUST THE VALIDATION ITSELF. Measured 2026-09-13:
# the 2026 depth chart is a STATIC PRESEASON SNAPSHOT -- 98.5% of
# player-slots hold the identical rank from week 1 to week 9, against 69%
# in 2025, and only 1.7% of 2026 slots ever change rank at all versus
# 57-62% in 2022-2025. Everything above is measured on charts that
# updated in-season. The chart the live slate is served is a different
# and worse object, so if anything these weights understate how much the
# live board should lean on history.
#
# NOT YET VALIDATED END TO END. Measure with
#   compare_props.py --props receptions receiving_yards --min-week 1 \
#       --cross-team-w 0.35
# against CRPS 6.629 recv / 0.536 rec, skill +0.1143 / +0.1069.
CROSS_TEAM_BLEND_W = 0.35

# Deepest role bucket CROSS_TEAM_BLEND_W applies to. Measured 2026-09-13:
# at w = 0.35 with NO bucket limit the blend fixed Waller (P(over 1.5)
# 0.096 -> 0.282 against the book's 0.60) but moved 183 of 518 receivers
# and handed absurd multiples to deep reserves whose history is stale --
# A.T. Perry 0.040 -> 0.841 expected targets (21x), DJ Turner 0.043 ->
# 0.890 (20x), Sterling Shepard 0.095 -> 1.035 (11x). Per-team
# normalisation then takes that volume straight off the starters books
# price, and the wk1 mean edge went -0.1411 -> -0.1472.
#
# A deep bucket is the chart SAYING the player is buried, which for a WR7
# is usually right. A bucket-1-3 player's ordinal position is a preseason
# guess about a genuine contributor, which is where his own level is worth
# something. Waller is TE3, so he survives the limit.
CROSS_TEAM_BLEND_MAX_BUCKET = 3


# ---------------------------------------------------------------------
# SNAP_DERIVED_ROLE
#
# Re-rank each position group by PRIOR-WEEK OFFENSIVE SNAP SHARE where it
# exists, instead of trusting the depth chart's ordering. Chart order is
# kept for anyone with no snap number, and they sort below everyone who
# has one -- not having taken a snap is itself evidence.
#
# WHY THE CHART NEEDS REPLACING AT ALL. Measured 2026-09-13: the 2026
# depth chart is a STATIC PRESEASON SNAPSHOT. 98.5% of player-slots hold
# an identical rank from week 1 to week 9, and only 1.7% ever change rank
# all season, against 57-62% in 2022-2025. It cannot incorporate camp,
# injuries or what actually happened on Sunday. A snap count can.
#
# WHY THIS IS NOT THE REFUTED PRODUCTION RE-RANK. Ranking by prior-season
# production measured WORSE than the chart (2.7431 vs 2.6974). Ranking by
# prior-week snaps measures BETTER, on identical rows (n = 22,232,
# 2022-2025, weeks 2+, causally lagged):
#
#     chart prior          RMSE 2.7336   bias +0.1344
#     SNAP-ranked prior    RMSE 2.6536   bias -0.0297
#
# Better on RMSE *and* it nearly zeroes the bias. Production is an
# outcome carrying every confound that implies; a snap count is a direct
# observation of whether the staff put him on the field.
#
# It moves 50.3% of 2025 skill-position rows, which is the size of the
# disagreement between what the chart says and what teams did.
#
# WEEK 1 GETS NOTHING. There is no prior game, so week 1 keeps the chart
# and Darren Waller's week-1 price is unaffected by this flag.
#
# VALIDATED END TO END AND TURNED ON 2026-09-13.
# compare_props.py --props receptions receiving_yards --min-week 2,
# identical command, n = 9,358 each:
#
#   setting       CRPS recv  CRPS rec  skill recv  skill rec  bias recv
#   chart roles      6.559     0.529     +0.1084    +0.1027     +0.36
#   SNAP roles       6.531     0.526     +0.1123    +0.1075     +0.35
#
# Better on every measure, and the Brier table over nine book-style lines
# is 9 better, 0 worse -- gains of +0.0024 to +0.0054 at every single
# line. That is an order of magnitude larger than CROSS_TEAM_BLEND_W,
# whose whole Brier table moved by 0.0001 either way, and unlike that
# flag this one did not need a live-board argument to justify it.
#
# This is the first change to the role signal itself that has survived.
# Four earlier attempts to fix the same problem did not: the per-player
# checkmark, full cross-team history, the production re-rank, and
# SHARE_PRIOR_K. The difference is that snaps are a DIFFERENT INPUT
# rather than a better estimator over the same input -- which is exactly
# what the deep-bucket negative result concluded would be needed.
SNAP_DERIVED_ROLE = True
