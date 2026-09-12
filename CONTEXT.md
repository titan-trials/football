# CONTEXT.md — what has been tried, and what it measured

The file whose entire job is to record what has been tried. In
`baseball_predictor` this file was the thing that would have prevented
re-testing weather, and it was not opened. Open it first.

Rules for this file, same as the sibling project:
- Every entry records a **measurement**, not an intention.
- Negative results stay. They are the expensive part.
- If something was measured and lost, say so — "not reported anywhere" is
  itself a finding.

---

## 2026-09-10 — Recon and scaffold

### Verified data availability

Pulled live, not read from docs. Full table in the project doc
`nfl-data-recon-2026-09-10.md`.

- `nflreadpy` 0.1.5 is the maintained library. **`nfl_data_py` is stale** —
  `import_weekly_data` 404s and installing it downgrades pandas to 1.5.3.
- `load_schedules()` — 7,548 rows, **1999–2026**, `spread_line` and
  `total_line` 100% populated from 1999, moneylines from **2006**.
- `load_pbp(2025)` — 48,771 × 397, 285 games, ~46.5k REG plays.
- `load_player_stats(week)` — 19,422 × 150 for 2025.
- `load_participation` — 2016–2025 only; route data complete only from
  2023 (2016–2022 is ~38% non-null).
- Coverage starts, measured: snap counts **2013**, injuries **2009**,
  FTN charting **2022**, NGS **2016**.

**Two documentation errors caught by calling the loaders.** The nflverse
schedule page says the injury source ended after 2024 with no 2025 data —
false; 2025 returns 6,068 rows and 2026 wk1 returns 139. And nothing warns
that participation stops at 2025. *Pull it and count the rows.*

### The availability split — the single most important structural fact

| Feed | In-season? |
|---|---|
| pbp, player_stats, schedules, injuries, depth charts, rosters | yes |
| snap counts, NGS, FTN charting | yes, week N−1 (2026 files not yet published) |
| **participation (routes, coverage, personnel)** | **no — after postseason only** |
| ff_opportunity | not yet for 2026 |

Enforced in `data/nflverse.py` via `UnservableFeedError`. A lab may unlock
it with `allow_unservable=True`; production cannot.

### Measured: usage vs efficiency — decided the architecture

26,292 player-weeks, 2019–2025, `receiving_yards = targets × ypt`, in logs:

```
Var(log yards)     1.0894
  Var(log targets) 0.5716   52.5%
  Var(log ypt)     0.5359   49.2%
  2·Cov           -0.0181   -1.7%
```

An even variance split, so that alone decides nothing. Reliability does
(odd vs even weeks within a player-season, n = 1,913):

| Quantity | Split-half r | Full-season | Lag-1 week |
|---|---|---|---|
| **targets** | **0.863** | 0.927 | **0.525** |
| yards per target | 0.237 | 0.384 | **0.056** |
| receiving yards | 0.794 | 0.885 | — |

**Half the variance is efficiency and almost none of it is forecastable.**
ypt at 0.056 week-over-week is this project's `wind_pull_mph` (univariate
AUC 0.5004) — not weak, absent.

Consequence: two-stage, usage first. Stage 2 shrinks hard toward a
role baseline because a 0.24 reliability says per-player deviation is
mostly sampling noise.

### Measured: the shape of the target

2025, rows with ≥1 target: mean 28.4, sd 30.0, **var/mean 31.7**,
**P(0 yards) = 0.129**. Massively overdispersed with a spike at zero.
The zero comes from a *catch failing*, not from the yardage distribution,
so a hurdle/zero-inflated form is indicated rather than plain negative
binomial. (`team-run-model.md` measured var/mean 2.18 for team runs and
used NB by moments; this is the same move, harder.)

### Sample-size reality

- 42,428 player-weeks with ≥1 target, 2016–2025, across 1,626 players.
- Median career history for a targeted player: **14 games** (p25 3, p75 39).
- Only 1,026 players have ever posted a 4-target game; median 9 such games.
- 2025 alone: 4,316 targeted player-weeks, 502 players.

Per-player parameters are barely estimable. The model leans on role.

### Built

Scaffold only, no model yet. `config.py`, `data/{cache,nflverse,market_lines}.py`,
`features/shrinkage.py`, `model/scoring.py`, 33 tests. Suite passes in a
clean venv with pandas 2.3.3 / polars 1.44.2 / nflreadpy 0.1.5.

### Bug found and fixed during the build — worth recording

`implied_team_totals` had the home/away sign **inverted**, and every
structural test still passed, because the two columns reconcile to
`total_line` whichever way round they are. Caught only by eyeballing a real
row (2025 DAL@PHI: PHI favoured by 8.5, model gave PHI the *lower* total).

Verified the convention against 27 seasons of outcomes rather than against
documentation: `corr(spread_line, home_margin) = 0.426`, home margin
averages **+7.95** when `spread_line > 3` and **−6.95** when below −3. So
positive `spread_line` = home favoured, and

    home_implied = total/2 + spread/2

`test_market_lines.py` now asserts the favourite gets the larger total, not
just that the two sum correctly.

> General lesson, and it is the V4 calibrator lesson again: a test that
> checks an invariant the bug also satisfies is not a test of the bug.

---

## 2026-09-11 — Stage 1 usage model, and both open questions settled

### Architecture: targets = N x p

    N   = team pass volume that game     (team-level, volatile)
    p_i = player's share of it           (player-level, stable)

Chosen because it makes three problems fall out of one structure: it puts
the model where the reliability is, it *generates* teammate dependence
instead of bolting it on, and it isolates the market question to a single
term. Implemented in `features/usage.py`.

### RESULT: the market question is settled — MARKET_IN_VOLUME stays off

Rolling origin, test seasons 2022–2025, train on all prior, 2,174 held-out
team-games. Predicting team pass attempts:

| subset | RMSE no-market | RMSE market | gain | 95% CI on MSE gain |
|---|---|---|---|---|
| all games | 7.450 | 7.387 | +0.062 | [+0.220, +1.665] |
| weeks 1–4 | 8.082 | 7.982 | +0.100 | [+0.286, +2.909] |
| weeks 5+ | 7.244 | 7.194 | +0.050 | [-0.198, +1.557] |
| \|spread\| >= 7 | 7.253 | 7.180 | +0.073 | [-0.324, +2.400] |

Coefficients stable across all four test seasons: **+0.495 pass attempts
per point of implied team total**, **-0.194 per point of own expected
margin**. Favourites throw less. The signal is real and the direction is
exactly what football sense says.

It is also under 1% of RMSE, and only unambiguously non-zero in weeks 1–4
where the team has no trailing history. **Off by default**: a 1% gain does
not buy a benchmark contaminated by the thing being benchmarked against.

**What the data does NOT say.** corr(trailing volume, implied total) =
**0.081** — the market is *nearly orthogonal* to the trailing rate, not
redundant with it. So this is NOT the baseball "level feature overlapping
an existing one" finding, and it should not be filed as the sixth instance
of it. The reason the gain is small is more uncomfortable: **team pass
volume is barely predictable from anything.** R^2 of N on its own trailing
mean is **0.0896**; the market alone manages corr 0.117. Two weak
predictors of a noisy quantity.

That is itself an architectural result. If N is near-unpredictable, Stage
1's job is to get the *distribution* of N right, not its mean — and the
forecastable signal lives almost entirely in p_i.

### RESULT: within-team correlation — my earlier claim was wrong

The recon doc asserted residuals are "strongly negatively correlated within
a team." Measured on 2,009 teammate pairs (team-seasons >= 12 weeks,
players above 8% target share):

| what | mean Cov | % negative |
|---|---|---|
| both active, raw targets | **+0.483** | 41% |
| fixed-share multinomial predicts | +0.583 | 14% |
| all weeks, injuries included | +0.122 | 47% |
| residual, conditional on realised N | **rho = -0.147** | 69% |

Read in order:

- Teammates are **positively** correlated in raw targets. That is just the
  overdispersion — Var(N) = 63.8 against E[N] = 32.3, so the identity
  `Cov(T_i,T_j) = p_i p_j (Var(N) - E[N])` predicts a positive sign. A team
  that drops back 45 times feeds everybody.
- The fixed-share multinomial gets that roughly right, over-predicting
  slightly (+0.58 vs +0.48).
- Injuries drag the unconditional covariance down by **-0.36**. Real, and a
  *different mechanism* — availability, not allocation. Model it as
  availability, not as correlation.
- Conditional on realised volume, residuals are negatively correlated at
  **-0.147**. Given 38 dropbacks, a target to one man is a target not
  thrown to another.

**The practical consequence is the opposite of the usual warning.** Since
`Var(mean of k) = sigma^2/k * (1 + (k-1)rho)`, negative rho makes k
correlated rows worth **more** than k independent ones: at rho = -0.147,
three teammates are worth **4.24** independent rows, not fewer.

But that holds only conditional on getting N right. Miss the volume and the
positive unconditional correlation applies instead, and every teammate is
wrong in the same direction at once.

**So: the sign of teammate correlation depends on whether the error is in
volume or in allocation.** Resample GAMES, not rows — that captures both
regimes without having to pick one. `paired_bootstrap` still resamples rows
and is still wrong; now there is a number saying how.

### Validation: rolling origin, 15,171 held-out player-games

Seasons 2022–2025, week 5+, fit strictly on prior weeks. `validate_usage.py`
calls the same fit/predict functions the predictor will.

```
mean CRPS          model 1.1114   team-blind 1.7697   climatology 1.5092
CRPS skill         vs team-blind +0.3720     vs climatology +0.2636

Brier skill        over 2.5  +0.3942   (base rate 0.476)
                   over 4.5  +0.3899   (base rate 0.277)
                   over 6.5  +0.3205   (base rate 0.150)
                   over 4.5, climatology  +0.2848
```

**Do not read +0.39 as "twenty times better than the baseball model."** It
is not. The null is weaker. V4 said it directly: the baseball pool was nine
elite sluggers, so the base rate was pool-specific and there was little
between-player variance to exploit. Here the pool is every pass-catcher in
the league, so between-player variance is enormous and ranking a WR1 above
a backup TE is trivially easy. **The honest comparison is against
climatology**, and there the margin is +0.264 CRPS and +0.39 vs +0.28
Brier — real, but a fraction of what the base-rate number suggests.

Calibration, over 4.5 targets, quintiles of predicted probability:

```
pred 0.001  actual 0.016  n=3034  gap +0.015
pred 0.030  actual 0.056  n=3034  gap +0.026
pred 0.144  actual 0.161  n=3035  gap +0.017
pred 0.410  actual 0.398  n=3033  gap -0.013
pred 0.790  actual 0.756  n=3034  gap -0.034
```

Monotone and close, but the pattern is the V4 signature in miniature:
under-predicting the bottom, over-predicting the top — **spread slightly
too wide**. Watch item, not yet an action.

### Known flaw, and it is the top of the next list

**A player with no history gets share = 0, which is a point mass at zero.**
The bottom calibration bucket says predicted 0.001 against actual 0.016 —
1.6% of players the model calls impossible go over 4.5 targets. This is the
football version of the baseball Tier-3 finding where `fillna(0.0)` gave a
debut hitter a HR rate below anyone alive. A rookie WR1 in week 1 is the
case that breaks it, and it is not rare.

Fix: unknown and thin players get a role prior from depth-chart position,
not zero.

---

## 2026-09-11 (later) — Thin-player prior: depth-chart role

### The bug, and what actually fixed it

`ShareModel` returned 0.0 for an unseen player — a point mass at zero, so a
rookie WR1 in week 1 was called impossible.

**The fix needed no special case.** Shrinkage already does it:

    shrink(0, 0, prior, k) = (0 + k*prior) / (0 + k) = prior

A player with no games now gets exactly his prior, one with two games gets
mostly it, and there is no `if unknown` branch in the file. What had to
change was that the model carries a prior *per player* at all, rather than
a dict of shares with a `.get(key, 0.0)`.

That made the remaining question not "zero or prior" but **which prior**.

### Depth chart beats position, six to one

On 493 first-ever games (2022–2024), variance explained in target share:

    position only                 R^2 = 0.053
    depth_position x depth_rank   R^2 = 0.311

Slot means are monotone and sensible: WR1 0.176, WR2 0.079, WR3 0.051;
TE1 0.121, TE2 0.057; RB1 0.118, RB2 0.070, RB3 0.024.

### The schema trap — and why the prior is fitted on 2025+ only

nflverse changed the depth-chart schema after 2024, and **the two eras do
not mean the same thing by "rank"**:

- **<= 2024**: `depth_team` is string depth *within a slot*, so a team
  fielding three receivers has THREE players at `depth_team = 1`.
- **>= 2025**: `pos_rank` orders players *across the whole position group*,
  so there is exactly one WR at rank 1. Verified on PHI's 2026-09-10
  snapshot: DeVonta Smith 1, Dontayvion Wicks 2, Makai Lemon 3, Hollywood
  Brown 4. Also keyed by ISO8601 `dt` rather than week.

`data/depth.py` normalises both to the 2025+ meaning, and `era_consistency()`
checks the result before anyone pools. It does not pool:

| bucket | 2025+ | <=2024 |
|---|---|---|
| WR1 | **0.252** | 0.196 |
| WR2 | 0.180 | 0.191 |
| WR3 | 0.116 | 0.148 |

The old era is nearly flat across WR1–3, because the tie-break among three
`depth_team = 1` receivers is arbitrary. Pooling would blunt the prior
exactly where it matters most. RB and TE agree across eras within ~0.01;
**WR does not**, and WR is most of the market.

So: **the role prior is fitted on 2025+ only**, which is also the schema
that will be served in 2026, and coverage there is 99.7% against 86–89%
before. Validation moved to 2025-only for the same reason. The 2022–2025
run earlier today stays as the historical record, with that caveat.

### A/B: position prior vs role prior, 2025, rolling origin

Note what this compares. The original point-mass-at-zero bug is gone by
construction in both arms; the A/B measures **what the prior should be**.

| history | n | CRPS off -> on | Brier skill off -> on |
|---|---|---|---|
| 1–2 prior games | 57 | 0.8770 -> 0.7126 (**+0.164**) | −0.011 -> **+0.083** |
| 3–8 prior games | 273 | 0.7787 -> 0.7637 (+0.015) | +0.301 -> +0.311 |
| 9+ prior games | 2,886 | 1.0926 -> 1.0775 (+0.015) | +0.353 -> +0.359 |
| all | 3,244 | 1.0652 -> 1.0432 (+0.022) | +0.352 -> +0.361 |

Players with 1–2 games go from **negative skill to positive** — they were
being predicted worse than the base rate. Thin players are only 2.6% of
rows, so the headline moves little; that is the point of fixing it anyway.

It also helps established players slightly (+0.015 CRPS at 9+ games),
because a role-based shrinkage target is better than a position-based one
for everybody, not just rookies.

### An assumption that was backwards

I described this as the model *under*-predicting historyless players. On
the 28 such rows in 2025 it was **over**-predicting them badly: mean
predicted P(over 4.5) was **0.244** with a position prior against an actual
**0.036**. A position average is dragged up by established starters. The
role prior cuts it to 0.082 — three times better and still roughly double
the actual. **n = 28. Watch item, not resolved.**

### Headline, 2025 only, role prior on, 3,244 held-out player-games

```
mean CRPS      model 1.0432   climatology 1.7641   team-blind 1.6922
               skill vs climatology +0.4087
Brier skill    over 2.5  +0.3980   (base 0.455)
               over 4.5  +0.3605   (base 0.249)
               over 6.5  +0.2912   (base 0.133)
               over 4.5, climatology  +0.1889
```

Calibration is monotone, but the top bucket reads predicted 0.777 against
actual 0.684, a gap of **−0.093** — wider than the −0.034 on the multi-season
run. **Spread too wide at the top end**, the V4 signature, and now the
largest open calibration problem. Next measurement, not next feature.

---

## 2026-09-11 (later still) — Chasing the top-bucket gap: one fix, three dead ends, one verdict

The -0.093 top-bucket over-confidence was the largest open calibration
problem. Four hypotheses were testable cheaply. One was real, two were
small, one was actively wrong, and the gap turned out not to be a stable
property of the model at all.

### Ruled IN: era drift in the league volume prior — fixed

League mean team targets per game, measured:

    2019  33.21    2022  31.70    2025  30.28
    2020  33.54    2023  31.92
    2021  32.94    2024  31.09

Monotone decline. A prior pooled over 2019-2024 sits at **32.38** against
an actual 2025 of **30.28** — **+6.9% high**. The baseball project measured
all-history strikeout rates at **+6.8%** against a trailing 12 months.
**The same error, to within a tenth of a point, in a different sport.**

Symptom: predicted team volume ran **+1.26 targets** high. `TeamVolumeModel`
now takes `prior_seasons` (default 2, `config.PRIOR_TRAILING_SEASONS`).
Verified on identical rows, training 2019-2025 and scoring 2025:

    pooled prior     CRPS 1.0564   Brier(4.5) +0.3694   top-bucket gap -0.082
    trailing 2yr     CRPS 1.0535   Brier(4.5) +0.3707   top-bucket gap -0.072

Real, correct, and small: it closes about a tenth of the gap. Kept on its
own merits regardless.

*A note on how this was nearly missed.* The first re-validation showed NO
change, because it ran `--seasons 2025` — the training pool was one season,
so there was no drift to fix. The bias had been measured on a multi-season
fit. A fix looks worthless when tested in the configuration that cannot
exhibit the bug.

### Ruled OUT: injuries explain almost none of it

Injury report status on top-bucket rows (a serve-time feed, available
before kickoff):

    no report listed    n=776   pred 0.774   actual 0.705   gap -0.069
    Questionable        n= 47   pred 0.808   actual 0.660   gap -0.149

Flagged players miss badly, and even "Full Participation in Practice" rows
run -0.138 — appearing on the report at all is a negative signal. But they
are **5.7%** of the bucket, so removing them moves the aggregate gap only
from -0.073 to **-0.069**. Worth a flag later; not the cause.

### Ruled OUT: shrinking the share harder — correcting it makes things worse

The share estimate IS over-confident on 2025. Regression of realised share
on predicted:

    realised = +0.0064 + 0.9364 x predicted     corr 0.761
    top share bucket   predicted 0.2967  realised 0.2842   +4.2%
    bottom bucket      predicted 0.0157  realised 0.0187   -19.2%

Slope below 1 is the textbook signature of insufficient shrinkage. So sweep
the prior strength — **select on 2023-2024, verify on 2025**, never tune on
the season being judged:

    k x1.0   CRPS 1.1180   Brier(4.5) +0.4067   share slope 0.9722
    k x1.5   CRPS 1.1188   Brier(4.5) +0.4063   share slope 0.9913
    k x2.0   CRPS 1.1207   Brier(4.5) +0.4055   share slope 1.0083
    k x3.0   CRPS 1.1262   Brier(4.5) +0.4032   share slope 1.0378
    k x4.0   CRPS 1.1331   Brier(4.5) +0.4003   share slope 1.0626

**More shrinkage fixes the slope and costs accuracy, monotonically.**
k x2.0 lands the slope on 1.008 and makes both CRPS and Brier worse. The
empirical-Bayes k is already right; the slope deviation is not a defect to
correct. **No change made.**

Note also the slope is 0.972 on 2023-2024 and 0.936 on 2025 — the
"over-confidence" is itself mostly a 2025 phenomenon, which is the next
section.

### The verdict: it is not a stable property of the model

Top-quintile calibration gap, by season, role prior on and trailing prior:

| season | n | pred | actual | gap | se |
|---|---|---|---|---|---|
| 2022 | 793 | 0.798 | 0.754 | -0.0434 | 0.015 |
| 2023 | 789 | 0.802 | 0.788 | -0.0133 | 0.015 |
| 2024 | 798 | 0.785 | 0.783 | **-0.0017** | 0.015 |
| 2025 | 823 | 0.776 | 0.702 | **-0.0734** | 0.016 |

2024 is calibrated essentially perfectly. 2023 is within noise. 2025 is
~4.5 standard errors off, and season-to-season spread (sd ~0.031) is
**double** the within-season sampling error (0.015) — so there is real
between-season variation in this statistic beyond sampling.

**One anomalous season is not evidence of a model defect.** Three of four
seasons are fine, and the three mechanisms that could have explained it are
ruled out or ruled small. Continuing to chase it means fitting a story to
four data points — and V4 already wrote the rule: *complexity should clear
a bar, not win a coin flip.*

**Action: stopped. Watch item, revisit with 2026 rows.** If 2026 also comes
in near -0.07 that is two seasons and a real pattern; if it lands near zero
like 2024, 2025 was a season.

### What actually changed in the code

- `TeamVolumeModel.fit(prior_seasons=2)`, `config.PRIOR_TRAILING_SEASONS = 2`
- `validate_usage.py` gained `--prior-seasons` and `--score-season`, so the
  league prior and the scored window can be varied independently. The first
  null result came from being unable to do that.
- Two regression tests on the trailing prior, including the thin-window
  fallback.

Nothing else. Three hypotheses were tested and left no code behind, which
is the correct outcome for a hypothesis that did not survive.

---

## 2026-09-11 (Stage 2) — Yards per opportunity

### aDOT is the stable structural parameter, and it beats past efficiency

    average depth of target (aDOT)   split-half r = 0.885
    targets                          split-half r = 0.863
    catch rate                       split-half r = 0.402
    yards per target                 split-half r = 0.323

**aDOT is more reliable than target volume** — the most stable quantity in
the project. And it predicts future efficiency better than past efficiency
does, on 866 player-seasons:

    corr(first-half YPT,  second-half YPT)  = 0.255
    corr(first-half aDOT, second-half YPT)  = 0.343
    both together: R^2 = 0.130 (coefs +0.129 aDOT, +0.120 YPT — complementary)

This is level-versus-structure again. YPT is a *level*, a noisy outcome
average. aDOT is *structure* — where a player is thrown the ball. Baseball's
`bases_per_hit` was the same kind of quantity.

**Why YPT looks so unstable**: aDOT moves the hurdle and the tail in
OPPOSITE directions, and YPT is their product, so they partly cancel.

    aDOT band   P(catch)   yards|catch   P(>=20|catch)    YPT
    < 5           0.772        7.77          0.067        6.00
    5-8           0.715       10.34          0.115        7.39
    8-11          0.662       12.15          0.166        8.04
    11-14         0.605       13.65          0.206        8.26
    14+           0.547       15.93          0.262        8.72

YPT spans a factor of 1.45; P(>=20 | catch) spans **3.9**.

### The model

Hurdle per target — with P(catch | aDOT) the ball is caught and yards are
drawn from F(. | aDOT), otherwise zero. Measured P(complete | target) =
0.6728 over 68,751 targets; an incomplete gains non-zero yards 0.7% of the
time (laterals, penalties), folded into the zero.

Total game yards is that distribution convolved k times and mixed over
Stage 1's target pmf — an **exact discrete convolution**, no simulation.
Verified: mean equals `E[targets] x YPT` to 3 decimals, and variance equals
`k x` per-target variance to 1e-8. Truncation retains mass 1.000000 at a
30%-share receiver on a 35-target team.

### Result: end-to-end receiving yards, 2025, 4,115 held-out player-games

```
mean CRPS      model 10.638   league-shape 10.742   climatology 11.514
               skill vs climatology  +0.0761

Brier skill    over 24.5   model +0.3245   climatology +0.2608
               over 39.5   model +0.2781   climatology +0.2250
               over 54.5   model +0.2253   climatology +0.1811
               over 69.5   model +0.1939   climatology +0.1449

bias           predicted 23.16 vs actual 22.69   (+0.47 yards, 2%)
```

Calibration, over 39.5 yards — gaps −0.003 to +0.012, worst bucket −0.023.
**Much better than Stage 1 alone managed** (−0.073). Worth noting: summing
over targets washes out some of Stage 1's miscalibration along with
everything else.

### The uncomfortable part: aDOT shape is worth only +0.0097

Despite being the most reliable quantity in the model and reshaping the
per-target distribution dramatically, conditioning on aDOT improves total
CRPS by **+0.1045 absolute on a base of 10.74 — under 1%**.

Significant: paired bootstrap 95% CI [+0.0642, +0.1448]. Resampling **games**
instead of rows gives [+0.0646, +0.1452] — essentially identical, which is
itself worth recording: for this statistic the within-game clustering does
not widen the interval the way the teammate-covariance work suggested it
might.

**Why it is small — two hypotheses, one confirmed, one refuted.**

*Confirmed: the sum forgets the shape.* Comparing a deep-aDOT total to a
shallow one shifted to the same mean, max |CDF difference|:

    k=1   0.438      k=6   0.170
    k=2   0.225      k=8   0.168
    k=4   0.189      k=12  0.148

That is the CLT. Sum 6–8 draws and the result is near-normal regardless of
what you summed; only the mean survives, and aDOT's effect on the *mean*
is the modest 1.45x column, not the 3.9x one.

*Refuted: band coarseness.* The realised aDOT spread is wide, not narrow
(p5 −0.73, q25 0.46, median 7.00, p95 12.19 — the low end is running backs
on checkdowns), and `ADOT_CENTRES[0] = 3.0` clamps a quarter of rows into
one band. Plausible cause, so it was tested — finer low-end bands, selected
on 2024 and verified on 2025:

    2024 select   current [5,8,11,14]      CRPS 11.2180
                  low-split [0,3,...]      CRPS 11.2203
                  fine [-1,1,3,5,8,11,14]  CRPS 11.2176
    2025 verify   current                  CRPS 10.6377   gain +0.0097
                  fine                     CRPS 10.6329   gain +0.0102

A difference of 0.0004 on the selection season. **Noise. Bands unchanged.**

So ~1% is simply what aDOT shape is worth on a game total, and no amount of
resolution recovers more. Keep it — it is free, significant, and correct —
but do not expect efficiency modelling to move the needle much. The recon
doc's original conclusion survives in a sharper form: **the signal is in
Stage 1, and Stage 2's job is mostly to not be wrong.**

### Files

`features/efficiency.py` (PerTargetModel, AdotModel, exact convolution),
`validate_yards.py`, `tests/test_efficiency.py` (21 tests, including the
convolution mean/variance identities and the k=1 off-by-one check).

---

## 2026-09-11 (skeleton) — All four props through one harness

### The structure

`features/props.py` is a registry: everything that differs between markets
is **data**, everything shared is **code** in `usage.py` / `efficiency.py`.
`compare_props.py` runs them all through one rolling-origin harness.

    prop              opportunity   positions        shape
    receiving_yards   target        WR/TE/RB         aDOT
    receptions        target        WR/TE/RB         aDOT
    rushing_yards     carry         RB/WR/QB/FB      none
    passing_yards     attempt       QB               aDOT

`shape=None` for rushing is the honest default. No stable structural
parameter has been found for carries, and inventing one because receiving
has one is the "granular features are not additive" mistake the baseball
project made five times.

`efficiency.py` was generalised to `PerOpportunityModel` / `ShapeModel`
with per-prop supports and optional shape. Old names kept as aliases.

Reason for a registry rather than four scripts: `model-review-2026-09-05.md`
Tier-1 was a deployed prop compounding a rate that was never backtested,
because the backtest walked a different code path. Four hand-written prop
scripts is that failure waiting to happen four times.

### It immediately paid for itself: the sack bug

`passing_yards` came out biased **-21.05 yards** on a mean of 184.6 while
every other prop looked fine and nothing crashed.

**nflfastR sets `pass_attempt = 1` on a sack. The official stat does not
count it.** 2025 reconciliation:

    pbp      18,741 attempts   114,144 yards   6.091 y/a
    official 17,412 attempts   122,227 yards   7.020 y/a
    gap      +1,329 attempts   -8,083 yards

+1,329 is the season's sack count; -8,083 is 1,329 x -6.1. After filtering
`sack == 0` and `play_type == "pass"`, pbp reconciles to within **0.24%**
on attempts and **0.21%** on yards.

    bias        -21.05  ->  +4.61
    CRPS skill  +0.1216 ->  +0.1681
    Brier@174.5 +0.1818 ->  +0.2861
    Brier@224.5 +0.0204 ->  +0.1310

**This was invisible in isolation and obvious in comparison.** Receiving was
never affected — a sack has no receiver, so those rows were already excluded
by the null check. One prop out of four was silently wrong for a reason that
had nothing to do with the model.

`tests/test_props.py` now reconciles every prop's play-level extraction
against official weekly totals. That test would have caught it on day one.

### The comparison, 2025, rolling origin, against climatology

> **SUPERSEDED 2026-09-12 -- do not quote these numbers.** They were
> measured on the weekly-stats population rather than the population the
> predictor serves, and against a climatology whose history was read in
> non-deterministic order. Both are corrected in "The re-validation" below,
> and the corrected figures differ materially: passing yards falls from
> +0.168 to +0.022. The table is kept because the *reasoning* under it is
> still mostly right and because deleting a wrong measurement hides that it
> was made.

| prop | n | mean | CRPS | clim | skill | shape | bias |
|---|---|---|---|---|---|---|---|
| passing_yards | 508 | 184.6 | 43.882 | 52.747 | **+0.1681** | +0.0000 | +4.61 |
| receiving_yards | 4,115 | 22.7 | 10.639 | 11.517 | +0.0762 | +0.0097 | +0.61 |
| receptions | 4,115 | 2.1 | 0.811 | 0.872 | +0.0697 | **+0.0133** | +0.04 |
| rushing_yards | 3,690 | 13.2 | 5.922 | 6.356 | +0.0682 | +0.0000 | -0.24 |

Best Brier gains over climatology: passing +0.180 at 174.5, receiving +0.064
at 24.5, receptions +0.061 at 1.5, rushing +0.055 at 79.5.

### What the table says

**Passing is twice as good as anything else, and it is not the model.** A QB
takes ~95% of his team's attempts, so Stage 1's share term is nearly
degenerate and the prediction is almost entirely team volume — the one thing
the model does estimate. Climatology is also much worse for QBs (+0.106 at
174.5) because a QB's own trailing distribution is a poor guide when volume
swings with game script. Easy null, not strong model.

**Rushing gets zero from Stage 2 — by construction, and it shows.** `shape`
is None so model and shapeless are identical to four decimals, which also
confirms the harness is consistent across props. Rushing has the lowest
skill of the four. That is where a shape parameter would earn its keep, and
the candidates are structural rather than level: carries inside the 5,
goal-line share, gap vs zone.

**Receptions gets MORE from the shape parameter than receiving yards does**
(+0.0133 vs +0.0097), which is the CLT story from the Stage 2 entry read in
reverse. Receptions is a count of successes, so aDOT acts directly on the
hurdle and there is no yardage variance on top to wash it out. The prop
where structure matters most is the one with the least going on.

**Every one of these is against climatology.** That is the low bar, and it
is still the only bar attempted. The closing prop line is the real
benchmark and remains untouched.

---

## 2026-09-12 — Live slate, TD props, availability, odds client, dashboard

### Rushing shape parameter: found a good one, and it does not work

Four candidates tested for reliability (split-half, within player-season):

    shotgun share    0.824     short-yardage   0.487
    yardline         0.369     goal-line       0.163

Shotgun share is the clear winner and is second only to aDOT's 0.885 across
the whole project. It reshapes the per-carry distribution properly: YPC
**4.37 -> 5.20** and P(>=10 yards) **0.106 -> 0.160** across bands.

End-to-end it measured **-0.0009**. Slightly negative.

**The test that separates it from aDOT is predictive, not descriptive.**
aDOT BEATS past efficiency at predicting future efficiency (0.343 vs 0.255);
shotgun share LOSES to it (0.254 vs 0.367). A parameter that reshapes the
distribution but predicts the mean worse than the incumbent will move the
mean wrong often enough to cost more than the shape gains -- and the
convolution washes out shape and leaves only the mean.

Shape is now ON for exactly two props, by measurement:

    receptions       aDOT      +0.0133   ON
    receiving_yards  aDOT      +0.0097   ON
    passing_yards    aDOT      +0.0000   off
    rushing_tds      shotgun   -0.0002   off
    receiving_tds    aDOT      -0.0008   off
    rushing_yards    shotgun   -0.0009   off

**The rule: reliability plus a visible effect on the distribution is NOT
sufficient evidence to carry a shape parameter.** It must beat the incumbent
estimator of the mean. That is one correlation, and it would have predicted
every row above.

### Touchdown props needed no new machinery

A TD prop is a counting prop: per-opportunity outcome 0/1, total is the
number of scores, "anytime" is over 0.5. `receiving_tds` and `rushing_tds`
are separate specs combined at slate level:

    P(anytime) = 1 - P(no receiving TD) x P(no rushing TD)

The independence assumption is wrong in a known direction -- goal-line backs
get both carries and targets inside the 5, so the two are positively
correlated and this UNDERSTATES them, for exactly the players the market
prices most sharply. Recorded, not fixed.

Current-week comparison: receiving_tds skill **+0.1027**, rushing_tds
**+0.0873** -- both above every yardage prop, because climatology is
terrible at rare events.

### Availability flag: built, logged, fed to nothing

`data/availability.py`. Injury + practice status as a column on every slate
and an input to no model, per the flag-before-feature rule. `self_test()`
is ready for ~20 weeks from now and reports residuals by bucket; if flagged
players underperform, wire it in, and if not, delete the column.

### The live slate found three serve-time bugs, which is what it is for

`predict_slate.py` run against 2026 week 2, the first prediction this
project has ever made about a game that has not happened:

1. **`passing_yards: 0 rows`.** `data/depth.py` filtered depth charts to
   `("WR","TE","RB","FB")` -- no QB. A whole prop silently produced nothing
   and said so only in a count.
2. **Player names were gsis ids.** Depth charts key on `gsis_id`; the slate
   read `00-0032764`. `with_player_names()` added.
3. **Zero availability flags — and the report did not exist.** Week-2 injury
   reports had not published. "0 flagged" was indistinguishable from
   "everyone healthy". `report_published()` now separates the two and the
   predictor says which. **A silent absence that reads as a confident
   negative** is the same failure class as baseball's calibration block that
   printed nothing because it looked for the wrong column name.

After fixing: 4,214 rows, 994 players, all six props, all clean. Top
anytime-TD probabilities came out Chase Brown .371, Derrick Henry .354,
Jonathan Taylor .348, Jahmyr Gibbs .346, Bijan Robinson .345 — workhorse
backs, which is the right face-validity answer.

### Odds client

`data/odds.py`, mirroring `baseball_predictor/data/odds_lines.py` in shape
and key handling (ODDS_API_KEY or a gitignored `.odds_api_key`, with the
UTF-16 BOM trap handled). `dry_run=True` prints the credit cost without
spending anything.

**The clock that matters:** live props cost ~16 credits per market per week;
historical props cost **10x** and start 2023-05-03. Every uncaptured week
can only be bought back at ten times the price. That is the argument for
starting the log before the model is finished.

Nothing in `predict_slate.py` has seen a price, and MARKET_IN_VOLUME is off,
so the eventual comparison is honest.

### Dashboard

`dashboard.py` — Slate / Edges / Scoring / Flags. Mostly tables on purpose:
a slate is an identity-and-lookup problem and a table answers it better than
a chart. Charts appear only for calibration (against the identity diagonal)
where the question is genuinely about shape. The three-colour categorical
palette passed all six checks of the dataviz validator (worst adjacent CVD
dE 22.9, normal 27.4).

### A test caught a real truncation

`test_total_support_covers_the_worst_case` failed on `rushing_tds`: support
was 0..5 and six rushing touchdowns in a game has happened. Widened both TD
supports to 0..6. Small, but it is the MAX_RUNS rule doing its job -- a cap
tuned to the cases you tested is a cap that fails on the case you did not.

---

## Next

1. **Capture week 2 props before Sunday.** ~16 credits per market. The 10x
   historical multiplier means this is the only item with a deadline.
2. Join book player names to nflverse ids — fuzzy within team. This is the
   unsolved piece between a market log and an edge.
3. Score week 2 after the games, and start the running log.
4. Re-run `predict_slate.py` Wednesday+ to pick up the injury report.
5. Re-check the Stage 1 top-bucket gap once 2026 rows accumulate.
6. Availability self-test at ~20 scored weeks.

---

## 2026-09-12 (odds client) — Two near-misses caught before spending

### The events endpoint returns the season, not the slate

First live dry run: **212 upcoming events**. Two markets across them would
have cost **424 credits** — thirteen times the ~32 one slate needs, on a
quota that does not refund.

`list_events` now takes `days_ahead`, defaulting to 8 (one week's five
windows, Thursday through the following Monday, plus a day of timezone
slack). `fetch_slate_props` also enforces `max_credits=60` as a **hard stop,
not a warning**: it raises rather than spends, and has to be raised
deliberately.

The dry run is what caught this, which is the argument for it existing.

### Key precedence worked, and the warning fired correctly

`resolve_api_key` reported `from ODDS_API_KEY` with the warning that this is
`baseball_predictor`'s shared key and its quota. That is the designed
behaviour — football-specific sources first, shared last, and always say
which. Set `NFL_ODDS_API_KEY` to separate them.

### Also fixed

`data/odds.py` imported numpy and never used it — the only unused import in
the repo, and the one that surfaced when a fresh venv had no packages.
`doctor.py` added: prints Python version and every dependency's status, and
explains the atomic-pip-resolution failure mode if the version is below 3.12
(numpy 2.5.1 and scipy 1.18.0 need 3.12+, and one unsatisfiable pin installs
nothing, which is what an empty venv looks like).

---

## 2026-09-12 (live) — First market capture, and five serve-time bugs it exposed

The first real prop capture: 596 rows, 14 events, 157 players, mean hold
**6.68%**, all de-vigged, 28 credits. Name matching **157/157, all exact**.

Then the edge came out at **-0.17 mean**, which is not a result, it is a
bug. Chasing it found five, in order. Every one of them was invisible in
backtest and visible the moment real upcoming games were involved.

### 1. Players on two teams at once  (FIXED)

A traded player appears on his old AND new team's depth chart. The slate
predicted Kayshon Boutte for HOU (share .074) and NE (.044) in the same
week. `data/depth.py` now keeps one team per player -- the most recent
chart wins, since a move is published by the team he joined. Slate went
4,214 -> 3,998 rows.

Surfaced only by the name join, which reported them as `ambiguous_exact`.

### 2. Preservation faithfully preserved the bug  (FIXED)

`preserve_committed_rows` keeps rows written before kickoff so a re-run
cannot overwrite them with hindsight. That also preserved rows written by
the buggy version. `enforce_one_row_per_player` now reconciles committed
rows against the current chart. **Preserving history is not the same as
preserving mistakes.**

### 3. The role bucket flattened after 4th  (FIXED)

Measured share of team targets by depth rank, 2025:

    WR  r1 .381  r2 .273  r3 .177  r4 .097  r5 .047  r6 .020  r7 .004

The curve keeps falling; it does not flatten. Collapsing everything from 4th
down into one bucket handed every WR5-WR8 the WR4 prior. Now 7 buckets.

### 4. Zero-filling against UNPLAYED weeks  (FIXED -- the worst one)

The 2025+ depth chart is keyed by timestamp, and a snapshot attaches to
every week whose kickoff follows it, so a September chart produces rows for
weeks 1-18. Zero-filling invented a row per unplayed game, and `ShareModel`
takes the **last 8 rows per player** -- which for every player in the 2026
slate were weeks 11-18 of a season that has played one week. **Eight zeros,
all from the future.** Sam LaPorta, a TE1 with 120 targets in 2023, came out
at a .0063 share and 0.2 expected receptions.

Invisible in backtest because a backtest only ever asks for weeks that have
already happened.

### 5. Inactive weeks counted as zeros  (FIXED)

Zero-filling is how a WR6 teaches the model that WR6s get nothing. It is NOT
how an injured starter should be treated. LaPorta went on injured reserve
for weeks 11-18 of 2025; eight zeros from games he was absent for put a TE1
at a .0068 share even after fix 4. `rosters_weekly.status` separates ACT
from RES/INA/DEV/CUT, and only ACT weeks now count.

### 6. Role priors mixed a player's CURRENT rank with his PAST production (FIXED)

The prior for (WR, rank 7) was estimated by joining each player's 2026 rank
to all of his history -- so a WR7 today who was a WR2 two years ago
contributed WR2 volume to the WR7 cell. Result: **WR7 prior = 1.18 targets
a game** when a real WR7 sees 0.004 of a team's targets.

Fixed by carrying the contemporaneous role through `roster_player_games`
and estimating each bucket from games actually played at that rank:

    before (current-rank join)   WR7 1.182  WR6 1.137  RB5 0.875  TE5 1.207
    after  (contemporaneous)     WR7 0.063  WR6 1.066  RB5 0.026  TE5 0.082

### STILL OPEN: the rates sum to more than the team has

Each fix improved the slate and none closed the gap. The mechanism is now
located precisely:

    sum of unnormalised player rates   48.25
    team volume                        29.92
    ratio                              1.613

`team_vector` normalises shares to sum to 1, so that ratio is applied as a
**uniform 38% haircut to every player**, which is exactly the observed
shortfall -- model/actual targets 0.55, roughly flat across the whole
distribution (0.44 at the bottom quartile, 0.59 at the top).

Ja'Marr Chase: book line 83.5 yards, model 68.6; 8.68 targets against a real
10.9. Receptions ratio (0.745) is worse than yards (0.864), which locates it
in targets rather than efficiency -- catch rate and yards-per-target are both
about right.

**A roster cutoff does not fix it.** Even restricted to depth rank <= 3
(13.1 players a team, against 7.8 who actually record a target) the ratio is
still 1.269. The individual rates are themselves inflated relative to the
team's volume, because a backup acquired from elsewhere carries the usage
rate of the role he had there.

**The suspect is `prior_k = 0.5`** -- the floor of
`estimate_prior_strength_counts`. At that value a player's own trailing
8 games almost entirely override his role prior, and the role priors are the
things that sum to team volume by construction. Zero-heavy roster data makes
between-player variance enormous relative to within-player, which drives the
estimator to its floor.

**Not fixed, deliberately.** An earlier k sweep (select on 2023-24, verify
on 2025) found MORE shrinkage strictly worse -- but that sweep ran on the
old stats-based roster construction, which no longer exists. It has to be
re-run on the roster-based one before k is touched, and tuning it against
this week's market would be fitting to 596 correlated rows.

**The edge numbers in `cache/edges_2026_wk1.parquet` are NOT usable yet.**
The capture is good and the log is worth keeping; the model side is known
biased low by roughly a third.

### What is trustworthy from today

- The capture pipeline, de-vig, and credit guards.
- Name matching: 157/157 exact, team-scoped, with unmatched returned.
- `upcoming_week` (predicting week 2 while week 1 was still being played).
- The six fixes above, each with a regression test. 125 tests.

---

## Next

1. **Re-run the k sweep on the roster-based construction.** Select on
   2023-24, verify on 2025, never on this week's market.
2. Re-validate `compare_props.py` end to end -- it still builds player_games
   from the stats table, so the validated numbers and the served numbers
   come from different populations. That is the exact asymmetry this session
   spent its time on.
3. Only then re-read the edges.
4. Capture week 2 props Wednesday+ (also picks up that week's injury report).
5. Score week 1 after Monday.

---

## 2026-09-12 (diagnosis) — The model is unbiased in aggregate and COMPRESSES at the top

Four hypotheses tested. The first three were wrong and are recorded because
each looked right.

### WRONG: "the rates sum to more than the team has"

    sum of unnormalised rates / team volume = 1.61

True, and it does apply a uniform 0.62x multiplier through normalisation.
But normalisation is the CORRECT operation -- it enforces the team
constraint, and the backtest is unbiased with the same 1.23-1.58 ratio
present. `pred/act` = **1.024** on 2025, **1.080** on 2023-24. The ratio is
not the bug.

### WRONG: "k is too small"

Sweep on the roster-based construction, select 2023-24, verify 2025:

    k x1   CRPS 1.1248   Brier +0.4089   pred/act 1.080
    k x2   CRPS 1.1230   Brier +0.4095   pred/act 1.081
    k x4   CRPS 1.1298   Brier +0.4063   pred/act 1.081
    k x64  CRPS 1.3548   Brier +0.2590   pred/act 1.081

x2 wins by 0.002 -- a rounding difference -- and `pred/act` does not move at
all. **k is not the lever.** Same conclusion the earlier sweep reached, now
re-established on the construction that actually ships.

### WRONG: "my earlier quartile analysis showed a uniform shortfall"

It compared against *mean targets in games with >=1 target*, which
conditions on success and overstates the comparator. A methodological error
on my part, and it made a compression look like a uniform scale bug.

### WRONG: "the market side is broken"

`market_prob` is 0.500 at nearly every receiving-yards line, which looks
like a collapsed de-vig and is not: different books hang different lines,
each at roughly -110/-110, so each book's own line is its own median. The
market side is correct.

### THE ACTUAL FINDING: aggregate unbiased, top compressed

Binning by PREDICTED value hides this, which is why it took four tries --
conditioning on the predictor is exactly how regression to the mean
disguises itself. Binning by the player's own history does not.

For the 138 priced players with >=8 games of 2025 history:

    their 2025 average targets      4.70
    model projects                  3.53      ratio 0.752
    book line / their 2025 average             ratio 0.975 (receptions)
    model      / their 2025 average            ratio 0.740 (receptions)

**The market prices a prominent player at ~98% of his own recent average.
The model projects ~75%.** Meanwhile the whole-roster aggregate is unbiased
(1.024), because deep players are correspondingly over-predicted.

So the model is not low. It is COMPRESSED: it pulls everyone toward the
middle, which is invisible in a mean and fatal at the ends -- and the ends
are the only part the market prices.

Some compression is CORRECT. A 4.70 average is partly luck and should
regress. The open question is how much, and 0.75 versus the market's 0.98 is
a large gap to attribute entirely to legitimate shrinkage.

### The question is empirically answerable tomorrow

Week 1 is played 2026-09-13/14. The model's projections are committed in
`slates/slate_2026_wk1.parquet` and the market's are in
`cache/market_log.parquet`, both stamped before kickoff.

**Scoring them settles it without any further reasoning.** If the prominent
players come in near their 2025 averages, the market is right and the model
over-shrinks. If they come in near the model, the compression is correct
regression to the mean and the market is the one paying for name
recognition.

That is a far better test than another round of introspection, and it costs
nothing but waiting two days.

---

## Next

1. **Score week 1 after Monday.** It answers the compression question
   directly. Do this before changing anything.
2. Re-validate `compare_props.py` on the roster-based construction -- it
   still builds from the stats table, so the validated population and the
   served population differ.
3. Capture week 2 props Wednesday+ (also picks up that week's injury report).
4. Only then revisit shrinkage, with two weeks of market data instead of
   one.

---

## 2026-09-12 (resolved) — The predictor and the validator used different rosters

`score_slate.py` was built so Monday's results could settle the compression
question. It settled it in one run, on a week whose answers were already
known, without waiting.

Scoring **2025 week 10** through the live path:

    receiving_yards   bias -4.30 on a mean of 21.95
    receptions        bias -0.32 on 1.91
    passing_yards     bias -43.25 on 178.32

The same model measured UNBIASED in the validation harness (pred/act 1.024).
Same model, two code paths, two answers.

### The cause

`roster_player_games` -- what the model is fitted and validated on -- keeps
only **ACT** player-weeks. `predict_slate` normalised shares across the
entire depth chart.

    2025 week 10: 705 skill players listed, 329 ACT
    predictor  22.0 players/team
    validator  10.3 players/team
    dilution   2.14x

Every real contributor was scaled down to make room for practice-squad
bodies and players on injured reserve.

This is the "same code path" rule -- the one this project exists to keep --
applied to the ROSTER rather than to a function, and I broke it. The
baseball Tier-1 bug was a deployed prop compounding a rate the backtest
never touched. This is the same failure with the population instead of the
rate.

### The fix, and what it bought

`restrict_to_available` drops DEV / RES / CUT / RET before normalising.
Those are known days ahead. **INA is deliberately excluded from that list**:
a gameday scratch is not published until 90 minutes before kickoff, so using
it would be hindsight. Players with no status row are kept rather than
silently deleted.

    2025 week 10, re-scored          before    after
      receiving_yards bias            -4.30    -1.77
      receptions bias                 -0.32    -0.09
      passing_yards bias             -43.25   -26.86
      receiving_yards CRPS            10.733   10.233
      receptions CRPS                  0.786    0.741

Residual bias remains and is expected: 48 of those 705 were INA, which
cannot be known in advance, and 154 had no status row at all.

### On the compression question

It was the wrong question. The model does not over-shrink -- measured on
2025 mid-season, against the population the market prices:

    their own trailing average   6.269
    model predicted              6.038   (96.3%)
    what actually happened       6.106   (97.4%)

Shrinkage of 3.7% against real regression of 2.6%. Essentially right. The
apparent 25% compression on the 2026 slate was the roster dilution above,
arriving through a different path and looking like a modelling flaw.

### Three comparator errors made along the way, recorded because each cost time

1. **Binning by PREDICTED value hides compression.** A player predicted 3.5
   whose true rate is 4.7 lands in the "3-5" bucket alongside players whose
   true rate really is 3.5, and the bucket averages out unbiased.
   Conditioning on the predictor is how shrinkage bias disguises itself.
   Bin by something the model did not produce.
2. **"Mean targets in games with >=1 target"** conditions on success and
   overstates the comparator. It made a roster problem look like a uniform
   scale bug.
3. **`sum_rates / N = 1.61` is not a bug.** Normalisation is the correct
   operation and the backtest carries the same ratio while staying
   unbiased.

### Also established

`k` is not a lever. Swept x1 to x64 on the roster-based construction,
selecting on 2023-24: CRPS moves 0.002 and `pred/act` does not move at all.

---

## The re-validation, and the two things it found (2026-09-12)

`compare_props.py` was still building its rows from the weekly stats table
while `predict_slate.py` served the depth chart. Item 3 on the old list was
to fix that. Fixing it found a second defect that invalidated every
published skill number, and then a third that is now the most actionable
open item in the project.

### What changed in the harness

* **Scored population.** `served_rows()` builds the week's rows the way the
  predictor does -- depth chart, restricted through
  `restrict_to_available` (the predictor's own function, imported, not a
  copy), joined to actuals, missing actuals filled with 0. Row count
  roughly doubled: receiving yards 4,115 -> 7,599, passing yards 508 ->
  1,322. Every added row is a listed, available player who recorded
  nothing, and those are exactly the rows whose absence let a diluted model
  measure unbiased.
* **Fitted population.** The share model now fits on
  `roster_player_games`, as the predictor does.
* **Both are flags.** `--population {served,stats}` and
  `--fit {roster,stats}` so the old figures can be reproduced and the
  causes of any change separated. Without that, "the numbers moved" is
  ambiguous between the model changing and the sample changing.

### The baseline was not reproducible

Two runs of the identical command printed climatology CRPS **8.124** and
**8.068** for receiving yards. The model column did not move at all.

Cause: climatology takes `hist[pid][-trailing:]`, and `hist` was built by
iterating a frame in whatever order polars produced. `ShareModel` and
`TeamVolumeModel` both sort internally; that loop did not. So the *null*
was randomised run to run while the model was deterministic.

This matters more than it sounds. A scrambled history is a worse predictor
than a recent one, so **every skill-against-climatology figure published
before today was measured against an inflated baseline.** With the sort
fixed (`trailing_history()`, tested for order-independence):

| prop | published skill | skill, served population, sorted null |
|---|---|---|
| passing_yards | +0.1681 | **+0.0216** |
| receiving_yards | +0.0762 | +0.0886 |
| receptions | +0.0697 | +0.0779 |
| rushing_yards | +0.0682 | +0.0764 |
| receiving_tds | -- | +0.1220 |
| rushing_tds | -- | +0.1041 |

Passing yards is the casualty. Its headline -- "twice as good as anything
else" -- was a broken baseline. Against a climatology that actually reads a
QB's recent games in order, the model beats it by 2%. The old reading was
half right for the right reason (a QB's own average is a poor guide when
volume swings with game script) and wrong about the size by an order of
magnitude.

Lesson, in the same family as the `implied_team_totals` sign bug: **a
baseline is a measurement instrument and needs a determinism test.** Run
the identical command twice and diff it.

### The fitted construction is worth a lot

A/B on identical served rows, so this is the fit alone:

| prop | CRPS, fitted on roster | fitted on stats table | gain |
|---|---|---|---|
| passing_yards | 30.574 | 52.835 | **-42.1%** |
| receptions | 0.536 | 0.676 | -20.7% |
| rushing_yards | 3.542 | 4.511 | -21.5% |
| receiving_yards | 6.674 | 7.893 | -15.4% |

This is the strongest single result in the project so far and it validates
the deployed choice. It is also the answer to why the earlier numbers were
misleading in *both* directions at once: fitted on the stats table and
scored on the stats table, the two errors partly cancelled.

### Calibration by depth-chart rank

Sliced by `role_bucket`, which is known before kickoff. Slicing by the
prediction hides compression and slicing by the outcome conditions on
success -- both recorded mistakes above. `validate_population.py` does
this.

**On the population a book prices (ranks 1-3) the model is calibrated:**

    receiving_yards   predicted 21.90   actual 21.97   ratio 0.997
    receptions        predicted  1.99   actual  2.01   ratio 0.992
    rushing_yards     predicted 11.95   actual 12.28   ratio 0.973
    passing_yards     predicted 81.06   actual 80.20   ratio 1.011

The compression question is now closed twice over, on the served
population, per prop. There is no level problem for the players who get
lines.

### The two real problems it exposed

**1. The distribution is too narrow, and it is worst for QBs.** Mean
predicted probability against the realised base rate, ranks 1-3:

    passing_yards  over 174.5   model 0.198   actual 0.257   -5.9 pp
                   over 224.5   model 0.112   actual 0.176   -6.4 pp
                   over 264.5   model 0.063   actual 0.096   -3.3 pp
    receiving_yards over 69.5   model 0.074   actual 0.084   -1.0 pp
    rushing_yards   over 79.5   model 0.027   actual 0.037   -1.0 pp

The *mean* is right (ratio 1.011 for QBs) and the *tail is thin*. So the
model under-prices every high over. **This is the explanation for looking
far below the market on prominent players that was attributed to
compression** -- it is not the level, it is the width, and it shows up
only above the median. Every one of these has the same sign.

Where width could be missing: team volume is negative binomial fitted by
method of moments on trailing games, which captures season-level
overdispersion but not game-script variance; and the multinomial share
draw treats a player's share as fixed within the game when a blowout moves
it. Both compress the upper tail specifically.

**2. Deep-bucket over-prediction is the residual dilution, now measured.**
Predicted / actual by rank:

    WR6 1.78   WR7 2.12   RB4 3.16   TE4 1.77   QB3 6.71   QB5 (0 actual, 8.59 predicted)

Per team-week, passing yards allocate as:

    model    QB1 82.2%   QB2+ 17.8%
    actual   QB1 88.6%   QB2+ 11.4%

Total is right (+1.5%) and the split is not, so QB1 comes out 5.8% low.
That is the same dilution as the 2.14x roster bug, reduced from catastrophic
to small, arriving now through the *share prior* rather than the roster.
A backup QB throws either ~95% of his team's passes or none; a multinomial
with a shrunk mean share cannot represent a bimodal outcome, so it splits
the difference every week. Position-specific treatment of the QB slot is
the obvious next lever, and unlike aDOT or shotgun share it has a measured
5.8% of the headline prop attached to it.

---

## The width problem, found and fixed (2026-09-12)

The re-validation above said the mean was right and the tail was thin. That
is a hypothesis, not a measurement, so the next step was to measure the
width and find out WHICH stage was narrow. `validate_width.py` does it.

### The question, stated properly

A prop settles on P(X > L), not on E[X]. Those are different functionals of
the same distribution and a model can be perfect on one and wrong on every
value of the other:

    X ~ Normal(mu, sigma^2)      P(X > L) = 1 - Phi((L - mu) / sigma)

    mu = 200, sigma = 70   ->   P(X > 264.5) = 0.179
    mu = 200, sigma = 55   ->   P(X > 264.5) = 0.120

Same mean, 21% narrower, and the over is priced at 0.120 when it is worth
0.179. That is the entire reason "right on average" is not the finish line,
and it is why the mean-calibration checks were passing (1.011 for QB1)
while the model looked nothing like the market.

### The measurement: variance ratio = mean predicted variance / MSE

MSE and not the sample variance of the actuals, because E[(X - mu_i)^2] is
the quantity the predicted variance is a forecast OF; the sample variance
of a heterogeneous population also contains the between-player spread of
mu_i, which the model is supposed to predict rather than absorb.

    stage 1a  team volume (negative binomial)      1.07 - 1.08   calibrated
    stage 2   per-opportunity outcome              0.99 - 1.02   calibrated
    stage 1   player opportunities (compounded)    0.20 - 0.57   NOT

Both ends of the pipeline are right to within a few percent and the step
between them is short by a factor of two to five. That is a much stronger
result than a single end-to-end number, because it rules out four of the
five plausible fixes: the volume model is fine, the dispersion of team
volume is fine, the per-target yardage distribution is fine, and the
convolution is fine. Everything is in the share.

### Why the share, structurally

Under a fixed share the only variance a player has is sampling:

    Var(T) = p^2 Var(N) + p(1-p) E[N]

For a QB at p = 0.95 the second term is 0.0475 * E[N] -- almost nothing.
The model was asserting that a starter's attempt count is nearly certain,
when in reality he throws 41 in a shootout and 19 in a downpour. Hence QB
opportunities at ratio 0.199, the worst number in the project.

The share is not a constant. Letting it vary adds exactly the term a
multinomial cannot express:

    p ~ Beta(c*p, c*(1-p)),  T | N, p ~ Binomial(N, p)

    Var(T) = p^2 Var(N) + p(1-p) E[N] + (E[N^2] - E[N]) * Var(p)
    Var(p) = p(1-p) / (c + 1)

E[N^2] - E[N] is large and positive, so a modest Var(p) buys a lot of
width. As c -> infinity this reduces to the binomial exactly, which is
asserted in the tests: the old model is a limit of the new one, not a
sibling, so the A/B means something.

### Estimating c, and the step that is easy to get wrong

Per player, over the trailing window, with realised share s_g = T_g / N_g:

    V_obs      = Var(s_g)
    V_sampling = mean( p(1-p) / N_g )
    V_p        = V_obs - V_sampling
    c          = p(1-p) / V_p - 1

**The subtraction is the whole estimator.** Realised shares bounce around
even when the true share never moves, because 6 of 34 is a small sample.
Charging that bounce to Var(p) would double-count the noise the binomial
already models -- and it would be invisible, because it inflates the
variance in the direction we already wanted it to go. There is a test that
generates shares from a known beta and recovers c, which is the only way to
catch that class of error.

Pooled by depth-chart role, by MEDIAN, for the same reason the share prior
is pooled by role: per player, n is 8 and the one game he left with a
hamstring dominates. The per-player c distribution is heavy-tailed at both
ends, so the median is the only sane pooling.

### Result, and SHARE_DISPERSION turned on

2025 rolling origin, served population:

| prop | CRPS fixed | CRPS beta-binomial | gain |
|---|---|---|---|
| passing_yards | 30.574 | **26.803** | -12.3% |
| rushing_yards | 3.542 | **3.464** | -2.2% |
| receptions | 0.536 | **0.529** | -1.3% |
| receiving_yards | 6.674 | **6.625** | -0.7% |

Opportunity variance ratio: passing 0.199 -> **0.649**, receiving and
receptions 0.570 -> **0.868**, rushing 0.534 -> **0.720**.

Mean unmoved, as the structure requires: QB ranks 1-3 ratio 1.011 ->
1.010, receptions 0.992 -> 0.992, passing bias +1.07 -> +1.03.

Tail calibration, mean predicted p against realised base rate, ranks 1-3:

    passing over 174.5   0.198 -> 0.229   against 0.257
    passing over 224.5   0.112 -> 0.141   against 0.176
    passing over 264.5   0.063 -> 0.084   against 0.096
    passing over 299.5   0.035 -> 0.049   against 0.044
    receptions over 4.5  0.125 -> 0.135   against 0.141
    receptions over 5.5  0.073 -> 0.084   against 0.089

Brier skill rose at every line of every prop. Skill against climatology for
passing yards went +0.0216 -> **+0.1423** -- so the headline that the
sorted-baseline correction destroyed is back, this time for a reason that
is a real modelling gain rather than a broken null.

`SHARE_DISPERSION = True` as of 2026-09-12. Week 1's slate was regenerated
(2,631 rows, 2,268 clean) and is a **mixed slate**: 363 rows for games that
had already kicked off were preserved by design and carry the fixed-share
model. Do not pool week 1 with later weeks without splitting on that.

### What is still wrong with it

The ratio is 0.65-0.87, not 1.0, and the residual is worst for QBs. A beta
is unimodal; a starter's attempt count is not. He plays the whole game or
he leaves it in the second quarter, and no single beta represents a bimodal
outcome. That needs a mixture over "plays / does not finish", which is the
same shape as the QB-share concentration problem from the re-validation
above -- the two are one piece of work, not two.

---

## Next

1. Capture week 2 props Wednesday+ (picks up that week's injury report too).
2. Score week 1 Monday -- now a single command, and it will include the
   model-vs-market comparison on identical rows. Split week 1 on the
   SHARE_DISPERSION changeover when pooling.
3. **A starter/backup mixture for the QB slot.** Closes the remaining
   width gap (0.649) and the 5.8% share leak at the same time.
4. **Re-check the other three props' residual width** once the QB mixture
   lands; receiving at 0.868 may be close enough to leave alone.
5. ~~Widen the upper tail.~~ Done -- see above.
6. **Widen the upper tail further if 0.87 is not enough.** The thin-tail finding above is the first
   measured defect with a clear mechanism and a clear size. Candidates in
   order of expected value: game-script variance in team volume, then a
   within-game share draw instead of a fixed share.
4. **Concentrate the QB share.** 5.8% of QB1 passing yards is leaking to
   backups who will not play. A starter/backup mixture rather than a
   shrunk multinomial share.
5. Revisit the residual bias once INA handling is understood.
6. Re-run `k_sweep.py` and the aDOT and shotgun-share decisions against the
   sorted baseline. Those were measured against the randomised climatology
   and although they were model-vs-model comparisons (so the null does not
   enter), the CRPS levels quoted for them are on the old population.
