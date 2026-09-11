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

## Next

1. **Thin-player prior** — the flaw above. Cheap, and it is a correctness
   fix to something already measured.
2. Stage 2: per-opportunity yardage, hurdle form, shrunk to role.
3. Compound Stage 1 x Stage 2; score CRPS, then Brier at real posted lines.
4. Fix `paired_bootstrap` to resample games.
5. Only then a role-change flag — logged, fed to nothing.
