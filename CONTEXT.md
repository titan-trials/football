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

## Open, not yet decided

**Should the closing spread/total feed the usage model?** A 7-point
underdog throws more, and `implied_team_totals` is free and known before
kickoff. That is a structural advantage baseball never had. It is also the
fastest way to launder the market's opinion into a model that is then
compared *against* the market. Decide deliberately; keep a no-market
variant and report both.

**Within-team correlation.** Every receiver on one team draws from the same
~35 attempts, so residuals are strongly negatively correlated within a team
and positively correlated with the game total. A week's ~200 prop rows are
worth far fewer than 200 independent observations. `paired_bootstrap` in
`model/scoring.py` currently resamples rows and therefore understates
intervals; it should resample games. Flagged in the docstring, not fixed.

**TB-over-2.5 problem, football version.** Nothing yet, but the baseball
analogue — a measurement taken and then not reported anywhere, so it could
not be recovered without re-running — is worth guarding against from the
start. Lab runs write their output CSV.

---

## Next

1. Stage 1 usage model: targets and carries, trained 2019–2025, rolling
   origin by week, serve-time features only.
2. Stage 2 per-opportunity distribution, hurdle form, shrunk to role.
3. Compound; score CRPS, then Brier at posted lines.
4. Only then a role-change flag — logged, fed to nothing.

Do not start with a feature. Items 1–3 contain none.
