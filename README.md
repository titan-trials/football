# football_props

NFL player-prop model. Sibling to `baseball_predictor`, built on the same
methodology and deliberately structured to avoid that project's documented
failure modes.

## Setup

Two scripts, run once each, in this order.

```powershell
powershell -ExecutionPolicy Bypass -File .\git_setup.ps1
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

`git_setup.ps1` wires the folder to `titan-trials/football`. It adopts the
remote's existing commit as the parent via `git reset --soft origin/main`
rather than merging unrelated histories or force-pushing, so the result is
linear with no conflict on README.md.

`setup.ps1` creates `.venv`, installs pinned requirements, verifies
imports, runs the tests. Git first, because `.venv` is ~300MB and there is
no reason to have it on disk while sorting out a remote.

Do not move the folder after `setup.ps1` -- Windows venvs are not
relocatable, which is what `baseball_predictor\venv_broken` is.

## Running it

Two commands, neither takes arguments. See **RUNBOOK.md**.

```powershell
python run_slate.py     # Wednesday: capture lines -> predict -> compare
python score_slate.py   # Tuesday, after the Monday game
streamlit run dashboard.py
```

Both resolve the season and week from the schedule and are safe to re-run.
`run_slate.py` skips buying lines it already has, so re-running mid-week to
pick up the late games costs nothing for the early ones.

## Layout

```
config.py              seasons, prop targets, and the availability tiers
data/
  cache.py             parquet cache, TTL for in-season feeds
  nflverse.py          loaders with serve-time discipline enforced
  market_lines.py      closing lines + de-vig; the benchmark, free to 1999
features/
  shrinkage.py         empirical-Bayes, ported from rate_features.py
model/
  scoring.py           CRPS (development) + pooled Brier skill (market)
tests/                 33 tests, including planted-signal recovery
CONTEXT.md             running log of what was tried and what it measured
```

## The two things that make this different from baseball_predictor

**1. Usage is the model; efficiency is nearly noise.**

Receiving yards factor as `targets × yards_per_target`. Measured on 26,292
player-weeks, 2019–2025:

| Quantity | Split-half r | Lag-1 week corr |
|---|---|---|
| targets | **0.863** | **0.525** |
| yards per target | 0.237 | **0.056** |

Half the variance is efficiency and almost none of it is forecastable. So
the architecture is two-stage with the weight on Stage 1:

```
Stage 1   P(opportunities = k)     <- the model
Stage 2   P(yards | opportunity)   <- shrink hard toward role baseline
Compound  convolve Stage 2 k times, mix over Stage 1
```

This is the mirror image of baseball, where plate appearances were
near-constant and all the work went into the per-PA rate.

**2. Feeds do not all land before kickoff.**

Route participation, coverage and personnel exist for 45k plays a season
and update *only after the postseason completes*. They are trainable and
not servable. `data/nflverse.py` raises `UnservableFeedError` rather than
letting a feature quietly depend on one, and every loader takes
`through_week` so backtests and the live predictor walk the same path.

## Running the tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests\ -q
```

The market-lines tests hit the network on first run and cache afterwards.

## Non-negotiables carried over from baseball

- **Backtest the code path that actually runs.** The K-prop bug shipped a
  career-window rate while validating a 12-month one; the backtest script
  was not in the repo, so nobody could check. Backtests live here and call
  the same functions the predictor calls.
- **Pool, never average, skill scores.** `1 − brier/(p(1−p))`; the average
  of a ratio is not the ratio of averages. `model/scoring.py` has no
  per-week variant to average by accident.
- **Everything shrinks**, toward a trailing prior, toward a role baseline.
  No raw means, no `fillna(0.0)`.
- **Predict before anything kicks off**, stamp every row, preserve
  committed rows on re-run.
- **A flag before a feature.** Log the column, feed it to nothing, check it
  after N weeks.
- **Level features fail** when they overlap the rolling rates — five
  separate negative results in baseball. Note the asymmetry: in football,
  *usage* features are not levels, and Stage 1 is where the signal is.

See `CONTEXT.md` for the running record and the project's recon doc for the
full reasoning.
