# Runbook

```powershell
cd C:\Users\logic\OneDrive\Desktop\football_props
.\.venv\Scripts\Activate.ps1
```

## The whole week, in two commands

```powershell
python score_slate.py        # TUESDAY, after the Monday night game
python run_slate.py          # WEDNESDAY: capture lines -> predict -> compare
```

Then commit, or the Streamlit app keeps showing last week:

```powershell
git add slates cache/edges_*.parquet cache/market_log.parquet cache/scoring_log.parquet
git commit -m "week N slate, lines, scoring"
git push
```

Neither script takes arguments. Both work out the season and week from the
schedule, and both are safe to re-run.

Add `streamlit run dashboard.py` whenever you want to look at it.

---

## The weekly calendar

| Day | Do | Why that day |
|---|---|---|
| **Tuesday** | `python score_slate.py` then commit | Stats land nightly but the NFL issues corrections; Tuesday's data is materially cleaner than Monday night's. The snap feed also finishes landing. |
| **Wednesday** | `python run_slate.py` then commit | Props post midweek, that week's injury report is out, last week's snaps are in, and both sides of the comparison get committed at one moment. |
| Thu–Sun | look at the dashboard | Reading is free. |
| — | **do not re-run to "refresh"** | See below. |

**Why once a week and not once a day.** Re-running mid-week breaks nothing
— capture buys nothing it already has, and rows for games that kicked off
are preserved. But a Sunday re-run re-predicts the late games using
Thursday's results and Friday's injury news while those rows keep
Wednesday's *line*. The model then knows things the price does not, and any
edge it reports is partly just that gap. It flatters the model, which is
the direction nobody thinks to check.

`compare_market` measures the gap and warns above 24 hours. If you want a
genuinely updated mid-week slate, re-capture too (`--refresh`) so both
sides move together, and accept that those rows are a different comparator
from the rest of the log.

So: **one run a week** while the point is to measure whether the model
beats the market. Re-run freely once you are using it to look at games
rather than to score it.

---

## Where the model is right now

Two flags flipped on **2026-09-13**. Both are documented with their
measurements in `model_flags.py`; the short version:

| flag | value | what it does |
|---|---|---|
| `CROSS_TEAM_BLEND_W` | `0.35` | A player who changed teams gets 35% weight on his own measured level instead of 0%. Was worth 0.096 -> 0.297 on Darren Waller's over 1.5 receptions. |
| `CROSS_TEAM_BLEND_MAX_BUCKET` | `3` | The blend does not apply to deep reserves, whose history is stale. Without this it gave a WR7 a 21x bump. |
| `SNAP_DERIVED_ROLE` | `True` | Position groups are ranked by **prior-week snap share**, not the depth chart. Week 2 onward only. CRPS 6.559 -> 6.531, and 9 of 9 book-style lines improved. |

**The depth chart is a preseason document.** 98.5% of 2026 ranks hold
identical from week 1 to week 9, against 58% that move in a normal season.
That is why snaps replaced it, and it is why week 1 is the model's weakest
week — there are no snaps yet, so week 1 runs on the chart alone.

### Week 1 2026 is a THREE-model slate

`predicted_at` splits it: 09-12 04:00 (358 rows), 09-13 03:00 (1,269),
09-13 17:00 (1,061). Rows for games already kicked off keep their original
prediction, which is correct. **Do not pool week 1 with later weeks without
splitting on `predicted_at`.** From week 2 on, each week is one model.

---

## `run_slate.py`

Three steps in order — capture, predict, compare — because the order
matters and remembering it shouldn't be your job.

```powershell
python run_slate.py                # do it
python run_slate.py --dry-run      # say what it would do, spend nothing
python run_slate.py --week 3       # override the week if you must
```

What it handles so you don't have to:

* **The week** comes from the schedule: the week containing the next
  kickoff. Not a flag you could get wrong, and not last-completed-week + 1,
  which said week 2 on a day when week 1 had 14 games still to play.
* **It checks the snap feed** before predicting, and warns if last week's
  snaps have not fully landed (expect ~1,200 rows). A partial feed does not
  fail — it silently re-ranks some rosters and not others, which is worse
  than not using snaps at all. It filled in roughly a day after week 1.
* **Re-running costs nothing for games it already has.** Capture skips any
  event already in the market log for every requested market, so running it
  again Saturday to pick up the late games does not re-buy Thursday's. It
  prints how many credits that saved.
* **Spending is hard-capped** at 60 credits and the cap has to be raised on
  purpose. The first live attempt in this project would have spent 424,
  because the events endpoint returns the whole season.
* **The wrong key is refused.** `ODDS_API_KEY` in your Windows environment
  is baseball_predictor's key and its quota. This project reads
  `NFL_ODDS_API_KEY` from `.env`.
* **A capture failure does not stop the run.** Predicting is free and the
  slate is worth having on its own; only the compare step needs lines.
* **Re-running preserves committed rows.** Rows for games that already
  kicked off are kept as they were — refreshing them would be hindsight.

Capture runs first for one reason: predicting is free and repeatable, and a
missed capture can only be bought back at 10x the price. Props post around
Wednesday; before that the events exist and the markets come back empty,
which reads like a bug and is not one.

A full week costs roughly 28 credits.

Writes `slates/slate_<season>_wk<week>.parquet`, its companion pmf file,
and `cache/edges_<season>_wk<week>.parquet`.

## `score_slate.py`

```powershell
python score_slate.py              # newest week that is actually over
python score_slate.py --all        # every scorable week, oldest first
python score_slate.py --week 3     # a specific week, partial or not
```

It picks the newest week that has **both** a committed slate and a
completed set of games — 90% of that week's scheduled teams having results.
That second condition is load-bearing: on 2026-09-12, week 1 had results
for 4 of 32 teams, so "week 1 has data" was true and "week 1 can be scored"
was false. Without the check it would have scored a 16-game week off two
games and written the result into the permanent log.

If nothing is scorable it tells you how complete each week is rather than
guessing.

Re-scoring a week replaces its rows rather than duplicating them.

The block to read is **MODEL vs MARKET on N identical priced rows**. That
is the only output that decides whether any of this is worth anything.

---

## What the numbers mean

**CRPS** — lower is better; the distribution's total error, no line needed.
For comparing two versions of the model. Not interpretable alone.

**Brier skill** — `1 - brier/(p(1-p))`. Zero means no better than guessing
the base rate. **Always pooled, never averaged** — the average of a ratio
is not the ratio of averages, and baseball's dashboard once printed two
different numbers for the same rows because one averaged per-slate skills.

**vs climatology** — the null is the player's own trailing 8 games,
ignoring team and matchup. Currently **+0.112 receiving yards, +0.108
receptions**. Beating it is necessary and not impressive.

**vs market** — the bar that matters, **still unmeasured**. As of
2026-09-15 the scoring log holds one week (2025 wk10) and the market log
one captured week. Until roughly **12-15 scored weeks** show the model's
pooled skill above the market's on identical rows, this is a research
instrument and not a source of bets. At one scored week a season, that is
December. Nothing accelerates it.

**bias / ratio** — predicted mean over actual. 0.99-1.01 on the players who
get lines, which is as good as it needs to be.

**variance ratio** — predicted variance over realised squared error; 1.0 is
right and below 1.0 means overconfident. 0.87 for receivers, 0.65 for QBs,
up from 0.57 and 0.20. **Watch this one.** A model with the right mean and
too little spread under-prices every over above the median while looking
perfectly calibrated on the mean.

**mean edge on the slate** — around **-0.14**, with the model below the
book on **86%** of priced props. `compare_market` says outright that a
large mean edge usually means a bug rather than free money, and that is
still the honest reading.

What is known about it: the model is essentially **unbiased on the served
backtest population** (+0.41 yards on a mean of 12.4, +0.02 receptions on
1.1), team volume is 0.98 of actual, catch rate 0.667 against 0.672, and
top-5 team target share 0.716 against 0.721. Everything checks out except
the subset the books price. An unbiased model that is one-directionally
wrong on one subset is a **selection effect**, not a level error — and it
is the largest open question in the project. It is not a role problem, so
neither flag above touched it.

---

## Research commands

Not weekly. Each takes 4-6 minutes because it refits the model for every
week of the season.

```powershell
python compare_props.py --score-season 2025      # six props vs climatology
python validate_width.py                         # is the distribution wide enough
python validate_population.py --tag _roster      # calibration by depth-chart rank
python k_sweep.py                                # shrinkage strength (settled: no lever)
python doctor.py                                 # feeds reachable, cache sane
python -m pytest -q                              # 144 tests
```

`compare_props.py` carries flags so a change can be attributed rather than
guessed at. Always A/B with the **same command** on both sides, and always
end to end — an offline sweep on the rate lied twice in one day, because
per-team normalisation divides the level straight back out:

```powershell
--min-week 1                          # early weeks, where role errors live
--snap-roles / --no-snap-roles        # A/B snap-derived roles (needs --min-week 2)
--cross-team-w 0.35                   # mover blend weight; 0 is the old behaviour
--prior-k 4                           # shrinkage override (measured, rejected)
--population {served,stats}           # pre-2026-09-12 behaviour
--fit {roster,stats}
--share-dispersion / --no-share-dispersion
```

Current bars to beat, `--min-week 1`: CRPS **6.629** recv / **0.536** rec,
skill **+0.1143** / **+0.1069**. At `--min-week 2`: **6.531** / **0.526**,
skill **+0.1123** / **+0.1075**.

---

## When something looks off

* **"Nothing fully played to score"** — the week isn't over. It prints how
  complete each week is.
* **No lines returned** — it is before Wednesday.
* **Snap feed warning** — last week's snaps are still landing. Wait a day.
  Do not predict on a half-landed feed.
* **The dashboard shows no lines but you captured them** — you did not
  push. The cloud app serves your GitHub repo, not this folder.
* **A file looks stale** — this folder is under OneDrive, which has
  silently reverted a write in this project before. Check the file size,
  not just the timestamp.
* **Pooling across a changeover** — `SHARE_DISPERSION` flipped 2026-09-12;
  `CROSS_TEAM_BLEND_W` and `SNAP_DERIVED_ROLE` flipped 2026-09-13. Slates
  either side are different models. `CONTEXT.md` has both changeover notes.

## What is tracked and why

`.gitignore` keeps derived caches out (pbp is ~13MB a season and rebuilds
itself from nflverse) but **tracks the files that are evidence rather than
cache**: `market_log.parquet`, `edges_*.parquet`, `scoring_log.parquet`,
and everything in `slates/`. Captured odds are a snapshot of a price at a
moment — re-pull them days later and you get a different number, or nothing
— so they cannot be regenerated and must live in the repo. They add ~120KB.

This bit once already: `cache/*.parquet` swept the evidence up by where it
lived rather than what it was, so the cloud app loaded the slate fine and
then reported "no posted lines, run run_slate.py" with the lines sitting
un-pushed on disk.
