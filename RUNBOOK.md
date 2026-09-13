# Runbook

```powershell
cd C:\Users\logic\OneDrive\Desktop\football_props
.\.venv\Scripts\Activate.ps1
```

## Two commands

```powershell
python run_slate.py          # Wednesday: capture lines -> predict -> compare
python score_slate.py        # Tuesday, after the Monday game
```

That is the whole weekly routine. Neither one takes arguments. Both figure
out the season and week from the schedule, and both are safe to re-run as
many times as you like.

Add `streamlit run dashboard.py` whenever you want to look at it.

### If you view the dashboard on Streamlit Community Cloud

**The cloud app serves your GitHub repo, not this folder.** Running
`run_slate.py` writes to your disk; the cloud app cannot see any of it
until it is committed and pushed. So the weekly routine gains one step:

```powershell
python run_slate.py
git add slates cache/edges_*.parquet cache/market_log.parquet
git commit -m "week N slate + captured lines"
git push
```

The app rebuilds on push, usually within a minute.

`.gitignore` keeps the derived caches out (pbp is ~13MB a season and
rebuilds itself from nflverse) but **tracks the three files that are
evidence rather than cache**: `market_log.parquet`, `edges_*.parquet`, and
`scoring_log.parquet`. Captured odds are a snapshot of a price at a moment
— re-pull them days later and you get a different number, or nothing — so
they cannot be regenerated and must live in the repo. Those three add
~120KB.

This bit once already: `cache/*.parquet` swept the evidence up by where it
lived rather than what it was, so the cloud app loaded the slate fine and
then reported "no posted lines, run run_slate.py" with the lines sitting
un-pushed on disk.

### Once a week, not once a day

Wednesday is the run. Props post midweek, that week's injury report is out,
and both sides of the comparison get committed at the same moment.

**Re-running mid-week is safe but it is not free of consequences.** Nothing
breaks -- capture buys nothing it already has, and rows for games that have
kicked off are preserved. But a Sunday re-run re-predicts the late games
using Thursday's results and Friday's injury news, while those rows keep
Wednesday's *line*. The model then knows things the price does not, and
any edge it reports is partly just that gap. It flatters the model, which
is the direction nobody thinks to check.

`compare_market` now measures the gap and warns above 24 hours. If you want
a genuinely updated mid-week slate, re-capture too (`--refresh`) so both
sides move together, and accept that those rows are a different comparator
from the rest of the log.

So: **one run a week** while the point is to measure whether the model beats
the market. Re-run freely once you are using it to look at games rather
than to score it.

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
* **Re-running costs nothing for games it already has.** Capture skips any
  event already in the market log for every requested market, so running
  it again Saturday to pick up the late games does not re-buy Thursday's.
  It prints how many credits that saved.
* **Spending is hard-capped** at 60 credits and the cap has to be raised on
  purpose. The first live attempt in this project would have spent 424,
  because the events endpoint returns the whole season.
* **The wrong key is refused.** `ODDS_API_KEY` in your Windows environment
  is baseball_predictor's key and its quota. This project reads
  `NFL_ODDS_API_KEY` from `.env`.
* **A capture failure does not stop the run.** Predicting is free and the
  slate is worth having on its own; only the compare step needs lines, and
  it says so and skips.
* **Re-running preserves committed rows.** Rows for games that already
  kicked off are kept as they were — refreshing them would be hindsight.

Capture runs first for one reason: predicting is free and repeatable, and a
missed capture can only be bought back at 10x the price. Props post around
Wednesday; before that the events exist and the markets come back empty,
which reads like a bug and is not one.

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

Run it Tuesday, not Monday night. Stats land nightly but the NFL issues
corrections, and Tuesday's data is materially cleaner. Re-scoring a week
replaces its rows rather than duplicating them.

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
ignoring team and matchup. Currently +0.09 to +0.14. Beating it is
necessary and not impressive.

**vs market** — the bar that matters, still unmeasured. One captured week,
596 rows, as of 2026-09-12. Until ~12-15 scored weeks show the model's
pooled skill above the market's on identical rows, this is a research
instrument and not a source of bets.

**bias / ratio** — predicted mean over actual. 0.99-1.01 on the players who
get lines, which is as good as it needs to be.

**variance ratio** — predicted variance over realised squared error; 1.0 is
right and below 1.0 means overconfident. Currently 0.87 for receivers, 0.65
for QBs, up from 0.57 and 0.20. **Watch this one.** A model with the right
mean and too little spread under-prices every over above the median while
looking perfectly calibrated on the mean.

**mean edge on the slate** — currently **-0.136**, and `compare_market`
says outright that a large mean edge usually means a bug rather than free
money. Partly decomposed on 2026-09-12: our distributions are right-skewed
by +4.54 units (mean above median), so a line set anywhere near the
expectation correctly prices below 0.50 — that is an artifact of where the
line sits, not a disagreement. But the book's lines also sit **+2.38 above
our mean** and above it 64% of the time, which is a real level gap in the
book's favour. Week 1's scoring is the first evidence on which side is
right.

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
python -m pytest -q                              # 136 tests
```

`compare_props.py` carries flags that exist so a change can be attributed
rather than guessed at: `--population {served,stats}` and
`--fit {roster,stats}` reproduce pre-2026-09-12 behaviour, and
`--share-dispersion` / `--no-share-dispersion` A/B the beta-binomial share.

---

## When something looks off

* **"Nothing fully played to score"** — the week isn't over. It prints how
  complete each week is.
* **No lines returned** — it is before Wednesday.
* **A file looks stale** — this folder is under OneDrive, which has
  silently reverted a write in this project before. Check the file size,
  not just the timestamp.
* **Pooling across 2026-09-12** — `SHARE_DISPERSION` flipped that day, so
  slates before and after are two different models. Week 1 is a *mixed*
  slate: rows for games already kicked off use the old share model. Split
  on it before pooling week 1 with anything later.
