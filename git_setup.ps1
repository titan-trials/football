<#
    git_setup.ps1 -- turn this folder into a git repo pointed at
    titan-trials/football, without clobbering the remote's initial commit.

    Run from the repo root, once:
        powershell -ExecutionPolicy Bypass -File .\git_setup.ps1

    WHAT IT DOES AND WHY IT IS NOT THE OBVIOUS SEQUENCE
    ---------------------------------------------------
    The remote already has one commit: a README.md containing "# football
    / props". This folder also has a README.md. So the naive

        git init; git add -A; git commit; git remote add; git push

    fails -- the remote has history yours does not contain -- and the usual
    escapes are both bad:

      * `git pull --allow-unrelated-histories` merges two histories that
        share no ancestor and hands you a conflict on README.md.
      * `git push --force` works, but silently discards the remote commit.

    Instead this uses `git reset --soft origin/main` on a fresh repo. That
    adopts the remote's commit as HEAD without touching your working tree,
    so your files land as ONE ordinary commit on top of it. Linear history,
    no merge, no conflict, no force. Your README.md replaces the two-line
    placeholder as a normal edit.

    Verified end to end before this script was written: 19 tracked files,
    cache/*.parquet correctly ignored, cache/.gitkeep kept, diff vs the
    remote's initial commit = 19 files changed, 1607 insertions, 2
    deletions.
#>

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$REMOTE = "https://github.com/titan-trials/football.git"

Write-Host "git setup for football_props" -ForegroundColor Cyan
Write-Host "Repo root: $root"
Write-Host "Remote:    $REMOTE`n"

# --- 0. Guard: already a repo? ----------------------------------------
if (Test-Path (Join-Path $root ".git")) {
    Write-Warning "This folder is already a git repository. Nothing to do."
    Write-Host "Current remotes:"
    & git remote -v
    exit 0
}

# --- 1. Identity -------------------------------------------------------
$gitName  = & git config --global user.name
$gitEmail = & git config --global user.email
if (-not $gitName -or -not $gitEmail) {
    Write-Warning "git has no global user.name / user.email set."
    Write-Host "Set them first, or this commit will be attributed to nobody:"
    Write-Host '    git config --global user.name  "Your Name"'
    Write-Host '    git config --global user.email "you@example.com"'
    exit 1
}
Write-Host "Committing as: $gitName <$gitEmail>`n"

# --- 2. Init and wire up the remote ------------------------------------
Write-Host "Initialising ..." -ForegroundColor Cyan
& git init -b main
& git remote add origin $REMOTE

Write-Host "Fetching the remote's existing commit ..." -ForegroundColor Cyan
& git fetch origin
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Fetch failed. If this is an auth problem, sign in first:"
    Write-Host "    winget install --id GitHub.cli"
    Write-Host "    gh auth login"
    Write-Host "Then re-run this script (delete the .git folder it just made first)."
    exit 1
}

# --- 3. Adopt remote history without touching the working tree ---------
Write-Host "Adopting remote history as the parent commit ..." -ForegroundColor Cyan
& git reset --soft origin/main

# --- 4. Commit everything ----------------------------------------------
Write-Host "`nStaging ..." -ForegroundColor Cyan
& git add -A

Write-Host "`nWhat will be committed:" -ForegroundColor Yellow
& git status --short

Write-Host "`nSanity check -- the cache must NOT appear above." -ForegroundColor Yellow
Write-Host "(cache/.gitkeep is expected; cache/*.parquet is not.)"
$ans = Read-Host "`nLooks right? Commit and push? (y/N)"
if ($ans -ne "y") {
    Write-Host "Stopped. Nothing was committed. The .git folder exists; delete it to start over."
    exit 0
}

# Here-string assigned to a variable first, then passed. `-m @"` on one
# line parses, but only just; this form is unambiguous.
$msg = @"
Scaffold: nflverse data layer, shrinkage, scoring

Two-stage architecture (usage then efficiency), decided by measurement:
targets have split-half reliability 0.863 and lag-1 week correlation
0.525; yards per target are 0.237 and 0.056. Half the variance of
receiving yards is efficiency and almost none of it is forecastable.

- data/nflverse.py enforces serve-time discipline. Participation data
  (routes, coverage, personnel) updates only after the postseason and
  raises UnservableFeedError rather than silently becoming a feature
  that cannot be served.
- data/market_lines.py reads closing lines free from nflverse schedules,
  1999-present. De-vigged. The benchmark, at zero credits.
- features/shrinkage.py ports the empirical-Bayes estimators from
  baseball_predictor, shrinking toward a role baseline rather than one
  league mean.
- model/scoring.py: CRPS for development, pooled Brier skill for market
  comparison. No per-week variant, so the average-of-a-ratio bug cannot
  recur.

33 tests, including planted-signal recovery of the prior strength.
"@

& git commit -m $msg

Write-Host "`nPushing ..." -ForegroundColor Cyan
& git push -u origin main
if ($LASTEXITCODE -ne 0) {
    Write-Warning "Push failed -- almost certainly auth. The commit is safe locally."
    Write-Host "Sign in and re-run just the push:"
    Write-Host "    gh auth login"
    Write-Host "    git push -u origin main"
    exit 1
}

Write-Host "`nDone." -ForegroundColor Green
& git log --oneline
