<#
    setup.ps1 -- create the venv for football_props on Windows.

    Run from the repo root:
        powershell -ExecutionPolicy Bypass -File .\setup.ps1

    WHY THIS SCRIPT EXISTS RATHER THAN "just run python -m venv":
    it encodes the three things that went wrong in baseball_predictor.

      1. Windows venvs are NOT relocatable. The launchers in Scripts\ have
         the absolute path to python.exe compiled in, so moving the project
         folder afterwards breaks pip.exe with "Unable to create process".
         baseball_predictor\venv_broken is the fossil. This script refuses
         to run from a path it thinks you will move, and prints the warning
         either way.

      2. `pip` vs `python -m pip`. Only the latter survives a move.

      3. requirements.txt must be UTF-8. PowerShell redirection writes
         UTF-16 by default and pip cannot read it back.

    This folder sits under OneDrive. That is fine for source, but OneDrive
    will happily try to sync a 200MB venv and a growing parquet cache, so
    both are excluded in .gitignore -- and you may also want to mark
    .venv\ and cache\ as "Always keep on this device" or exclude them in
    the OneDrive client if syncing gets slow.
#>

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

Write-Host "football_props setup" -ForegroundColor Cyan
Write-Host "Repo root: $root"

# --- 1. Python version check ------------------------------------------
$pyVersion = & python -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"
Write-Host "Python: $pyVersion"

$parts = $pyVersion.Split('.')
$major = [int]$parts[0]; $minor = [int]$parts[1]
if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 12)) {
    Write-Warning "requirements.txt pins numpy 2.5.1 and scipy 1.18.0, which need Python 3.12+."
    Write-Warning "You are on $pyVersion. Either upgrade Python, or relax those two pins."
    $ans = Read-Host "Continue anyway? (y/N)"
    if ($ans -ne "y") { exit 1 }
}

# --- 2. Warn about the relocation trap --------------------------------
if ($root -match "Downloads") {
    Write-Warning "This repo is under Downloads. If you move it later, the venv will break."
    Write-Warning "That is exactly how baseball_predictor\venv_broken happened. Move it first."
    $ans = Read-Host "Continue anyway? (y/N)"
    if ($ans -ne "y") { exit 1 }
}

# --- 3. Create the venv -----------------------------------------------
$venv = Join-Path $root ".venv"
if (Test-Path $venv) {
    Write-Host "`n.venv already exists." -ForegroundColor Yellow
    $ans = Read-Host "Recreate it from scratch? (y/N)"
    if ($ans -eq "y") {
        Remove-Item -Recurse -Force $venv
    }
}

if (-not (Test-Path $venv)) {
    Write-Host "`nCreating .venv ..." -ForegroundColor Cyan
    & python -m venv $venv
}

$venvPy = Join-Path $venv "Scripts\python.exe"
if (-not (Test-Path $venvPy)) { throw "venv creation failed: $venvPy not found" }

# --- 4. Install, via python -m pip ------------------------------------
Write-Host "`nUpgrading pip ..." -ForegroundColor Cyan
& $venvPy -m pip install --upgrade pip --quiet

Write-Host "Installing requirements (this pulls ~300MB, give it a minute) ..." -ForegroundColor Cyan
& $venvPy -m pip install -r (Join-Path $root "requirements.txt")

# --- 5. Verify ---------------------------------------------------------
Write-Host "`nVerifying imports ..." -ForegroundColor Cyan
# Here-string assigned first, then passed. `-c @"` on one line parses, but
# only just; this form is unambiguous.
$verify = @"
import polars, numpy, pandas, nflreadpy, sklearn, scipy
print(f'  polars     {polars.__version__}')
print(f'  numpy      {numpy.__version__}')
print(f'  pandas     {pandas.__version__}')
print(f'  nflreadpy  {nflreadpy.__version__}')
print(f'  sklearn    {sklearn.__version__}')
print(f'  scipy      {scipy.__version__}')
"@
& $venvPy -c $verify

Write-Host "`nRunning the test suite ..." -ForegroundColor Cyan
& $venvPy -m pytest tests/ -q

Write-Host "`nDone." -ForegroundColor Green
Write-Host "Activate with:  .\.venv\Scripts\Activate.ps1"
Write-Host "Or just call:   .\.venv\Scripts\python.exe <script>.py"
Write-Host ""
Write-Host "Do NOT move this folder after this point -- the venv will break." -ForegroundColor Yellow
