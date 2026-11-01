# scripts/check.ps1 — Pre-commit quality & test checks for NIDAR
# Runs: ruff check, ruff format --check, and pytest
#
#   .\scripts\check.ps1         fast: heavy scene tests skipped (NIDAR_SCENE_TESTS cleared)
#   .\scripts\check.ps1 -Full   everything (NIDAR_SCENE_TESTS=1); run before asking to merge.
#                               Fails if any test is skipped, unless its reason is allowlisted.

param(
    [switch]$Full
)

$ErrorActionPreference = "Stop"

# Skip reasons that -Full tolerates. Each entry is a regex matched against the
# reason pytest prints with -rs, for example "could not import 'foo'". Empty on
# purpose: a skip in -Full is a test that did not run before a merge. Add an
# entry only for a test that cannot run on the dev laptop at all (for example
# hardware it does not have), with a comment saying why, e.g.
#   'needs a Raspberry Pi'   # deploy/pi benchmark, runs on the Pi only
$FullSkipAllowlist = @()

$repoRoot = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $repoRoot

# Locate executables (prefer local .venv if present)
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$pythonExe = if (Test-Path $venvPython) { $venvPython } else { (Get-Command python).Source }
$pythonPath = & $pythonExe -c "import sys; print(sys.executable)"

$mode = if ($Full) { "full (scene tests on, skips fail)" } else { "fast (scene tests skipped)" }
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  NIDAR Quality & Test Checks" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Python: $pythonPath"
Write-Host "  Mode:   $mode"

if ($Full) {
    & $pythonExe -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('pybullet') else 1)"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[FAIL] pybullet missing: activate .venv and run again" -ForegroundColor Red
        exit 1
    }
}

Write-Host "`n[1/3] Running Ruff linter..." -ForegroundColor Yellow
& $pythonExe -m ruff check .
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Ruff linting failed." -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "[PASS] Ruff linting clean." -ForegroundColor Green

Write-Host "`n[2/3] Running Ruff formatting check..." -ForegroundColor Yellow
& $pythonExe -m ruff format --check .
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Ruff format check failed." -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "[PASS] Ruff formatting clean." -ForegroundColor Green

Write-Host "`n[3/3] Running Pytest test suite..." -ForegroundColor Yellow
$pytestArgs = @("-m", "pytest")
if ($Full) { $pytestArgs += "-rs" }
# The script runs in the caller's session, so restore the variable afterwards.
$savedSceneTests = $env:NIDAR_SCENE_TESTS
try {
    $env:NIDAR_SCENE_TESTS = if ($Full) { "1" } else { $null }
    & $pythonExe @pytestArgs | Tee-Object -Variable pytestOutput
    $pytestExit = $LASTEXITCODE
} finally {
    $env:NIDAR_SCENE_TESTS = $savedSceneTests
}

$summary = $pytestOutput | Where-Object { $_ -match '\d+ (passed|failed|skipped|errors?)\b.* in [\d.]+s' } |
    Select-Object -Last 1
$skipped = if ($summary -match '(\d+) skipped') { [int]$Matches[1] } else { 0 }
Write-Host "`npytest skipped: $skipped"

if ($pytestExit -ne 0) {
    Write-Host "[FAIL] Pytest tests failed." -ForegroundColor Red
    exit $pytestExit
}

if ($Full) {
    # -rs lines look like "SKIPPED [3] tests\...py:12: reason"; [3] is the test count.
    $allowed = 0
    $refused = @()
    foreach ($line in $pytestOutput) {
        if ($line -notmatch '^SKIPPED \[(\d+)\] (.+)$') { continue }
        $count = [int]$Matches[1]
        $reason = $Matches[2]
        $isAllowed = $false
        foreach ($pattern in $FullSkipAllowlist) {
            if ($reason -match $pattern) { $isAllowed = $true }
        }
        if ($isAllowed) { $allowed += $count } else { $refused += $reason }
    }
    if ($skipped -gt $allowed) {
        if ($refused -match "could not import 'pybullet'") {
            Write-Host "[FAIL] pybullet missing: activate .venv and run again" -ForegroundColor Red
        }
        Write-Host "[FAIL] $($skipped - $allowed) skipped test(s) not on the -Full allowlist:" -ForegroundColor Red
        $refused | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
        exit 1
    }
    Write-Host "[PASS] All tests passed ($allowed allowlisted skip(s))." -ForegroundColor Green
} else {
    Write-Host "[PASS] All tests passed." -ForegroundColor Green
    Write-Host "scene tests skipped: run scripts/check.ps1 -Full before merging" -ForegroundColor Yellow
}

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host "  SUCCESS: All pre-commit checks passed!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
