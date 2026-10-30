# scripts/check.ps1 — Pre-commit quality & test checks for NIDAR
# Runs: ruff check, ruff format --check, and pytest

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path "$PSScriptRoot\..").Path
Set-Location $repoRoot

# Locate executables (prefer local .venv if present)
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$pythonExe = if (Test-Path $venvPython) { $venvPython } else { "python" }

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  NIDAR Quality & Test Checks" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Python: $pythonExe"

Write-Host "`n[1/3] Running Ruff linter..." -ForegroundColor Yellow
ruff check .
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Ruff linting failed." -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "[PASS] Ruff linting clean." -ForegroundColor Green

Write-Host "`n[2/3] Running Ruff formatting check..." -ForegroundColor Yellow
ruff format --check .
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Ruff format check failed." -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "[PASS] Ruff formatting clean." -ForegroundColor Green

Write-Host "`n[3/3] Running Pytest test suite..." -ForegroundColor Yellow
& $pythonExe -m pytest
if ($LASTEXITCODE -ne 0) {
    Write-Host "[FAIL] Pytest tests failed." -ForegroundColor Red
    exit $LASTEXITCODE
}
Write-Host "[PASS] All tests passed." -ForegroundColor Green

Write-Host "`n============================================================" -ForegroundColor Green
Write-Host "  SUCCESS: All pre-commit checks passed!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
