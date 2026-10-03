# setup_repo_structure.ps1
# Creates the standard repository structure for NIDAR CV + GCS

$directories = @(
    "cv",
    "cv/detectors",
    "cv/utils",
    "gcs",
    "gcs/backend",
    "gcs/frontend",
    "interfaces/schemas",
    "scripts",
    "notebooks",
    "tests/test_cv",
    "tests/test_interfaces",
    "configs",
    "docs/decisions",
    "docs/diary",
    "docs/setup",
    "data/samples",
    "data/test_videos",
    "models",
    "outputs"
)

Write-Host "Creating directory structure..."
foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    # Create .gitkeep so empty directories are tracked by git
    $gitkeep = Join-Path $dir ".gitkeep"
    if (-not (Test-Path $gitkeep)) {
        New-Item -ItemType File -Path $gitkeep -Force | Out-Null
    }
}

# Create initial diary entry for Sprint Day 1
$diaryPath = "docs/diary/day1_foundation.md"
if (-not (Test-Path $diaryPath)) {
    @"
# Engineering Diary — Day 1: Environment & Foundations

**Date:** $(Get-Date -Format 'yyyy-MM-dd')
**Lead:** Aryan (AI/CV + GCS Lead)

## Goals
- Tooling decisions documented (001_tooling.md)
- Python 3.12 verified
- Git repository & workflow established (002_git_workflow.md)
- Repository tree created
- Virtual environment setup and base libraries installed

## Notes & Observations
- Windows 11 host with RTX 3050 GPU.
- Git initialized with main branch and remote set to AryanGoyal17/NIDAR-Automation.
"@ | Set-Content -Path $diaryPath
}

Write-Host "Repository structure successfully initialized."
