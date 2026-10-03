#!/usr/bin/env bash
# setup_repo_structure.sh
# Creates standard repository structure for NIDAR CV + GCS

set -e

directories=(
    "cv"
    "cv/detectors"
    "cv/utils"
    "gcs"
    "gcs/backend"
    "gcs/frontend"
    "interfaces/schemas"
    "scripts"
    "notebooks"
    "tests/test_cv"
    "tests/test_interfaces"
    "configs"
    "docs/decisions"
    "docs/diary"
    "docs/setup"
    "data/samples"
    "data/test_videos"
    "models"
    "outputs"
)

echo "Creating directory structure..."
for dir in "${directories[@]}"; do
    mkdir -p "$dir"
    touch "$dir/.gitkeep"
done

echo "Repository structure successfully initialized."
