# Decision 002 — Git Branching & PR Workflow

| Field       | Value                        |
|------------|------------------------------|
| **Date**   | 2026-10-03                   |
| **Author** | Aryan (AI/CV + GCS Lead)     |
| **Status** | Accepted                     |
| **Scope**  | Team-wide git collaboration  |

## Context & Team Dynamic

The NIDAR AirMouse project involves multiple sub-teams (Autonomy/SLAM, Simulation, Flight Control, CV/GCS, Hardware).
The CV + GCS code will produce detection bounding boxes and coordinates consumed by the Autonomy state machine and the GCS interface.
A broken `main` branch halts simulation testing and autonomous flight testing for everyone.
Therefore, we need a lightweight, low-friction workflow that guarantees stability without slowing down individual iteration.

## The Workflow: GitHub Flow (Lightweight)

We use a simplified **GitHub Flow** with two rules:
1. `main` is always deployable and buildable (no broken code).
2. All active work happens on short-lived feature branches merged via Pull Requests (PRs).

```text
main ----------------------------------------* (v0.1 Tag / Milestone)
       \                                    /
        \-- feat/cv-yolo-baseline ---------/ (PR reviewed & merged)
```

## Branch Naming Convention

Keep branch names descriptive and prefixed with their domain:

| Prefix | Usage | Example |
|---|---|---|
| `feat/` | New functionality or module | `feat/cv-person-detector` |
| `fix/` | Bug fixes or issue resolutions | `fix/gcs-telemetry-disconnect` |
| `docs/` | Documentation, decision records, specs | `docs/detection-output-spec` |
| `exp/` | Research experiments (may not be merged) | `exp/low-light-clahe-filter` |
| `test/` | Adding test datasets, mocks, or unit tests | `test/sim-video-mock-stream` |

## Commit Message Convention

Use lightweight conventional commits:
- `feat(cv): add yolov8n inference loop on recorded video`
- `fix(gcs): resolve marker coordinate flip on 2D map`
- `docs(arch): update 001_tooling decision table`

## Pull Request (PR) & Review Rules for a 10-Person Team

1. **Self-Review First:**
   - Run verification tests locally before opening a PR.
   - Verify no large binary artifacts (`.venv/`, `.pt` weights, `.mp4` test videos) are tracked.
2. **One Peer Review for Cross-Cutting Changes:**
   - If a PR changes **interfaces or output formats** (e.g. survivor detection JSON or ROS 2 message formats), tag the **Autonomy Lead** and **System Lead** for review.
   - For internal CV experiments, your own review or a review by the CV research engineer is sufficient.
3. **Squash and Merge:**
   - Keep the `main` git history clean and readable for design review judges by using "Squash and merge" on GitHub.

## Repository Hygiene: What NEVER to Commit

- Virtual environments: `.venv/`, `env/`
- Model weight files: `*.pt`, `*.onnx`, `*.engine` (weights belong in release assets or a local cache directory)
- Large datasets & video files: `*.mp4`, `*.avi`, raw image batches
- OS/Editor junk: `.vscode/`, `.idea/`, `Thumbs.db`
