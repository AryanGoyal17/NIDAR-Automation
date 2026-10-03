# Git & GitHub Configuration Verification

**Date:** 2026-10-03  
**Machine:** Aryan's Windows 11 Laptop  

## Current Git Configuration

- **Git Version:** `git version 2.51.1.windows.1`
- **User Name:** `AryanGoyal17`
- **User Email:** `aryan.goyaltiet@gmail.com`
- **Credential Helper:** `manager` (Git Credential Manager for Windows)

## NIDAR Team Collaboration Model (Reference)

```text
       [Remote: GitHub (origin/main)]
                 |          ^
              pull          push
                 v          |
       [Local: main] ---> [Branch: feature/baseline-yolo]
             (Aryan)        (CV Research Engineer / Aryan)
```

1. **Repository (Repo):** The project folder containing code, history, and `.git`.
2. **Commit:** A cryptographic snapshot of changes with an author and message.
3. **Branch:** An isolated pointer to explore experiments (e.g. testing `yolov8n` vs `yolov8s`) without destabilizing the stable baseline.
4. **Remote:** The shared cloud replica on GitHub (`origin`) accessible to the wider sub-teams (Autonomy, Simulation, Flight Control).
5. **Push / Pull:** Synchronizing local commits to and from the remote.

## Verification Commands & Output

```powershell
# 1. Verify Git Installation
git --version
# Output: git version 2.51.1.windows.1

# 2. Verify Configured Identity
git config --get user.name
# Output: AryanGoyal17

git config --get user.email
# Output: aryan.goyaltiet@gmail.com

# 3. Verify Credential Helper
git config --get credential.helper
# Output: manager (Git Credential Manager)
```
