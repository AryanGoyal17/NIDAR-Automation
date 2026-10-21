# AGENTS.md — NIDAR Human Detection

**Single source of project context: `CLAUDE.md` in the repository root. Read it completely before doing anything.** It covers scope, hardware targets, decisions already made, open questions, status, conventions and the rules below in more detail. Do not duplicate its content here.

## Rules for this repo (summary)

1. Scope is **human (survivor) detection only**. Do not build or modify path planning, mapping, obstacle avoidance, flight control or ground-station code.
2. Use **Planning mode** for any change that touches more than one file: produce an implementation plan artifact and wait for approval before editing.
3. After implementing, produce a walkthrough artifact with the commands you ran and their **real output**, plus anything you could not test.
4. Never invent numbers, benchmark results or rulebook facts. Unknown values are `TBD` or `# ASSUMPTION`.
5. No internet access at runtime; no cloud APIs. Never commit datasets, weights (`.pt`, `.onnx`, `.engine`), videos or secrets.
6. Code in `deploy/jetson/` must run on **Python 3.6** (no dataclasses, no walrus operator, no `X | Y` types).
7. Do not commit; leave changes for the owner to review and commit.
8. If `CLAUDE.md` contradicts the repo, report it instead of silently choosing one.