# Python Environment Verification

**Date:** 2026-10-03
**Machine:** Aryan's Windows 11 laptop

## Verified State

| Check | Command | Expected Result |
|---|---|---|
| Installed versions | `py --list` | Shows 3.14 and 3.12 (3.14 has `*` — launcher default) |
| Default `python` | `python --version` | `Python 3.12.10` |
| Executable path | `python -c "import sys; print(sys.executable)"` | `C:\Users\aryan\AppData\Local\Programs\Python\Python312\python.exe` |
| pip owner | `python -m pip --version` | `pip 25.x.x ... (python 3.12)` |

## Verification Output (2026-10-03)

```
=== Python verification ===

1. Installed versions (py launcher):
 -V:3.14 *        Python 3.14 (64-bit)
 -V:3.12          Python 3.12 (64-bit)

2. Default python:
Python 3.12.10

3. Python executable path:
C:\Users\aryan\AppData\Local\Programs\Python\Python312\python.exe

4. pip version and owner:
pip 25.0.1 from ...\Python312\...\pip (python 3.12)

=== All checks passed ===
```

## Common Traps to Watch For

1. **`python` gives wrong version** → 3.14 is earlier in PATH. Fix: reorder PATH,
   or use `py -3.12`.
2. **`pip` belongs to different Python** → Always use `python -m pip install ...`
   instead of bare `pip install ...`.
3. **Windows Store stub** → If `sys.executable` points to `WindowsApps`, disable
   the stub in Settings → App execution aliases.
4. **`py` launcher default** → `py` without a version flag gives 3.14 (newest).
   Always use `py -3.12` or just `python`. Moot once inside a virtual environment.

## Key Rule

> Once the virtual environment is created (Step 1.3), always activate it first.
> Inside the venv, `python` and `pip` are locked to 3.12 regardless of PATH.
