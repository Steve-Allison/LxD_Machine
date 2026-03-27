---
description: Python code style and type-checking rules for this project
globs: "**/*.py"
---

# Python Style

- Target Python 3.14+. Use modern syntax: `type` statements, `match/case`, union `X | Y` not `Optional[X]`.
- Line length: 100 (enforced by Ruff).
- Ruff lint selects: E, F, I, UP, B, SIM, TCH. Do not add `# noqa` without justification.
- Pyright strict mode is enabled. All new code must pass strict type-checking.
- Use `pathlib.Path` not `os.path`. Use f-strings not `.format()`.
- All public functions and classes require type hints and a one-line docstring.
- Imports: use `from __future__ import annotations` only if needed. Prefer direct imports. Ruff handles import sorting (isort-compatible).
- Pydantic models: do not move field types to `TYPE_CHECKING` blocks (TC001/TC002/TC003 are ignored because Pydantic resolves types at runtime).
