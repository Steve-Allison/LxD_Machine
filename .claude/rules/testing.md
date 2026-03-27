---
description: Testing conventions and practices
globs: "tests/**/*.py"
---

# Testing

- Framework: pytest with `asyncio_mode = "auto"`.
- Test files live in `tests/`, mirroring the `src/lxd/` structure.
- Use real data stores where practical. Mock external services (Ollama, remote APIs) but not SQLite or LanceDB.
- Run tests with `pixi run pytest` (not bare `pytest`).
- Keep tests focused: one assertion per test where possible, descriptive names using `test_<unit>_<scenario>_<expected>` pattern.
