# tsbricks

TODO — one sentence on what this project does.

A minimal, production-grade Python project scaffold.

This project was generated from a Copier template designed to standardize:
- Python packaging and project structure
- Local development and CI workflows
- Linting, formatting, and testing conventions

The goal is to provide a **reliable foundation** that scales from local development to CI without surprises.

---

## Requirements

- Python 3.11.11
- [uv](https://github.com/astral-sh/uv)
- [Task](https://taskfile.dev/)

Make sure these three tools are installed.

## Getting started

Install dependencies:

```bash
uv sync
```

Pre-commit hooks can be installed with:
```bash
pre-commit install
```

## Project structure

```text
.
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── .vscode
│   └── settings.example.json
├── .github
│   └── workflows
│       └── ci.yml
├── docs
│   ├── conf.py
│   ├── index.rst
│   └── overview.rst
├── pyproject.toml
├── README.md
├── src
│   └── tsbricks
│       └── __init__.py
├── Taskfile.yml
└── tests
    ├── test_smoke.py
    └── conftest.py
```

* `src/` layout is used for correct packaging behavior
* Tooling configuration lives in `pyproject.toml`
* Automation is centralized in `Taskfile.yml`

## Development workflow

Common tasks:
```bash
task install        # Install dependencies (uv sync)
task lint           # Run linters
task lint-fix       # Run linters with automated fixes
task format         # Auto-format code
task format-check   # Check formatting without modifying
task md-format      # Auto-format Markdown files
task md-check       # Check Markdown formatting without modifying
task test           # Run tests
task check          # Run full CI suite (pre-commit + test + docs + build)
task pre-commit     # Run pre-commit hooks on all files (includes lint + format-check + md-check)
```

## How checks are organized

This project uses a **hybrid approach** for code quality:

- **Pre-commit hooks** handle file utilities (whitespace, YAML validation, secrets detection) and delegate linting/formatting to Taskfile
- **Taskfile** is the single source of truth for all project-specific checks (lint, format, md-check, test)
- **CI** runs individual tasks (`task pre-commit`, `task test`, etc.) for better visibility in GitHub Actions

`task pre-commit` handles file utilities plus lint, format-check, and md-check via delegation, so CI does not need separate steps for these.

## Documentation

Build documentation locally:
```bash
task docs
```

Then open `docs/_build/html/index.html` in your browser.

To clean built documentation:
```bash
task docs-clean
```


## Building the Package

Build the package:
```bash
task build
```



## Notes
* Tool versions are intentionally pinned where appropriate for reproducibility.
* CI (if enabled) mirrors local commands exactly via the Taskfile.
* This repository is intended to be adapted to your specific domain needs.
* Dependencies are handled by `uv` and `uv.lock` is where all dependencies are documented. As a result, `uv.lock` should be committed.
* Virtual environments must use the `.venv*` naming convention (e.g., `.venv`, `.venv-test`) so that mdformat excludes them from Markdown formatting checks.
* For VSCode users, copy `.vscode/settings.example.json` to `.vscode/settings.json`
