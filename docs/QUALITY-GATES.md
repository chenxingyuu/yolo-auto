# Quality Gates

This repository uses executable gates to keep docs and code aligned.

## Required Files

The repository must contain:

- `AGENTS.md`
- `docs/INDEX.md`
- `docs/QUALITY-GATES.md`
- `docs/CAPABILITY_BOUNDARIES.md`
- `docs/ITERATIONS.md`
- `scripts/validate-repo.sh`

## Docs Sync Gate

If changed files include code paths, docs updates are required unless a skip reason is explicitly declared.

- Trigger paths (`CODE_PATHS_REGEX` in `scripts/validate-repo.sh`):
  - `src/`
  - `scripts/`
  - `tests/`
  - `docker/`
  - `pyproject.toml`
- Acceptable docs paths:
  - `docs/**`
  - `README.md`
  - `AGENTS.md`

### Valid Skip Markers

Use one of:

- Commit/PR text marker: `[docs-skip: <reason>]`
- Environment variable: `DOCS_SKIP_REASON="<reason>"`

## Modes

- `DOCS_GATE_MODE=warn`: print warning and continue.
- `DOCS_GATE_MODE=enforce` (default): fail when docs gate is violated.

## Local + CI Parity

Local and CI must execute the same script:

```bash
scripts/validate-repo.sh
```
