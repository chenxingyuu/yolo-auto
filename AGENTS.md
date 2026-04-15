# AGENTS

## Start Here

This repository follows a harness workflow: map first, then enforce.

1. Read `docs/INDEX.md` for document navigation.
2. Read `docs/QUALITY-GATES.md` for merge requirements.
3. Run `scripts/validate-repo.sh` before opening or merging a PR.

## Required Execution Rules

- Update docs when behavior, interfaces, or operations change.
- If no docs change is needed, include `[docs-skip: <reason>]` in the commit message or PR body.
- Keep durable guidance in versioned files, not chat-only instructions.
- Treat `scripts/validate-repo.sh` as the single source of truth for repository gates.

## Verification Contract

For substantive changes, always report:

- Scope: what changed and where.
- Verification: what checks were run and outcomes.
- Risk: known limitations or unverified areas.
- Next action: smallest high-value follow-up.
