#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DOCS_GATE_MODE="${DOCS_GATE_MODE:-enforce}"
DOCS_SKIP_REASON="${DOCS_SKIP_REASON:-}"
PR_BODY="${PR_BODY:-}"
COMMIT_MESSAGE="${COMMIT_MESSAGE:-}"

REQUIRED_FILES=(
  "AGENTS.md"
  "docs/INDEX.md"
  "docs/QUALITY-GATES.md"
  "docs/CAPABILITY_BOUNDARIES.md"
  "docs/ITERATIONS.md"
  "scripts/validate-repo.sh"
)

# Files considered "code impact" for docs-sync enforcement.
CODE_PATHS_REGEX='^(src/|scripts/|tests/|docker/|pyproject\.toml$)'
DOCS_PATHS_REGEX='^(docs/|README\.md$|AGENTS\.md$)'
DOCS_SKIP_REGEX='\[docs-skip:[^]]+\]'

fail() {
  echo "ERROR: $1" >&2
  exit 1
}

warn() {
  echo "WARNING: $1" >&2
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "missing required command: $1"
}

collect_changed_files() {
  local merge_base
  if git rev-parse --verify origin/main >/dev/null 2>&1; then
    merge_base="$(git merge-base origin/main HEAD)"
    git diff --name-only "$merge_base...HEAD" || true
  elif git rev-parse --verify HEAD~1 >/dev/null 2>&1; then
    git diff --name-only HEAD~1...HEAD || true
  else
    git ls-files
  fi
}

extract_docs_skip_reason() {
  local combined
  combined="${DOCS_SKIP_REASON}"$'\n'"${PR_BODY}"$'\n'"${COMMIT_MESSAGE}"
  local marker
  marker="$(printf "%s" "$combined" | rg -o "$DOCS_SKIP_REGEX" -m 1 || true)"
  if [[ -n "$marker" ]]; then
    echo "$marker"
  fi
}

main() {
  require_command git
  require_command rg

  echo "== validate-repo =="
  echo "mode: ${DOCS_GATE_MODE}"

  for file in "${REQUIRED_FILES[@]}"; do
    [[ -f "$file" ]] || fail "required file missing: $file"
  done
  echo "required files: ok"

  local changed_files
  changed_files="$(collect_changed_files)"

  local triggering_code_files
  triggering_code_files="$(printf "%s\n" "$changed_files" | rg "$CODE_PATHS_REGEX" || true)"

  local changed_docs_files
  changed_docs_files="$(printf "%s\n" "$changed_files" | rg "$DOCS_PATHS_REGEX" || true)"

  local docs_skip_marker
  docs_skip_marker="$(extract_docs_skip_reason || true)"

  echo "triggering code files:"
  if [[ -n "$triggering_code_files" ]]; then
    printf "%s\n" "$triggering_code_files"
  else
    echo "(none)"
  fi

  echo "changed docs files:"
  if [[ -n "$changed_docs_files" ]]; then
    printf "%s\n" "$changed_docs_files"
  else
    echo "(none)"
  fi

  echo "docs-skip reason:"
  if [[ -n "$docs_skip_marker" ]]; then
    echo "$docs_skip_marker"
  elif [[ -n "$DOCS_SKIP_REASON" ]]; then
    echo "$DOCS_SKIP_REASON"
  else
    echo "(none)"
  fi

  if [[ -n "$triggering_code_files" ]] && [[ -z "$changed_docs_files" ]] && [[ -z "$docs_skip_marker" ]] && [[ -z "$DOCS_SKIP_REASON" ]]; then
    local remediation
    remediation="docs sync gate violated. add docs changes under docs/, README.md, or AGENTS.md; or include [docs-skip: <reason>] in commit/PR; or set DOCS_SKIP_REASON."
    if [[ "$DOCS_GATE_MODE" == "warn" ]]; then
      warn "$remediation"
    else
      fail "$remediation"
    fi
  fi

  echo "validate-repo: pass"
}

main "$@"
