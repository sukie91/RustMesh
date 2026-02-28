#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_IMPL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DEFAULT_SPRINT_FILE="$DEFAULT_IMPL_DIR/sprint-status.yaml"

SPRINT_FILE="$DEFAULT_SPRINT_FILE"
IMPL_DIR="$DEFAULT_IMPL_DIR"
declare -a EPICS=()

usage() {
  cat <<USAGE
Usage: $(basename "$0") [--sprint-file PATH] [--impl-dir PATH] [--epic N]...

Checks status sync between sprint-status.yaml and story markdown files.
Story filename convention: <story-key>.md (e.g., 3-7-implement-... .md)
Story doc status convention: first 'Status: <value>' line in file.

Examples:
  $(basename "$0") --epic 3 --epic 7
  $(basename "$0") --sprint-file path/to/sprint-status.yaml --impl-dir path/to/artifacts
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sprint-file)
      SPRINT_FILE="$2"; shift 2 ;;
    --impl-dir)
      IMPL_DIR="$2"; shift 2 ;;
    --epic)
      EPICS+=("$2"); shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2 ;;
  esac
done

if [[ ! -f "$SPRINT_FILE" ]]; then
  echo "ERROR: sprint status file not found: $SPRINT_FILE" >&2
  exit 2
fi

normalize_status() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | xargs
}

issue_count=0
checked_count=0

# Extract story key + status lines, e.g. "3-7-foo: done"
while IFS='|' read -r key sprint_status; do
  [[ -z "$key" ]] && continue

  epic_num="${key%%-*}"
  if [[ ${#EPICS[@]} -gt 0 ]]; then
    keep=0
    for e in "${EPICS[@]}"; do
      if [[ "$epic_num" == "$e" ]]; then
        keep=1
        break
      fi
    done
    [[ $keep -eq 0 ]] && continue
  fi

  checked_count=$((checked_count + 1))
  story_file="$IMPL_DIR/$key.md"

  if [[ ! -f "$story_file" ]]; then
    echo "MISSING_FILE|$key|sprint=$sprint_status|file=$story_file"
    issue_count=$((issue_count + 1))
    continue
  fi

  doc_status_line="$(rg -n "^Status:\\s*" "$story_file" | head -n1 || true)"
  if [[ -z "$doc_status_line" ]]; then
    echo "MISSING_DOC_STATUS|$key|sprint=$sprint_status|file=$story_file"
    issue_count=$((issue_count + 1))
    continue
  fi

  doc_status="$(echo "$doc_status_line" | sed -E 's/^[0-9]+:Status:[[:space:]]*//')"
  sprint_norm="$(normalize_status "$sprint_status")"
  doc_norm="$(normalize_status "$doc_status")"

  if [[ "$sprint_norm" != "$doc_norm" ]]; then
    echo "STATUS_MISMATCH|$key|sprint=$sprint_norm|doc=$doc_norm|file=$story_file"
    issue_count=$((issue_count + 1))
  fi

done < <(awk '
  /^development_status:/ { in_dev=1; next }
  in_dev && /^[[:space:]]+[0-9]+-[0-9]+-[^:]+:[[:space:]]*/ {
    line=$0
    sub(/^[[:space:]]+/, "", line)
    split(line, parts, ":")
    key=parts[1]
    status=parts[2]
    sub(/^[[:space:]]+/, "", status)
    print key "|" status
  }
' "$SPRINT_FILE")

if [[ $issue_count -eq 0 ]]; then
  echo "OK|checked=$checked_count|issues=0"
else
  echo "FAIL|checked=$checked_count|issues=$issue_count"
  exit 1
fi
