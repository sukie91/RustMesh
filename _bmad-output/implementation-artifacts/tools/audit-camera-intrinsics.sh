#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-RustSLAM/src}"

if [[ ! -d "$ROOT" ]]; then
  echo "ERROR: source directory not found: $ROOT" >&2
  exit 2
fi

# Heuristic audit for hardcoded camera intrinsics in non-test Rust files.
# This is a signal, not a proof.
rg -n "500\.0|320\.0|240\.0|525\.0|319\.5|239\.5|focal length ~500|Default intrinsics" \
  "$ROOT" \
  --glob '*.rs' \
  --glob '!**/*test*.rs' \
  --glob '!**/tests.rs' \
  --glob '!**/target/**' \
  || true
