#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROFILE="${RUSTSCAN_PROFILE:-release}"
MAX_FRAMES="${RUSTSCAN_MAX_FRAMES:-12}"
FRAME_STRIDE="${RUSTSCAN_FRAME_STRIDE:-2}"
MESH_VOXEL_SIZE="${RUSTSCAN_MESH_VOXEL_SIZE:-0.05}"
PREFER_HW="${RUSTSCAN_PREFER_HW:-false}"
COMPARE="${RUSTSCAN_COMPARE:-1}"

CARGO_FLAGS=()
if [[ "$PROFILE" == "release" ]]; then
  CARGO_FLAGS+=("--release")
fi

VIDEOS=(
  "sofa_sample_01"
  "sofa_sample_02"
  "sofa_sample_03"
)

OUTPUT_ROOT="$ROOT/output/examples"
EXPECTED_ROOT="$ROOT/test_data/expected"

mkdir -p "$OUTPUT_ROOT"

echo "Running RustScan example pipeline"
echo "  Profile:        $PROFILE"
echo "  Max frames:     $MAX_FRAMES"
echo "  Frame stride:   $FRAME_STRIDE"
echo "  Mesh voxel:     $MESH_VOXEL_SIZE"
echo "  Prefer HW:      $PREFER_HW"

for name in "${VIDEOS[@]}"; do
  video="$ROOT/test_data/video/${name}.MOV"
  output="$OUTPUT_ROOT/$name"
  expected="$EXPECTED_ROOT/$name"

  if [[ ! -f "$video" ]]; then
    echo "Missing video: $video" >&2
    exit 1
  fi

  mkdir -p "$output"
  echo "\n▶ Processing $name"

  (
    cd "$ROOT/RustSLAM"
    cargo run "${CARGO_FLAGS[@]}" --bin rustslam -- \
      --input "$video" \
      --output "$output" \
      --max-frames "$MAX_FRAMES" \
      --frame-stride "$FRAME_STRIDE" \
      --mesh-voxel-size "$MESH_VOXEL_SIZE" \
      --prefer-hardware "$PREFER_HW"
  )

  for file in mesh.obj mesh.ply results.json; do
    if [[ ! -f "$output/$file" ]]; then
      echo "Missing output file: $output/$file" >&2
      exit 1
    fi
  done

  if [[ "$COMPARE" == "1" ]]; then
    if [[ -f "$expected/mesh.obj" ]]; then
      python3 - <<'PY' "$expected/mesh.obj" "$output/mesh.obj"
import sys

expected, actual = sys.argv[1], sys.argv[2]

def counts(path: str) -> tuple[int, int]:
    v = f = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("v "):
                v += 1
            elif line.startswith("f "):
                f += 1
    return v, f

ev, ef = counts(expected)
av, af = counts(actual)
if (ev, ef) != (av, af):
    sys.stderr.write(
        f"Mesh count mismatch. expected v={ev}, f={ef}; got v={av}, f={af}\n"
    )
    sys.exit(1)
PY
      echo "  ✓ Mesh counts match expected"
    else
      echo "  ⚠ Expected mesh not found at $expected/mesh.obj (skipping compare)"
    fi
  fi

done

echo "\nAll examples completed. Outputs in $OUTPUT_ROOT"
