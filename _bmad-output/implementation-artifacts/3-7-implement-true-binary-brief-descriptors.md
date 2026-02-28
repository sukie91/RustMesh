# Story 3.7: Implement True Binary BRIEF Descriptors for ORB

Status: done

## Status Reconciliation Note

- Date: 2026-02-23
- This story status is aligned to `_bmad-output/implementation-artifacts/sprint-status.yaml`,
  where `3-7-implement-true-binary-brief-descriptors` is tracked as `done`.
- Historical checklist items in this artifact were not maintained during brownfield migration and
  should not be treated as authoritative execution evidence by themselves.

## Story

As a developer,
I want ORB to produce true binary BRIEF descriptors (256-bit binary comparison tests),
So that feature matching has proper discriminative power and rotation invariance, enabling accurate Hamming-distance-based matching.

## Acceptance Criteria

1. Each ORB descriptor is 32 bytes (256 bits) of binary intensity comparison results
2. Descriptors use 256 pre-defined random point pairs within a 31×31 patch
3. For each pair (p1, p2): bit = 1 if I(p1) > I(p2), else bit = 0
4. Point pairs are rotated by keypoint orientation angle before sampling
5. Descriptors are compatible with Hamming distance matching (HammingMatcher)
6. All existing tests pass (updated as needed for new descriptor format)
7. New tests validate binary descriptor properties (bit distribution, rotation behavior)

## Tasks / Subtasks

- [ ] Task 1: Define BRIEF point pairs constant (AC: #2)
  - [ ] 1.1 Create `BRIEF_PAIRS: [(i8, i8, i8, i8); 256]` in `utils.rs` — 256 pairs of (x1,y1,x2,y2) within ±15 pixel range (31×31 patch)
  - [ ] 1.2 Use Gaussian-distributed random pairs (standard BRIEF pattern) or port OpenCV's learned pairs
- [ ] Task 2: Implement `build_brief_descriptors()` function (AC: #1, #3, #4)
  - [ ] 2.1 Replace `build_patch_descriptors()` in `utils.rs` with `build_brief_descriptors()`
  - [ ] 2.2 For each keypoint: rotate all 256 pairs by `kp.angle`, sample two intensities, compare, pack bit
  - [ ] 2.3 Pack 256 bits into 32 bytes (bit 0 of byte 0 = first test, bit 7 of byte 0 = 8th test, etc.)
  - [ ] 2.4 Keep `ORB_DESCRIPTOR_SIZE = 32` unchanged (already correct)
- [ ] Task 3: Update all callers to use new descriptor function (AC: #6)
  - [ ] 3.1 Update `OrbExtractor::detect_and_compute()` in `orb.rs:130-133`
  - [ ] 3.2 Update `HarrisExtractor::detect_and_compute()` in `pure_rust.rs:427`
  - [ ] 3.3 Update `FastExtractor::detect_and_compute()` in `pure_rust.rs:500`
- [ ] Task 4: Update tests (AC: #6, #7)
  - [ ] 4.1 Update `test_build_patch_descriptors_respects_keypoint_angle` in `utils.rs`
  - [ ] 4.2 Update `test_harris_extractor_descriptors` and `test_fast_extractor_descriptors` in `pure_rust.rs`
  - [ ] 4.3 Update `test_orb_fallback_extracts_and_distributes_keypoints` in `orb.rs`
  - [ ] 4.4 Add new test: verify descriptors are binary (each byte is a packed bitfield, not raw intensity)
  - [ ] 4.5 Add new test: verify Hamming distance between identical keypoints is 0
  - [ ] 4.6 Add new test: verify rotation changes descriptor bits (not just intensity values)
- [ ] Task 5: Run full test suite and verify (AC: #6)
  - [ ] 5.1 `cargo test --lib` — all 249+ tests pass
  - [ ] 5.2 `cargo build --release` — no warnings

## Dev Notes

### Problem Statement

Current `build_patch_descriptors()` in `utils.rs:14-65` samples 32 raw intensity values at fixed grid positions (`PATCH_OFFSETS`). Each descriptor byte is a raw pixel intensity (0-255), NOT a binary comparison result. This means:
- Descriptors lack discriminative power (raw intensities are noisy)
- Hamming distance on raw intensities is meaningless
- KnnMatcher's Euclidean distance partially works by accident but is suboptimal
- HammingMatcher produces garbage results on non-binary descriptors

### Implementation Approach

**BRIEF (Binary Robust Independent Elementary Features):**
- 256 pre-defined point pairs within a 31×31 patch centered on keypoint
- Each pair (p1, p2): compare `I(p1)` vs `I(p2)`, produce 1 bit
- 256 bits → 32 bytes per descriptor
- For rotation invariance (rBRIEF): rotate pair coordinates by keypoint angle before sampling

**Bit packing convention:**
```rust
// For pair index i (0..256):
let byte_idx = i / 8;
let bit_idx = i % 8;
if intensity_at_p1 > intensity_at_p2 {
    descriptor[byte_idx] |= 1 << bit_idx;
}
```

**Point pair generation:**
- Use Gaussian distribution with σ = patch_size/5 ≈ 6.2 for a 31×31 patch
- Coordinates clamped to [-15, 15] range
- Use a fixed seed for reproducibility (same pairs every run)
- Alternative: port the 256 learned pairs from OpenCV's ORB implementation

### Key Code Locations

| File | What to change |
|------|---------------|
| `RustSLAM/src/features/utils.rs` | Replace `PATCH_OFFSETS` + `build_patch_descriptors()` with `BRIEF_PAIRS` + `build_brief_descriptors()` |
| `RustSLAM/src/features/orb.rs:130-133` | Call `build_brief_descriptors` instead of `build_patch_descriptors` |
| `RustSLAM/src/features/pure_rust.rs:427,500` | Same — update Harris/FAST extractor `detect_and_compute` |
| `RustSLAM/src/features/base.rs:6` | `ORB_DESCRIPTOR_SIZE = 32` — no change needed |

### Critical Constraints

- `ORB_DESCRIPTOR_SIZE` must remain 32 (already correct for 256-bit BRIEF)
- `Descriptors.data` is `Vec<u8>` — each byte now holds 8 packed bits instead of 1 raw intensity
- `HammingMatcher` in `hamming_matcher.rs` already implements correct Hamming distance with popcount LUT — it will work correctly with true binary descriptors
- `KnnMatcher` in `knn_matcher.rs` uses SquaredEuclidean on f64 — this is WRONG for binary descriptors but is a separate story (3.8). Do NOT fix KnnMatcher in this story.
- The `angle` field on `KeyPoint` defaults to `-1.0` (meaning "no orientation"). The rotation code in current `build_patch_descriptors` checks `kp.angle >= 0.0` — preserve this guard.
- Gaussian smoothing of the patch before sampling improves BRIEF stability. Apply a small Gaussian blur (σ=2) to the patch region before binary tests. This is standard practice from the BRIEF paper.

### Anti-Patterns to Avoid

- Do NOT change `ORB_DESCRIPTOR_SIZE` — it's already 32
- Do NOT modify `HammingMatcher` — it already works correctly for binary descriptors
- Do NOT modify `KnnMatcher` — that's Story 3.8
- Do NOT change the `FeatureExtractor` trait or `Descriptors` struct
- Do NOT add new dependencies — use `std` only for random pair generation
- Do NOT use `rand` crate — use a deterministic algorithm (e.g., simple LCG with fixed seed) to generate the 256 pairs as a compile-time constant

### Existing Code Patterns

- Rotation is already applied in `utils.rs:39-48` using `sin_cos()` — reuse this pattern
- Boundary checking pattern: `if px >= 0 && px < w && py >= 0 && py < h` — reuse
- Descriptor allocation: `Descriptors::with_capacity(count, size)` — reuse
- Test pattern: checkerboard image generation in `pure_rust.rs:582-593` and `orb.rs:346-357`

### Downstream Impact

After this story, descriptors will be true binary. This affects:
- `HammingMatcher`: Will work BETTER (designed for binary descriptors) ✅
- `KnnMatcher`: Will work WORSE (Euclidean on binary is wrong) — fixed in Story 3.8
- `VO tracker` (`vo.rs`): Uses matcher results — no code change needed
- `Loop closing` (`detector.rs`): BoW vocabulary hashes first 8 bytes — binary descriptors change hash distribution but still functional

### Project Structure Notes

- All changes within `RustSLAM/src/features/` module
- No new files needed — modify existing `utils.rs`, `orb.rs`, `pure_rust.rs`
- Module re-exports in `mod.rs` unchanged
- `utils` module is `mod utils` (private) — only accessed within `features/`

### References

- [Source: RustSLAM/src/features/utils.rs] — current `build_patch_descriptors()` and `PATCH_OFFSETS`
- [Source: RustSLAM/src/features/orb.rs:130-133] — ORB descriptor computation call site
- [Source: RustSLAM/src/features/pure_rust.rs:427,500] — Harris/FAST descriptor call sites
- [Source: RustSLAM/src/features/hamming_matcher.rs:25-31] — Hamming distance with popcount LUT
- [Source: RustSLAM/src/features/base.rs:6] — `ORB_DESCRIPTOR_SIZE = 32`
- [Source: _bmad-output/planning-artifacts/epics.md#Story 3.7] — Story definition and acceptance criteria
- [Source: _bmad-output/planning-artifacts/architecture.md#ADR-004] — glam-only math, no new deps

## Dev Agent Record

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List
