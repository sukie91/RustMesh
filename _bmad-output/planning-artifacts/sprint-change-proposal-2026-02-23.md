# Sprint Change Proposal — Code Review Findings

**Date:** 2026-02-23
**Triggered by:** Comprehensive code review across all RustSLAM modules
**Scope Classification:** Moderate — requires new stories and a new epic
**Recommended Approach:** Direct Adjustment (add stories to existing epics + new Epic 7)

---

## 1. Issue Summary

A comprehensive code review on 2026-02-23 examined all core RustSLAM modules (features, tracker, optimizer, loop_closing, fusion, pipeline). The review discovered 10 significant issues not previously documented in epics.md, while also confirming that several previously-reported issues had been fixed. The epics.md and CLAUDE.md documents have already been updated to reflect accurate code state.

**Key finding categories:**
- 4 critical algorithm correctness issues in SLAM pipeline (Epic 3)
- 1 critical rendering bug in 3DGS pipeline (Epic 4)
- 3 cross-cutting infrastructure issues affecting production readiness
- 2 major logic errors in tracker code

---

## 2. Impact Analysis

### Epic Impact

| Epic | Impact | Details |
|------|--------|---------|
| Epic 3 (SLAM) | HIGH | 4 new stories needed for ORB descriptors, matcher distance metric, relocalization, VO point updates |
| Epic 4 (3DGS) | MEDIUM | 1 new story for tiled_renderer dist⁴ bug fix |
| Epic 7 (NEW) | NEW | Cross-cutting infrastructure: Map thread safety, config validation, camera intrinsics |
| Epic 1/2/5/6 | NONE | No changes needed |

### Artifact Impact

| Artifact | Status | Action |
|----------|--------|--------|
| epics.md | ✅ Already updated | Code review findings added 2026-02-23 |
| CLAUDE.md | ✅ Already updated | Module status table corrected |
| sprint-status.yaml | ⚠️ Needs update | Add new stories and Epic 7 |
| architecture.md | ⚠️ Needs update | Map thread safety design, camera intrinsics config |
| prd.md | ✅ No change | MVP scope unchanged |

---

## 3. Recommended Approach: Direct Adjustment

### Rationale
- All existing code is functional (249/249 tests pass)
- Issues are quality/correctness improvements, not fundamental redesigns
- No rollback needed — code works, just needs targeted fixes
- MVP scope remains achievable after fixes

### New Stories for Epic 3 (SLAM Pipeline)

#### Story 3.7: Implement True Binary BRIEF Descriptors for ORB
**Priority:** P0 — Critical for matching quality
**Files:** `features/orb.rs`, `features/utils.rs`

**Current:** `build_patch_descriptors()` samples 32 bytes of raw intensity values. `wta_k` parameter exists but unused.
**Required:** Implement 256-bit binary descriptors using intensity pair comparisons (standard BRIEF pattern). Apply rotation from keypoint orientation (already computed via intensity centroid).

**Acceptance Criteria:**
- 256 pre-computed random point pairs for binary tests
- Each descriptor is 32 bytes (256 bits) of binary comparisons
- Rotated pairs based on keypoint angle
- Existing tests updated, new tests for descriptor discriminability

---

#### Story 3.8: Fix KnnMatcher Distance Metric for Binary Descriptors
**Priority:** P0 — Critical for matching correctness
**Files:** `features/knn_matcher.rs`

**Current:** Uses `SquaredEuclidean` distance on u8 descriptors converted to f64. Wrong metric for binary descriptors.
**Required:** Either switch to Hamming distance for binary descriptors, or add type-safe dispatch that selects correct metric based on descriptor type.

**Acceptance Criteria:**
- KnnMatcher uses Hamming distance when matching binary (ORB) descriptors
- Euclidean distance preserved for float descriptors (SIFT/SURF)
- Ratio test still works correctly with new metric
- Performance not degraded

---

#### Story 3.9: Implement Functional Relocalization
**Priority:** P1 — Required for robust long-sequence tracking
**Files:** `loop_closing/relocalization.rs`

**Current:** `try_pnp()` returns `success: false` (stub). `relocalize_essential()` also returns failed.
**Required:** Implement relocalization using BoW-based candidate retrieval + PnP pose estimation.

**Acceptance Criteria:**
- Query BoW database for similar keyframes when tracking is lost
- Attempt PnP pose estimation against top-N candidates
- Return valid pose if relocalization succeeds
- Graceful fallback if relocalization fails

---

#### Story 3.10: Add 3D Point Updates During VO Tracking
**Priority:** P0 — Critical for long-sequence tracking
**Files:** `tracker/vo.rs`

**Current:** `prev_3d_points` set only during initialization (line 280), never updated during tracking. Visible points decrease over time until tracking fails.
**Required:** Triangulate new 3D points from matched features during tracking, update map with new observations.

**Acceptance Criteria:**
- New 3D points triangulated from tracking matches
- Existing 3D points updated with new observations
- Map grows as camera explores new areas
- Tracking maintains >95% success rate on test sequences

---

#### Story 3.11: Fix PnP RANSAC Fallback and VO Init Logic
**Priority:** P1 — Prevents garbage results
**Files:** `tracker/solver.rs`, `tracker/vo.rs`

**Current:**
- `solver.rs:157-163`: When RANSAC+DLT both fail, returns identity pose with ALL points as inliers
- `vo.rs:257`: Uses OR instead of AND for initialization check

**Required:**
- Return failure (empty result) when both RANSAC and DLT fail
- Change OR to AND in initialization condition

**Acceptance Criteria:**
- PnP solver returns error/empty when no valid pose found
- VO initialization requires BOTH sufficient inliers AND sufficient matches
- No garbage poses propagated to downstream modules

---

### New Story for Epic 4 (3DGS Training)

#### Story 4.7: Fix TiledRenderer Gaussian Kernel Distance Bug
**Priority:** P0 — Critical for rendering quality
**Files:** `fusion/tiled_renderer.rs`

**Current (line 271):**
```rust
let dist = (g.cov_xx * dx * dx + 2.0 * g.cov_xy * dx * dy + g.cov_yy * dy * dy).sqrt();
let weight = (-0.5 * dist * dist).exp() * g.opacity;
```
This computes `exp(-0.5 * d⁴)` instead of `exp(-0.5 * d²)`. The Mahalanobis quadratic form is already d², so the sqrt is wrong.

**Required:** Remove the `.sqrt()` call:
```rust
let dist_sq = g.cov_xx * dx * dx + 2.0 * g.cov_xy * dx * dy + g.cov_yy * dy * dy;
let weight = (-0.5 * dist_sq).exp() * g.opacity;
```

**Acceptance Criteria:**
- Gaussian kernel uses correct quadratic decay exp(-0.5 * d²)
- Rendering quality visually improved (wider, smoother Gaussians)
- Existing tests updated to reflect correct behavior

---

### New Epic 7: Cross-Cutting Infrastructure Fixes

#### Story 7.1: Add Thread Safety to Map Struct
**Priority:** P0 — Required for multi-threaded pipeline
**Files:** `core/map.rs`, `core/map_point.rs`, `pipeline/realtime.rs`

**Current:** `Map` uses plain `HashMap` with non-atomic `u64` ID counters. Used in multi-threaded `realtime.rs` without synchronization.
**Required:** Wrap Map in `Arc<RwLock<Map>>`, use `AtomicU64` for ID counters.

**Acceptance Criteria:**
- Map struct uses `AtomicU64` for `next_point_id` and `next_keyframe_id`
- `realtime.rs` accesses Map through `Arc<RwLock<Map>>`
- No data races under concurrent access
- All existing tests pass

---

#### Story 7.2: Unify Camera Intrinsics from Configuration
**Priority:** P1 — Prevents inconsistent behavior
**Files:** `config/params.rs`, `tracker/vo.rs`, `loop_closing/closing.rs`, `cli/mod.rs`

**Current:** Camera intrinsics hardcoded in multiple places:
- `fx=525, fy=525` in SLAM tracker
- `(500, 500, 320, 240)` in loop closing global BA
- No config-driven intrinsics

**Required:** Single source of truth for camera intrinsics in config, passed to all modules.

**Acceptance Criteria:**
- Camera intrinsics defined in TOML config
- All modules read intrinsics from config (no hardcoded values)
- Auto-detection from video metadata as optional enhancement

---

#### Story 7.3: Add Configuration Parameter Validation
**Priority:** P2 — Prevents invalid configurations
**Files:** `config/params.rs`, `config/mod.rs`

**Current:** All config structs are plain data with `Default` implementations. No validation.
**Required:** Add validation logic for parameter ranges and constraints.

**Acceptance Criteria:**
- `voxel_size > 0`, `max_features > 0`, `max_features > min_features`
- Validation runs at config load time with clear error messages
- Invalid configs rejected before pipeline starts

---

## 4. Sprint Status Updates

### New entries to add to sprint-status.yaml:

```yaml
# Epic 3 additions:
3-7-implement-true-binary-brief-descriptors: backlog
3-8-fix-knnmatcher-distance-metric: backlog
3-9-implement-functional-relocalization: backlog
3-10-add-3d-point-updates-during-vo-tracking: backlog
3-11-fix-pnp-ransac-fallback-and-vo-init-logic: backlog

# Epic 4 addition:
4-7-fix-tiled-renderer-gaussian-kernel-distance-bug: backlog

# New Epic 7:
epic-7: backlog
7-1-add-thread-safety-to-map-struct: backlog
7-2-unify-camera-intrinsics-from-configuration: backlog
7-3-add-configuration-parameter-validation: backlog
epic-7-retrospective: optional
```

### Recommended implementation order:

1. **Story 4.7** (tiled_renderer fix) — smallest, highest impact, one-line fix
2. **Story 3.11** (PnP fallback + VO init) — small fix, prevents garbage results
3. **Story 3.7** (ORB BRIEF) — foundational for matching quality
4. **Story 3.8** (KnnMatcher metric) — depends on 3.7
5. **Story 7.1** (Map thread safety) — foundational for multi-threaded pipeline
6. **Story 3.10** (VO 3D point updates) — critical for long sequences
7. **Story 7.2** (Camera intrinsics) — cleanup
8. **Story 3.9** (Relocalization) — nice-to-have for robustness
9. **Story 7.3** (Config validation) — polish

---

## 5. Implementation Handoff

**Scope:** Moderate — new stories within existing structure + one new epic

**Handoff plan:**
- **Development team:** Execute stories 3.7-3.11, 4.7, 7.1-7.3
- **Scrum Master (Bob):** Update sprint-status.yaml, create story files via `/bmad-bmm-create-story`
- **Architect (Winston):** Review Map thread safety design (Story 7.1) before implementation

**Success criteria:**
- All new stories implemented and tested
- 249+ lib tests pass (no regressions)
- Epic 3 and Epic 4 can be marked `done`
- Epic 7 completed
- Quality standards met: PSNR > 28dB, tracking > 95%

---

## 6. Epics.md Update Required

The following changes need to be made to `epics.md`:
- Add Stories 3.7-3.11 under Epic 3
- Add Story 4.7 under Epic 4
- Add Epic 7 with Stories 7.1-7.3
- Update Epic 3 status to reflect new stories
