# Camera Intrinsics Verification (P1)

- Date: 2026-02-23
- Scope: Verify whether runtime paths use a single configurable camera-intrinsics source across VO, BA, loop closing, and 3DGS-related modules.
- Sources:
  - `_bmad-output/implementation-artifacts/intrinsics-audit-report-2026-02-23.txt`
  - `RustSLAM/src/*`

## Changes Applied in This Pass

1. Local mapping triangulation now uses configured intrinsics (or explicit default fallback) instead of `500.0`.
- `RustSLAM/src/mapping/local_mapping.rs`

2. Loop-closing global BA intrinsics are now configurable via `LoopClosing::set_ba_intrinsics(...)`.
- `RustSLAM/src/loop_closing/closing.rs`

3. Sparse-dense SLAM integrator now uses internal intrinsics fields for keyframe creation and rendering.
- `RustSLAM/src/fusion/slam_integrator.rs`

4. Gaussian tracker now uses instance intrinsics and image width for ICP pixel backprojection (no hardcoded `640` indexing, no hardcoded `500/320/240`).
- `RustSLAM/src/fusion/tracker.rs`

## Verification Matrix

| Module Area | Current Intrinsics Source | Status | Evidence |
|---|---|---|---|
| VO (realtime pipeline) | Dataset camera passed into `VisualOdometry::new(camera)` | Pass | `RustSLAM/src/pipeline/realtime.rs:273`, `RustSLAM/src/tracker/vo.rs:108` |
| Local BA (realtime) | `frame.camera` intrinsics | Pass | `RustSLAM/src/pipeline/realtime.rs:689` |
| Local mapping triangulation | `self.camera` or explicit default fallback | Pass (fixed) | `RustSLAM/src/mapping/local_mapping.rs:262`, `RustSLAM/src/mapping/local_mapping.rs:301` |
| Loop-closing global BA | Configurable field (`set_ba_intrinsics`) with default fallback | Partial (fixed API, caller wiring pending) | `RustSLAM/src/loop_closing/closing.rs:34`, `RustSLAM/src/loop_closing/closing.rs:116` |
| 3DGS training targets (realtime) | `frame.camera` intrinsics | Pass | `RustSLAM/src/pipeline/realtime.rs:568`, `RustSLAM/src/pipeline/realtime.rs:608` |
| Sparse-dense integrator render/keyframe | `self.fx/self.fy/self.cx/self.cy` | Pass (fixed) | `RustSLAM/src/fusion/slam_integrator.rs:220`, `RustSLAM/src/fusion/slam_integrator.rs:313` |
| Gaussian tracker ICP | Tracker instance intrinsics + image width indexing | Pass (fixed) | `RustSLAM/src/fusion/tracker.rs:59`, `RustSLAM/src/fusion/tracker.rs:174` |

## Remaining Risk / Follow-up

1. Loop-closing API now supports runtime intrinsics injection, but call sites that create `LoopClosing` still use `new()` and do not yet call `set_ba_intrinsics(...)`.
2. Heuristic audit still reports constants in many files; several are defaults, examples, or test-only code paths embedded in non-test files.
3. If strict “single source only” is required, a next step is to enforce intrinsics injection at constructor level (instead of optional setter + defaults).

## Audit Artifact

- Audit script: `_bmad-output/implementation-artifacts/tools/audit-camera-intrinsics.sh`
- Output report: `_bmad-output/implementation-artifacts/intrinsics-audit-report-2026-02-23.txt`

## Targeted Regression Checks Run

- `cargo test --lib loop_closing::closing::tests::test_loop_closing_empty_map`
- `cargo test --lib loop_closing::closing::tests::test_loop_closing_with_minimal_keyframes`
- `cargo test --lib mapping::local_mapping::tests::test_local_mapping_creation`
- `cargo test --lib fusion::slam_integrator::tests::test_slam_creation`
- `cargo test --lib fusion::slam_integrator::tests::test_slam_integrator`
- `cargo test --lib fusion::tracker::tests::test_tracker_creation`
- `cargo test --lib fusion::tracker::tests::test_tracking_empty_map`
- `cargo test --lib fusion::tracker::tests::test_tracking_with_gaussians`

All above tests passed in this run.

