# SLAM-to-3DGS Contract Validation (Epic 3 -> Epic 4)

- Date: 2026-02-23
- Owner: Dana (QA Engineer)
- Scope: Validate Epic 3 assumptions consumed by Epic 4 (3DGS training and scene generation).
- Requirement references:
  - `_bmad-output/planning-artifacts/epics.md:954` (Story 3.8)
  - `_bmad-output/planning-artifacts/epics.md:973` (Story 3.9)
  - `_bmad-output/planning-artifacts/epics.md:992` (Story 3.10)

## Validation Matrix

| Contract assumption | Evidence (code) | Evidence (tests) | Result |
| --- | --- | --- | --- |
| Descriptor/matcher compatibility for binary ORB path | `RustSLAM/src/tracker/vo.rs:103` uses `HammingMatcher`; `RustSLAM/src/features/knn_matcher.rs:218` dispatches by metric with `Hamming` default at `RustSLAM/src/features/knn_matcher.rs:23`; relocalization uses Hamming ratio matching at `RustSLAM/src/loop_closing/relocalization.rs:131` | `cargo test --lib features::knn_matcher::tests::` -> 12 passed; `cargo test --lib features::hamming_matcher::tests::` -> 1 passed | PASS |
| Relocalization functional path quality | BoW candidate retrieval + PnP path implemented at `RustSLAM/src/loop_closing/relocalization.rs:75` and `RustSLAM/src/loop_closing/relocalization.rs:117`; however VO lost-tracking path currently resets state at `RustSLAM/src/tracker/vo.rs:393` and does not call `Relocalizer` | `cargo test --lib loop_closing::relocalization::tests::` -> 6 passed | PARTIAL (module PASS, runtime integration GAP) |
| VO 3D-point propagation behavior | New 3D points and propagation logic in `RustSLAM/src/tracker/vo.rs:334` and `RustSLAM/src/tracker/vo.rs:365`; initialization triangulation at `RustSLAM/src/tracker/vo.rs:260` | Indirect coverage: `cargo test --lib fusion::tracker::tests::test_tracking_with_gaussians` -> passed | PASS with limited direct VO test depth |
| Epic 4 consumption contract (poses/intrinsics into BA + 3DGS) | Pose flows tracking -> mapping -> optimization via `RustSLAM/src/pipeline/realtime.rs:294`, `RustSLAM/src/pipeline/realtime.rs:385`, `RustSLAM/src/pipeline/realtime.rs:458`; 3DGS camera intrinsics from frame camera at `RustSLAM/src/pipeline/realtime.rs:568`; BA intrinsics from frame camera at `RustSLAM/src/pipeline/realtime.rs:689` | `cargo test --lib pipeline::realtime::tests::test_message_sending` -> passed; `cargo test --lib pipeline::realtime::tests::test_ba_threshold_logic` -> passed; `cargo test --lib fusion::slam_integrator::tests::test_slam_integrator` -> passed | PASS |

## Commands Executed

```bash
cd RustSLAM
cargo test --lib features::knn_matcher::tests:: -- --nocapture
cargo test --lib features::hamming_matcher::tests:: -- --nocapture
cargo test --lib loop_closing::relocalization::tests:: -- --nocapture
cargo test --lib pipeline::realtime::tests::test_message_sending -- --nocapture
cargo test --lib pipeline::realtime::tests::test_ba_threshold_logic -- --nocapture
cargo test --lib fusion::tracker::tests::test_tracking_with_gaussians -- --nocapture
cargo test --lib fusion::slam_integrator::tests::test_slam_integrator -- --nocapture
```

## Findings

1. Descriptor-matcher contract needed by Epic 4 is satisfied in active runtime paths (VO and relocalization).
2. Dedicated relocalization module is implemented and unit-tested, but `VisualOdometry` lost-tracking branch still performs reset-only fallback and does not invoke relocalization recovery.
3. Pose and intrinsics handoff into BA/3DGS training paths is consistent with Epic 4 dependency assumptions.

## Decision

- Contract validation activity: COMPLETE.
- Epic 4 continuity readiness: CONDITIONAL.
- Blocking risk to monitor: runtime relocalization integration gap in `RustSLAM/src/tracker/vo.rs:393`.

## Follow-up

- Add a follow-up implementation story/task to integrate `Relocalizer` into VO lost-tracking runtime flow (not only module-level availability).
