# Story 7.2: Unify Camera Intrinsics from Configuration

Status: in-progress

## Story Key

- `7-2-unify-camera-intrinsics-from-configuration`

## Canonical Sources

- Sprint status: `_bmad-output/implementation-artifacts/sprint-status.yaml`
- Story definition: `_bmad-output/planning-artifacts/epics.md`

## Acceptance Focus

- Intrinsics loaded from TOML configuration
- VO/BA/loop closing/3DGS consume consistent intrinsics source
- No hardcoded intrinsics remain in active paths

## Available Evidence in Artifacts

- Story status is `done` in sprint tracking.
- Target behavior is documented in planning artifacts.

## Evidence Gaps

- Module-by-module consumption trace is not attached in implementation artifacts.

## Follow-up Verification Needed

- Add a module matrix showing intrinsics source for VO, BA, loop closing, and 3DGS.
- Attach proof that hardcoded fallback values were removed or isolated.

## Tasks/Subtasks

### Review Follow-ups (AI)

- [ ] [AI-Review][HIGH] Route loop-closure BA intrinsics from runtime config and remove default camera fallback in constructor. [RustSLAM/src/loop_closing/closing.rs:19]
- [ ] [AI-Review][HIGH] Wire production call sites to invoke `set_ba_intrinsics()` so configured intrinsics are actually used. [RustSLAM/src/loop_closing/closing.rs:34]
- [ ] [AI-Review][HIGH] Replace GaussianTracker default-camera-derived intrinsics with explicit config-backed initialization path. [RustSLAM/src/fusion/tracker.rs:59]
- [ ] [AI-Review][HIGH] Remove `CameraConfig::default()` intrinsics bootstrapping from SparseDenseSlam and inject runtime camera config. [RustSLAM/src/fusion/slam_integrator.rs:147]
- [ ] [AI-Review][HIGH] Replace heuristic `width*1.2` intrinsics in VideoLoader with config/video-metadata-backed values. [RustSLAM/src/io/video_loader.rs:116]
- [ ] [AI-Review][MEDIUM] Eliminate LocalMapping default-intrinsics fallback when camera is unset; fail fast or require injected camera config. [RustSLAM/src/mapping/local_mapping.rs:274]
- [ ] [AI-Review][MEDIUM] Add Dev Agent Record and File List to this story so implementation claims can be audited against git reality. [_bmad-output/implementation-artifacts/7-2-unify-camera-intrinsics-from-configuration.md:1]
- [ ] [AI-Review][MEDIUM] Reconcile this story’s documentation with current changed source files under `RustSLAM/src/` to restore review traceability. [_bmad-output/implementation-artifacts/7-2-unify-camera-intrinsics-from-configuration.md:1]
- [ ] [AI-Review][LOW] Add integration test asserting VO/BA/loop-closing/3DGS consume the same configured intrinsics end-to-end. [RustSLAM/src/cli/integration_tests.rs:1]
