# Story 7.1: Add Thread Safety to Map Struct

Status: in-progress

## Story Key

- `7-1-add-thread-safety-to-map-struct`

## Canonical Sources

- Sprint status: `_bmad-output/implementation-artifacts/sprint-status.yaml`
- Story definition: `_bmad-output/planning-artifacts/epics.md`

## Acceptance Focus

- ID counters use `AtomicU64`
- Realtime path accesses map via `Arc<RwLock<Map>>`
- No data races under concurrent access
- Existing tests remain passing

## Available Evidence in Artifacts

- Story status is `done` in sprint tracking.
- Story requirement and acceptance criteria are documented in planning artifacts.

## Evidence Gaps

- No implementation diff, review transcript, or QA execution record is currently attached in implementation artifacts.

## Follow-up Verification Needed

- Link code locations proving atomic counters and lock strategy.
- Add concurrency test evidence or test report reference.

## Tasks/Subtasks

### Review Follow-ups (AI)

- [ ] [AI-Review][HIGH] Ensure `Map::add_point()` writes generated atomic ID back into `MapPoint.id` before insert, preventing key/object ID divergence. [RustSLAM/src/core/map.rs:27]
- [ ] [AI-Review][HIGH] In `LocalMapping::add_map_point()`, use the actual ID returned by `map.add_point()` to update `local_map_points` tracking. [RustSLAM/src/mapping/local_mapping.rs:329]
- [ ] [AI-Review][MEDIUM] Initialize/sync `LocalMapping.next_point_id` from shared map state when using `set_map_and_init` / `set_shared_map_and_init` to avoid stale local counters. [RustSLAM/src/mapping/local_mapping.rs:125]
- [ ] [AI-Review][MEDIUM] Add multithreaded tests to validate map access under `Arc<RwLock<Map>>` and detect ID consistency races across threads. [RustSLAM/src/pipeline/realtime.rs:812]
- [ ] [AI-Review][MEDIUM] Add Dev Agent Record and File List to this story to support claim-vs-git auditability. [_bmad-output/implementation-artifacts/7-1-add-thread-safety-to-map-struct.md:1]
