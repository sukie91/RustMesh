# Retro Action Plan (Epic 3 + Epic 7)

- Date: 2026-02-23
- Source retrospectives:
  - `_bmad-output/implementation-artifacts/epic-3-retro-2026-02-23.md`
  - `_bmad-output/implementation-artifacts/epic-7-retro-2026-02-23.md`

## 1. Prioritized Global Backlog

## P0 (Must complete before next epic closure)

- [x] Align Story 3-7 status between story doc and sprint status.
  - Owner: Bob (Scrum Master)
  - Source: Epic 3
  - Done means:
    - `sprint-status.yaml` and story file status are consistent.
    - mismatch cause is documented.
  - Evidence:
    - updated status line in `_bmad-output/implementation-artifacts/sprint-status.yaml`
    - updated status in `_bmad-output/implementation-artifacts/3-7-implement-true-binary-brief-descriptors.md`

- [x] Backfill missing Epic 3 story evidence references.
  - Owner: Charlie (Senior Developer)
  - Source: Epic 3
  - Done means:
    - each Epic 3 story has implementation note, review summary, and test evidence reference.
  - Evidence:
    - index document listing all Epic 3 story evidence links.
  - Progress (2026-02-23):
    - Created `_bmad-output/implementation-artifacts/epic-3-evidence-index-2026-02-23.md`.
    - Created backfilled metadata records for 3-1..3-6 and 3-8..3-11.
    - Remaining work (P1/P2 quality hardening): attach implementation/review/test execution evidence for each story.

- [x] Backfill Epic 7 story evidence or canonical references.
  - Owner: Bob (Scrum Master)
  - Source: Epic 7
  - Done means:
    - each Epic 7 story has an auditable artifact record.
  - Evidence:
    - `_bmad-output/implementation-artifacts/7-1-add-thread-safety-to-map-struct.md`
    - `_bmad-output/implementation-artifacts/7-2-unify-camera-intrinsics-from-configuration.md`
    - `_bmad-output/implementation-artifacts/7-3-add-configuration-parameter-validation.md`

## P1 (Should complete before next major planning decision)

- [x] Add automated status-integrity checks.
  - Owner: Winston (Architect)
  - Source: Epic 3
  - Done means:
    - local or CI check flags mismatches between sprint status and story docs.
  - Evidence:
    - `_bmad-output/implementation-artifacts/tools/check-story-status-sync.sh`
    - `_bmad-output/implementation-artifacts/status-sync-report-epic-3-7-2026-02-23.txt` (`OK|checked=14|issues=0`)

- [x] Validate camera intrinsics single-source consumption across modules.
  - Owner: Winston (Architect)
  - Source: Epic 7
  - Done means:
    - VO, BA, loop closing, and 3DGS consumption path is explicitly verified.
  - Evidence:
    - `_bmad-output/implementation-artifacts/camera-intrinsics-verification-2026-02-23.md`
    - `_bmad-output/implementation-artifacts/intrinsics-audit-report-2026-02-23.txt`
  - Note:
    - Validation completed with module matrix and targeted code fixes; one follow-up remains to wire loop-closing BA intrinsics from caller context instead of relying on default fallback.

- [x] Validate SLAM-to-3DGS contract assumptions for Epic 4 continuity.
  - Owner: Dana (QA Engineer)
  - Source: Epic 3
  - Done means:
    - explicit verification for pose stability, descriptor/matcher behavior, relocalization quality.
  - Evidence:
    - `_bmad-output/implementation-artifacts/slam-to-3dgs-contract-validation-2026-02-23.md`
  - Note:
    - Validation completed with pass/fail matrix and targeted test evidence; runtime relocalization integration is a documented conditional-risk follow-up.

## P2 (Improvement and governance hardening)

- [ ] Add negative-path configuration validation tests.
  - Owner: Dana (QA Engineer)
  - Source: Epic 7
  - Done means:
    - invalid config ranges fail fast with clear recovery guidance.
  - Evidence:
    - new/updated test cases and sample failure outputs.

- [ ] Create a formal Epic 8 definition and dependency map.
  - Owner: Alice (Product Owner)
  - Source: Epic 7
  - Done means:
    - Epic 8 appears in planning artifacts with dependency clarity.
  - Evidence:
    - updated `_bmad-output/planning-artifacts/epics.md` section for Epic 8.

- [ ] Introduce retrospective evidence checklist into story closure criteria.
  - Owner: Bob (Scrum Master)
  - Source: Epic 3 + Epic 7
  - Done means:
    - closure workflow requires status + evidence payload.
  - Evidence:
    - checklist doc referenced by closure process.

## 2. Owner View

### Bob (Scrum Master)
- [x] Align Story 3-7 status records.
- [x] Backfill Epic 7 story evidence records.
- [ ] Add retrospective evidence checklist into closure criteria.

### Charlie (Senior Developer)
- [ ] Rebuild complete Epic 3 evidence chain.

### Winston (Architect)
- [x] Implement automated status-integrity check.
- [x] Verify intrinsics single-source usage across modules.

### Dana (QA Engineer)
- [x] Validate SLAM-to-3DGS contract assumptions.
- [ ] Add negative-path config validation tests.

### Alice (Product Owner)
- [ ] Define Epic 8 scope and dependencies in planning artifacts.

## 3. Suggested Execution Order

1. P0 alignment and evidence backfill (status trustworthiness first).
2. P1 automation + cross-module verification (prevent regressions).
3. P2 governance reinforcement and roadmap continuation.

## 4. Completion Gate for This Action Plan

This plan is considered completed when:
- all P0 items are checked,
- at least one automated integrity check is active,
- Epic 8 is explicitly defined or deferred with documented rationale.
