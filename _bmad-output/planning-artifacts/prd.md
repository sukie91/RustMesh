---
stepsCompleted:
  - step-01-init
  - step-02-discovery
  - step-03-success
  - step-04-journeys
  - step-05-domain
  - step-06-innovation
  - step-07-project-type
  - step-08-scoping
  - step-09-functional
  - step-10-nonfunctional
  - step-11-polish
inputDocuments:
  - _bmad-output/planning-artifacts/product-brief-RustScan-2026-02-16.md
  - _bmad-output/project-context.md
workflowType: 'prd'
classification:
  projectType: cli_tool (current) → desktop_app (future)
  domain: scientific/computer_vision
  complexity: high
  projectContext: brownfield
  platform: mac
  keyFeatures:
    - GPU acceleration (Metal/MPS)
    - 3D reconstruction (SLAM + 3DGS)
    - End-to-end pipeline integration
---

# Product Requirements Document - RustScan

**Author:** 飞哥
**Date:** 2026-02-17

---

## Executive Summary

RustScan is a pure Rust implementation of an end-to-end 3D reconstruction pipeline combining Visual SLAM with 3D Gaussian Splatting (3DGS) technology. The MVP accepts iPhone video input and outputs trained 3DGS scene files and high-quality triangle meshes.

**Project Type:** CLI Tool → Desktop Application (future)
**Platform:** Mac with Metal/MPS GPU acceleration
**Current Status:** Brownfield project -打通端到端流程

---

## Success Criteria

### User Success

- Users run a single command to obtain complete 3DGS scene file and high-quality mesh
- No manual intervention required between modules

### Technical Success

- End-to-end pipeline runs: Video → SLAM → 3DGS Training → Mesh Extraction
- No crashes or fatal errors during processing
- Output 3DGS scenes render normally
- Output meshes are valid and exportable

### Quality Standards

- 3DGS Rendering Quality: PSNR > 28 dB
- Mesh Quality: < 1% isolated triangles (Blender/Unity compatible)
- Processing Time: ≤ 30 minutes (2-3 minute video)
- SLAM Tracking Success Rate: > 95%

---

## Product Scope

### MVP - Minimum Viable Product

**Input:** iPhone recorded video (MP4/MOV/HEVC)
**Output:** Trained 3DGS scene file + High-quality reconstructed mesh

**Required Modules:**

1. Video Decoding Module (iPhone video formats)
2. SLAM Module (Feature extraction, matching, BA, loop closing)
3. 3DGS Training Module (Depth-constrained training, GPU acceleration)
4. Mesh Extraction Module (TSDF + Marching Cubes)
5. CLI Interface

### Growth Features (Post-MVP)

- GUI Desktop Application
- Cloud Processing
- Measurement Tools

### Vision (Future)

- Mobile App
- Multi-map SLAM for large scenes
- IMU Integration

---

## User Journeys

### Primary Journey - Developer Complete Workflow

- Install RustScan CLI tool
- Prepare iPhone video (MP4/MOV/HEVC)
- Run: `rustscan --input video.mp4 --output ./output`
- Wait for processing
- Receive: 3DGS scene file + mesh file

### Library Integration Journey

- Add RustScan as Cargo dependency
- Initialize pipeline with configuration
- Call processing programmatically
- Receive processed results

### Error Handling Journey

- Invalid video format → Error with supported formats
- GPU unavailable → Clear error message
- Processing failure → Error with recovery suggestions

---

## Domain-Specific Requirements

### GPU Acceleration

- Metal/MPS backend for Apple Silicon
- Maximize GPU utilization
- Optimized memory management

### Texture Quality

- High-fidelity color reconstruction
- SSIM quality metric
- PSNR > 28 dB

---

## Project Scoping & Phased Development

### MVP Strategy

- **Approach:** Problem-Solving MVP - Complete end-to-end pipeline
- **Resources:** Individual project

### Post-MVP Features

**Phase 2:** GUI Desktop Application, Cloud Processing
**Phase 3:** Mobile App, Multi-map SLAM

### Risk Mitigation

- Technical Risks: None (existing foundation)
- Resource Risks: Individual developer

---

## Functional Requirements

### Video Input Processing

- FR1: Users can input iPhone video files (MP4/MOV/HEVC)
- FR2: System validates video format and reports errors

### SLAM Processing

- FR3: System extracts features (ORB/Harris/FAST)
- FR4: System performs feature matching between frames
- FR5: System estimates camera poses
- FR6: System executes bundle adjustment
- FR7: System detects and closes loops

### 3DGS Training

- FR8: System performs 3DGS training with depth constraints
- FR9: System utilizes GPU acceleration (Metal/MPS)
- FR10: System outputs trained 3DGS scene files

### Mesh Generation

- FR11: System fuses depth maps into TSDF volume
- FR12: System extracts mesh via Marching Cubes
- FR13: System outputs exportable mesh files (OBJ/PLY)

### CLI Interface

- FR14: Users execute complete pipeline via command line
- FR15: System runs in non-interactive mode
- FR16: System outputs structured data (JSON)
- FR17: System reads configuration files (YAML/TOML)
- FR18: Command-line arguments override config settings

### Logging & Diagnostics

- FR19: System outputs configurable log levels
- FR20: System provides clear error messages with recovery suggestions
- FR21: System provides diagnostic information on failure

---

## Non-Functional Requirements

### Performance

- Processing Time: ≤ 30 minutes (2-3 minute video)
- 3DGS Rendering: PSNR > 28 dB
- SLAM Tracking: > 95% success rate
- Mesh Quality: < 1% isolated triangles

### Integration

- Output Formats: OBJ, PLY mesh files
- Compatibility: Blender and Unity importable

---

## CLI Tool Requirements

### Interaction Model

- Scriptable (non-interactive, automation-friendly)
- No prompts during execution

### Output Format

- Structured output: JSON format
- Console logging: Configurable levels (debug/info/warn/error)

### Configuration

- Config file: YAML or TOML
- CLI arguments override config settings

### Output Management

- Output directory: `RustSCAN/test_data`
- Output files: 3DGS scene + mesh

### Logging Library

- Use mature library: `log` + `env_logger` or `tracing`

### Not Required

- Shell integration
