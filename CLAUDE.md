# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RustScan is a pure Rust implementation of 3D scanning algorithms, comprising two main libraries:

- **RustMesh**: A mesh processing library (Rust port of OpenMesh)
- **RustSLAM**: A Visual SLAM library with 3D Gaussian Splatting support

## Common Commands

### Build
```bash
# Build RustMesh
cd RustMesh && cargo build

# Build RustSLAM
cd RustSLAM && cargo build --release
```

### Test
```bash
# Test RustMesh
cd RustMesh && cargo test

# Test RustSLAM
cd RustSLAM && cargo test
```

### Run Examples
```bash
# Run VO example
cd RustSLAM && cargo run --example run_vo

# Run RustMesh demo
cd RustMesh && cargo run --example smart_handles_demo
```

### Benchmark
```bash
cd RustMesh && cargo bench
```

## Architecture

### RustMesh (`RustMesh/`)

Mesh data structure library implementing half-edge data structure with SoA (Structure of Arrays) memory layout.

```
RustMesh/src/
├── Core/               # Core data structures
│   ├── handles.rs      # Handle types (VertexHandle, EdgeHandle, HalfedgeHandle, FaceHandle)
│   ├── connectivity.rs # Half-edge connectivity relations
│   ├── soa_kernel.rs   # SoA storage layer
│   ├── kernel.rs       # ArrayKernel base
│   ├── attrib_kernel.rs # Attribute management
│   ├── geometry.rs     # Geometric operations
│   └── io.rs           # File I/O (OFF, OBJ, PLY, STL)
├── Tools/              # Mesh algorithms
│   ├── decimation.rs   # Quadric-based mesh simplification
│   ├── subdivision.rs  # Loop, Catmull-Clark, Sqrt3 subdivision
│   ├── smoother.rs     # Laplace, Tangential smoothing
│   ├── hole_filling.rs # Hole filling
│   └── mesh_repair.rs # Mesh repair utilities
└── Utils/
    ├── circulators.rs  # Vertex/Edge/Face circulators
    └── quadric.rs      # Quadric error computation
```

### RustSLAM (`RustSLAM/`)

Visual SLAM library with sparse feature-based VO and dense 3D Gaussian Splatting reconstruction.

```
RustSLAM/src/
├── cli/               # CLI & pipeline orchestration
│   ├── mod.rs         # CLI args, run_pipeline(), stage functions
│   ├── pipeline_checkpoint.rs  # Cross-stage checkpoint management
│   └── integration_tests.rs   # End-to-end integration tests
├── io/                # I/O utilities
│   └── video_decoder.rs  # ffmpeg-next decoder, VideoToolbox HW accel, LRU cache
├── core/              # Core data structures
│   ├── frame.rs       # Frame
│   ├── keyframe.rs    # KeyFrame
│   ├── map_point.rs   # MapPoint
│   ├── map.rs         # Map management
│   ├── camera.rs      # Camera model
│   └── pose.rs        # SE3 Pose
├── features/          # Feature extraction
│   ├── orb.rs         # ORB extractor
│   ├── pure_rust.rs   # Harris/FAST corner detection
│   ├── matcher.rs     # Feature matching
│   └── knn_matcher.rs
├── tracker/           # Visual Odometry
│   └── vo.rs          # Main VO pipeline
├── optimizer/         # Bundle Adjustment
│   └── ba.rs
├── loop_closing/      # Loop Detection
│   ├── vocabulary.rs  # BoW Vocabulary
│   ├── detector.rs    # Loop Detector
│   └── relocalization.rs
├── pipeline/          # Pipeline infrastructure
│   ├── checkpoint.rs  # SLAM checkpoint save/load
│   └── realtime.rs    # Multi-threaded realtime pipeline
└── fusion/            # 3D Gaussian Splatting
    ├── gaussian.rs    # Gaussian data structures
    ├── gaussian_init.rs     # Init from SLAM map points
    ├── diff_renderer.rs     # Differentiable renderer (CPU)
    ├── diff_splat.rs        # GPU differentiable splatting
    ├── tiled_renderer.rs    # Tiled rasterization + densify + prune
    ├── training_pipeline.rs # Training with SSIM loss
    ├── complete_trainer.rs  # Complete trainer with LR scheduler
    ├── slam_integrator.rs   # Sparse + Dense SLAM integration
    ├── scene_io.rs          # PLY scene save/load
    ├── training_checkpoint.rs  # Training checkpoint
    ├── tsdf_volume.rs       # TSDF volume
    ├── marching_cubes.rs    # Marching Cubes (256-case)
    ├── mesh_extractor.rs    # High-level mesh extraction API
    ├── mesh_io.rs           # OBJ/PLY mesh export
    └── mesh_metadata.rs     # JSON metadata export
```

## Module Status

### RustSLAM Progress (~80% functional — Code Review 2026-02-23)

| Feature | Status |
|---------|--------|
| SE3 Pose | ✅ |
| ORB/Harris/FAST | ⚠️ Orientation computed (intensity centroid), but descriptors are raw intensity patches, NOT binary BRIEF |
| Feature Matching | ⚠️ HammingMatcher has LSH acceleration; KnnMatcher uses wrong distance metric (Euclidean instead of Hamming) |
| Visual Odometry | ⚠️ DLT-PnP works (mislabeled as P3P), Gauss-Newton refinement functional; 3D points never updated after init; relocalization is stub |
| Bundle Adjustment | ⚠️ Optimizes both poses and landmarks via finite-difference GD (docs say Gauss-Newton, actually GD) |
| Loop Closing | ✅ Sim3 rotation via SVD (Umeyama algorithm); BoW uses TF-IDF; hardcoded intrinsics in global BA |
| Video Input (MP4/MOV/HEVC) | ✅ |
| Hardware Decoding (VideoToolbox) | ✅ |
| LRU Frame Cache | ✅ |
| CLI Infrastructure | ✅ |
| Config File (TOML) | ✅ |
| Structured JSON Output | ✅ |
| Pipeline Checkpoints | ✅ Path traversal protection present |
| Progress Reporting | ✅ |
| 3DGS Data Structures | ✅ |
| Tiled Rasterization | ❌ CPU per-pixel, dist⁴ bug in Gaussian kernel (`tiled_renderer.rs:271`) |
| Depth Sorting | ✅ |
| Alpha Blending | ✅ Front-to-back alpha compositing with Gaussian kernel (correct in diff_splat.rs) |
| Gaussian Densification & Pruning | ✅ Gradient accumulation from local depth/color errors |
| Candle + Metal GPU | ⚠️ GPU used for projection only, rasterization is CPU |
| Backward Propagation | ✅ Real analytical backward pass (`analytical_backward.rs`), finite-diff fallback available |
| SLAM Integration | ✅ Framework works, Sim3 rotation fixed |
| TSDF Volume Fusion | ✅ Division guards present (.max(1e-8)) |
| Marching Cubes Extraction | ✅ No sqrt bug (previously reported bug was in tiled_renderer, not here) |
| Mesh Export (OBJ/PLY) | ✅ Correct formats (OBJ 1-based, PLY 0-based) |
| Mesh Metadata (JSON) | ✅ |
| End-to-End Integration Tests | ✅ 249/249 lib tests pass, all examples compile |
| Map Thread Safety | ❌ No Arc/RwLock, non-atomic ID counters |
| Relocalization | ❌ Stub — returns failure |
| Config Validation | ❌ No parameter range validation |
| IMU Integration | ⏳ |
| Multi-map SLAM | ⏳ |

### RustMesh Progress (~50-60%)

| Feature | Status |
|---------|--------|
| Handle System | ✅ |
| Half-edge + SoA | ✅ |
| OFF/OBJ/PLY/STL IO | ✅ |
| Smart Handles | ✅ |
| Decimation | ⚠️ Basic |
| Subdivision | ⚠️ Loop/CC/√3 |
| Hole Filling | ✅ |
| AttribKernel Integration | ⏳ |
| 3DGS → Mesh | ✅ (via RustSLAM) |

### 3DGS → Mesh Extraction (IMPLEMENTED)

New files in `RustSLAM/src/fusion/`:

1. **tsdf_volume.rs** - Pure Rust TSDF volume implementation
   - `TsdfVolume` - volumetric fusion
   - `TsdfConfig` - configurable voxel size, truncation distance
   - Supports depth map integration from Gaussian rendering

2. **marching_cubes.rs** - Marching Cubes algorithm
   - Full 256-case lookup table
   - Mesh vertex and triangle generation
   - Color interpolation

3. **mesh_extractor.rs** - High-level API
   - `MeshExtractor` - easy-to-use interface
   - `MeshExtractionConfig` - post-processing options
   - Cluster filtering (remove floaters)
   - Normal smoothing

**Usage**:
```rust
use rustslam::fusion::{MeshExtractor, MeshExtractionConfig};
use glam::Vec3;
use glam::Mat4;

// Create extractor
let mut extractor = MeshExtractor::centered(Vec3::ZERO, 2.0, 0.01);

// Integrate depth frames from Gaussian rendering
extractor.integrate_from_gaussians(
    |idx| depth[idx],
    width, height,
    [fx, fy, cx, cy],
    &camera_pose,
);

// Extract mesh with post-processing
let mesh = extractor.extract_with_postprocessing();
```

## Key Dependencies

- **glam**: SIMD-accelerated math library (Vec3, Mat4, Quat)
- **nalgebra**: Linear algebra (for BA solver)
- **rayon**: Data parallelism
- **opencv** (optional): Image processing
- **candle-metal**: GPU acceleration via Apple MPS
- **apex-solver**: Bundle adjustment

## Pipeline

```
iPhone Video (MP4/MOV/HEVC)
    ↓ ffmpeg-next + VideoToolbox HW accel + LRU cache
Frame Extraction
    ↓
RustSLAM (VO + Mapping)     → Sparse Map + Camera Poses
    ↓                              ↓
Loop Closing              Bundle Adjustment
    ↓
3DGS Fusion (Gaussian Splatting)
    ↓ Metal/MPS GPU, depth constraints, SSIM loss
Trained 3DGS Scene (scene.ply)
    ↓
TSDF Volume Fusion + Marching Cubes
    ↓
Mesh Post-processing (cluster filter, normal smooth)
    ↓
Export: mesh.obj + mesh.ply + mesh_metadata.json + results.json
```

**CLI Usage:**
```bash
# Full pipeline
rustscan --input video.mp4 --output ./results

# With config file
rustscan --config rustscan.toml --input video.mp4

# With debug logging
rustscan --input video.mp4 --output ./results --log-level debug

# JSON output format
rustscan --input video.mp4 --output ./results --output-format json

# Run example videos
./run_examples.sh
```
