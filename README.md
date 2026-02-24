# RustScan

<p align="center">
  <img src="https://img.shields.io/badge/Rust-1.75+-dea584?style=for-the-badge&logo=rust" alt="Rust">
  <img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" alt="License">
</p>

A complete 3D scanning and reconstruction technology stack implemented in pure Rust.

## Project Goals

Build a pure Rust implementation of 3D scanning and reconstruction technology, covering the complete pipeline from data acquisition to mesh processing.

```
Pipeline: Camera Input → RustSLAM → 3DGS Fusion → Mesh Extraction → RustMesh Post-processing → Export
```

---

## Core Modules

### 🟩 RustMesh (Mesh Processing)

**Core mesh representation and geometric processing library**

- Mesh data structures (Half-edge, SoA layout)
- IO format support (OBJ, OFF, PLY, STL, OM)
- Mesh algorithms
  - Subdivision (Loop, Catmull-Clark, Sqrt3)
  - Simplification (Decimation + Quadric error)
  - Smoothing (Laplace, Tangential)
  - Hole filling
  - Mesh repair
  - Dualization
  - Progressive mesh (VDPM)
- Smart Handle navigation system
- Attribute system

**Progress: ~85%** | [Details](./RustMesh/README.md)

---

### 🟩 RustSLAM (Visual SLAM)

**Pure Rust implementation of Visual SLAM library**

- Feature extraction (ORB, AKAZE, SuperPoint)
- Visual Odometry (VO + PnP)
- Local mapping (Triangulation + BA)
- Loop closing (BoW)
- **3D Gaussian Splatting** - Real-time/offline dense reconstruction
- SLAM + 3DGS fusion

**Tech Stack**:
- opencv-rust: Image processing
- glam: SIMD math library
- candle: PyTorch bindings + Metal GPU
- apex-solver: Graph optimization
- g2o-rs: Graph optimization

**Progress: ~85%** | [Details](./RustSLAM/README.md)

---

### ⬜ RustGUI (GUI + 3D Rendering)

**To be developed - Planned using egui + wgpu**

- Real-time 3D visualization
- GUI interface
- Camera control

**Progress: 0%**

---

## Complete Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    3D Scanning Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Acquisition] → [SLAM] → [3DGS] → [Mesh Extract] → [Post] → [Export] │
│                      ↓                                          │
│                 Real-time Rendering                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module Design

```
RustScan/
├── RustMesh/           # Core mesh library (~85%)
│   ├── Core/           # Basic data structures
│   │   ├── handles.rs      # Handle system
│   │   ├── connectivity.rs  # Connectivity relations
│   │   ├── soa_kernel.rs  # SoA storage
│   │   ├── smart_handles.rs # Smart Handle
│   │   └── om_format.rs    # OM format
│   ├── Tools/          # Mesh algorithms
│   │   ├── decimation.rs   # Simplification
│   │   ├── subdivision.rs  # Subdivision
│   │   ├── smoother.rs    # Smoothing
│   │   ├── hole_filling.rs # Hole filling
│   │   └── ...
│   └── Utils/          # Utilities
│
├── RustSLAM/           # SLAM + 3DGS (~85%)
│   ├── core/           # Core structures
│   │   ├── frame.rs       # Frame
│   │   ├── keyframe.rs    # KeyFrame
│   │   ├── map_point.rs   # MapPoint
│   │   └── camera.rs      # Camera model
│   ├── features/        # Feature extraction
│   │   ├── orb.rs         # ORB
│   │   └── pure_rust.rs   # Harris/FAST
│   ├── tracker/         # Visual Odometry
│   ├── optimizer/       # BA optimization
│   ├── loop_closing/    # Loop closing
│   └── fusion/          # 3DGS fusion
│       ├── gaussian.rs    # Gaussian data structures
│       ├── renderer.rs    # Renderer
│       └── trainer.rs      # Training
│
└── RustGUI/            # GUI (to be developed)
```

---

## Tech Stack

- **Language**: Rust 2021
- **Math Library**: glam (SIMD accelerated)
- **GPU**: wgpu, candle-metal
- **Multithreading**: rayon
- **Comparable to**: OpenMesh, Open3D, ORB-SLAM3

---

## Quick Start

### RustMesh

```bash
cd RustMesh
cargo build
cargo test
cargo run --example smart_handles_demo
```

### RustSLAM

```bash
cd RustSLAM
cargo build --release
cargo run --example run_vo
cargo test
```

---

## Examples

Run the end-to-end sample pipeline on three short iPhone clips with expected outputs:

```bash
./run_examples.sh
```

Outputs are written to `output/examples` and compared against `test_data/expected` by default.

Environment overrides:
- `RUSTSCAN_PROFILE` default `release`
- `RUSTSCAN_MAX_FRAMES` default `12`
- `RUSTSCAN_FRAME_STRIDE` default `2`
- `RUSTSCAN_MESH_VOXEL_SIZE` default `0.05`
- `RUSTSCAN_PREFER_HW` default `false`
- `RUSTSCAN_COMPARE` default `1` (set to `0` to skip mesh count comparison)

---

## Progress Overview

| Module | Completion | Priority | Notes |
|------|--------|--------|------|
| **RustSLAM** | ~85% | P0 | Core SLAM + 3DGS complete |
| **RustMesh** | ~85% | P1 | Solid foundation, all tests passing |
| **RustGUI** | 0% | P2 | To be started |

### RustSLAM Checklist

- [x] SE3 Pose
- [x] ORB Feature Extraction
- [x] Feature Matching
- [x] Visual Odometry
- [x] Bundle Adjustment
- [x] Loop Closing
- [x] Relocalization
- [x] 3D Gaussian data structures
- [x] Gaussian Renderer
- [x] Tiled Rasterization
- [x] Depth Sorting
- [x] Alpha Blending
- [x] Gaussian Tracking
- [x] Densification
- [x] Pruning
- [x] Differentiable Renderer
- [x] Training Pipeline
- [x] SLAM Integration

### RustMesh Checklist

- [x] Handle system
- [x] Half-edge data structure
- [x] SoA memory layout
- [x] OFF/OBJ/PLY/STL IO
- [x] MTL material support
- [x] OM format (basic)
- [x] Smart Handle system
- [x] EdgeFace circulators
- [x] Quadric Decimation
- [x] Loop/Catmull-Clark/√3 subdivision
- [x] Laplace/Tangential smoothing
- [x] Hole Filling
- [x] Mesh Repair
- [x] VDPM basics

---

## Priorities

| Priority | Module | Notes |
|--------|------|------|
| P0 | SLAM | Core, simultaneous localization and mapping |
| P1 | Mesh post-processing | 3DGS → Mesh extraction |
| P2 | Surface reconstruction | Poisson, Ball-Pivoting |
| P3 | RustGUI | Visualization interface |
| P4 | Texture mapping | UV unwrapping + texturing |

---

## References

- [OpenMesh](https://www.openmesh.org/) - C++ mesh processing library
- [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) - Visual SLAM
- [Open3D](http://www.open3d.org/) - 3D reconstruction library
- [SplaTAM](https://github.com/spla-tam/SplaTAM) - 3DGS SLAM (CVPR 2024)
- [RTG-SLAM](https://github.com/MisEty/RTG-SLAM) - Real-time 3DGS
- [PensieveRust](https://github.com/sukie91/PensieveRust) - 3D Gaussian Splatting

---

## License

MIT License - see LICENSE file for details.

---

<p align="center">
Built with ❤️ in Rust
</p>
