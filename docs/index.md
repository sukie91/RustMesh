# RustScan Documentation Index

Welcome to the RustScan documentation. This index provides a comprehensive guide to all available documentation for the RustScan project.

## Project Overview

RustScan is a pure Rust implementation of 3D scanning algorithms, comprising two main libraries:

- **RustMesh**: A mesh processing library (Rust port of OpenMesh)
- **RustSLAM**: A Visual SLAM library with 3D Gaussian Splatting support

**Version**: 0.1.0
**Language**: Rust (Edition 2021)
**License**: MIT

## Quick Start

- [README](README.md) - Project overview and quick start guide
- [DEVELOPMENT](DEVELOPMENT.md) - Development setup and workflow
- [CLAUDE](CLAUDE.md) - Claude Code integration guide

## Core Documentation

### Architecture & Design

- [ARCHITECTURE](ARCHITECTURE.md) - System architecture and component overview
- [RustSLAM Design](RustSLAM-DESIGN.md) - Detailed RustSLAM design document

### API Reference

- [API Reference](API.md) - Complete API documentation for both libraries
  - RustMesh API
  - RustSLAM API
  - Common patterns and examples

### Development

- [Development Guide](DEVELOPMENT.md) - Building, testing, and contributing
  - Prerequisites and installation
  - Build instructions
  - Testing strategies
  - Running examples
  - Debugging and profiling
  - Code style guidelines

## Component Documentation

### RustMesh

- [RustMesh README](RustMesh-README.md) - RustMesh-specific documentation
- **Key Features**:
  - Half-edge data structure with SoA layout
  - Mesh I/O (OFF, OBJ, PLY, STL)
  - Decimation, subdivision, smoothing
  - Hole filling and mesh repair

### RustSLAM

- [RustSLAM README](RustSLAM-README.md) - RustSLAM-specific documentation
- [RustSLAM Design](RustSLAM-DESIGN.md) - Architecture and design decisions
- [RustSLAM ToDo](RustSLAM-ToDo.md) - Development roadmap and tasks
- **Key Features**:
  - Visual Odometry (VO)
  - Bundle Adjustment (BA)
  - Loop Closing
  - 3D Gaussian Splatting
  - Mesh extraction (TSDF + Marching Cubes)

## Project Planning

- [ROADMAP](ROADMAP.md) - Project roadmap and future plans
- [RustSLAM ToDo](RustSLAM-ToDo.md) - Detailed task list and progress tracking

## Getting Started

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd RustScan

# Build RustMesh
cd RustMesh && cargo build --release

# Build RustSLAM
cd ../RustSLAM && cargo build --release
```

### Running Examples

```bash
# RustMesh examples
cd RustMesh
cargo run --example smart_handles_demo

# RustSLAM examples
cd RustSLAM
cargo run --release --example run_vo
```

### Running Tests

```bash
# Test RustMesh
cd RustMesh && cargo test

# Test RustSLAM
cd RustSLAM && cargo test
```

## Documentation by Topic

### For New Users

1. Start with [README](README.md) for project overview
2. Read [ARCHITECTURE](ARCHITECTURE.md) to understand the system design
3. Follow [DEVELOPMENT](DEVELOPMENT.md) to set up your environment
4. Explore [API Reference](API.md) for code examples

### For Contributors

1. Read [DEVELOPMENT](DEVELOPMENT.md) for workflow and guidelines
2. Check [ROADMAP](ROADMAP.md) for planned features
3. Review [RustSLAM ToDo](RustSLAM-ToDo.md) for specific tasks
4. Follow code style guidelines in [DEVELOPMENT](DEVELOPMENT.md)

### For Researchers

1. Read [ARCHITECTURE](ARCHITECTURE.md) for system overview
2. Study [RustSLAM Design](RustSLAM-DESIGN.md) for algorithm details
3. Review [API Reference](API.md) for implementation details
4. Check [ROADMAP](ROADMAP.md) for research directions

## Key Concepts

### RustMesh Concepts

- **Half-Edge Data Structure**: Efficient mesh representation
- **SoA Layout**: Structure of Arrays for cache efficiency
- **Smart Handles**: Type-safe mesh element references
- **Circulators**: Efficient mesh traversal

### RustSLAM Concepts

- **Visual Odometry**: Camera pose estimation from images
- **Bundle Adjustment**: Global optimization of poses and 3D points
- **Loop Closing**: Detecting and correcting drift
- **3D Gaussian Splatting**: Dense 3D reconstruction
- **TSDF Fusion**: Volumetric integration
- **Marching Cubes**: Mesh extraction from volumes

## Technology Stack

### Core Dependencies

- **glam**: SIMD-accelerated math library
- **nalgebra**: Linear algebra
- **rayon**: Data parallelism
- **serde**: Serialization

### RustMesh Dependencies

- **criterion**: Benchmarking
- **byteorder**: Binary I/O

### RustSLAM Dependencies

- **apex-solver**: Bundle adjustment
- **candle-core/candle-metal**: GPU acceleration
- **kiddo**: KD-Tree for KNN matching
- **opencv** (optional): Image processing
- **tch** (optional): Deep learning

## Build Profiles

### Release (Optimized)

```bash
cargo build --release
```

- LTO enabled
- Single codegen unit
- Maximum optimization (opt-level 3)
- Stripped symbols

### Development (Fast Compilation)

```bash
cargo build
```

- Basic optimization (opt-level 1)
- Minimal debug info

## Testing

### Unit Tests

```bash
cargo test --lib
```

### Integration Tests

```bash
cargo test --example test_name
```

### Benchmarks

```bash
cd RustMesh
cargo bench
```

## Project Structure

```
RustScan/
├── RustMesh/              # Mesh processing library
│   ├── src/
│   │   ├── Core/          # Core data structures
│   │   ├── Tools/         # Mesh algorithms
│   │   └── Utils/         # Utilities
│   ├── examples/          # Example programs (27)
│   └── benches/           # Benchmarks
├── RustSLAM/              # Visual SLAM library
│   ├── src/
│   │   ├── core/          # Core data structures
│   │   ├── features/      # Feature extraction
│   │   ├── tracker/       # Visual Odometry
│   │   ├── optimizer/     # Bundle Adjustment
│   │   ├── loop_closing/  # Loop detection
│   │   ├── fusion/        # 3D Gaussian Splatting
│   │   ├── mapping/       # Mapping
│   │   ├── pipeline/      # SLAM pipeline
│   │   └── io/            # I/O utilities
│   └── examples/          # Example programs (5)
├── docs/                  # Documentation
├── test_data/             # Test datasets
└── README.md              # Project overview
```

## Statistics

- **Total Source Files**: 90 Rust files
  - RustMesh: 25 files
  - RustSLAM: 65 files
- **Examples**: 32 total
  - RustMesh: 27 examples
  - RustSLAM: 5 examples
- **Test Coverage**: Comprehensive unit and integration tests
- **Documentation**: 8 markdown files

## Recent Updates

- ✅ Complete 3DGS → Mesh extraction pipeline (TSDF + Marching Cubes)
- ✅ Comprehensive test coverage for P0 modules
- ✅ GPU acceleration via Apple Metal
- ✅ Real-time SLAM pipeline
- ✅ Loop closing and relocalization

## Future Directions

- ⏳ IMU integration
- ⏳ Multi-map SLAM
- ⏳ Enhanced RustMesh-RustSLAM integration
- ⏳ Additional mesh processing algorithms

## Support & Contributing

- **Issues**: Report bugs and request features via GitHub Issues
- **Contributing**: See [DEVELOPMENT](DEVELOPMENT.md) for contribution guidelines
- **Code Style**: Follow Rust standard style (enforced by `cargo fmt`)
- **Testing**: All contributions must include tests

## External Resources

- [Rust Book](https://doc.rust-lang.org/book/)
- [OpenMesh Documentation](https://www.graphics.rwth-aachen.de/software/openmesh/)
- [ORB-SLAM Paper](https://arxiv.org/abs/1502.00956)
- [3D Gaussian Splatting Paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)

## License

MIT License - See LICENSE file for details

---

**Last Updated**: 2026-02-16
**Documentation Version**: 1.0
**Project Status**: Active Development (~85% complete)
