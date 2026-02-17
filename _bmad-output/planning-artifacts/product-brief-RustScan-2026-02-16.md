---
stepsCompleted: [1, 2, 3, 4, 5, 6]
inputDocuments:
  - _bmad-output/project-context.md
  - docs/index.md
  - docs/ROADMAP.md
  - docs/RustSLAM-ToDo.md
date: 2026-02-16
author: 飞哥
---

# Product Brief: RustScan

<!-- Content will be appended sequentially through collaborative workflow steps -->

## Executive Summary

RustScan is a pure Rust implementation of an end-to-end 3D reconstruction pipeline that combines Visual SLAM with 3D Gaussian Splatting (3DGS) technology to produce both high-fidelity visual scenes and high-quality triangle meshes from iPhone videos. Unlike existing solutions that either lack 3DGS support (ORB-SLAM3) or are not implemented in Rust (Open3D), RustScan provides a complete, efficient, and type-safe pipeline from video input to final mesh output.

The project leverages recent advances in 3DGS training (FastGS) and depth-constrained optimization to achieve fast convergence and superior reconstruction quality. By implementing the entire stack in Rust, RustScan delivers performance, memory safety, and seamless integration across all pipeline stages.

**Current Status**: ~85% complete, with Phase 1 (core connection) finished. The immediate goal is to complete the end-to-end pipeline integration to enable offline processing of iPhone videos into trained 3DGS scenes and clean triangle meshes.

---

## Core Vision

### Problem Statement

Existing 3D reconstruction solutions face critical limitations:
- **ORB-SLAM3**: Excellent SLAM but no 3DGS support, cannot produce high-fidelity visual scenes
- **Open3D**: Good mesh processing but not a complete SLAM pipeline, not implemented in Rust
- **Fragmented Toolchains**: Researchers and developers must stitch together multiple tools (C++, Python) with different memory models and performance characteristics
- **Incomplete Pipelines**: No single solution provides SLAM → 3DGS → High-Quality Mesh in one cohesive system

### Problem Impact

Without an integrated solution:
- Developers waste time integrating incompatible tools
- Performance suffers from cross-language boundaries and data conversions
- Memory safety issues arise from mixing C++ and Python codebases
- 3D reconstruction quality is limited by using only sparse SLAM or only dense 3DGS, not both

### Why Existing Solutions Fall Short

**ORB-SLAM3** (C++):
- No dense reconstruction or 3DGS support
- Only produces sparse point clouds
- Cannot generate high-fidelity visual scenes

**Open3D** (C++/Python):
- Not a complete SLAM system
- Lacks real-time camera tracking
- No 3DGS integration

**3DGS Implementations** (Python/CUDA):
- Require pre-computed camera poses (no SLAM)
- Slow training without depth constraints
- No mesh extraction pipeline

**Gap**: No solution combines SLAM + 3DGS + Mesh Extraction in a single, efficient, type-safe implementation.

### Proposed Solution

RustScan provides a **pure Rust end-to-end 3D reconstruction pipeline**:

**Input**: iPhone video files (MP4/MOV/HEVC)

**Pipeline**:
1. **Visual SLAM**: Camera pose estimation per frame using ORB features, Bundle Adjustment, and Loop Closing
2. **3DGS Training**: Depth-constrained Gaussian Splatting for fast convergence and high-quality novel view synthesis
3. **Mesh Extraction**: TSDF volume fusion + Marching Cubes to produce clean triangle meshes
4. **Post-Processing**: RustMesh algorithms (decimation, smoothing, hole filling) for final mesh refinement

**Output**:
- Trained 3DGS scene (high-fidelity visualization)
- High-quality triangle mesh (minimal noise, ready for export)

**MVP Scope**:
- Offline processing (not real-time)
- iPhone video support
- Fast training with depth constraints
- High-quality mesh extraction

### Key Differentiators

1. **Pure Rust Implementation**
   - Memory safety without garbage collection overhead
   - Zero-cost abstractions for performance
   - Seamless integration across all pipeline stages
   - Growing Rust ecosystem support

2. **End-to-End Pipeline**
   - Complete solution from video → 3DGS → mesh
   - No tool stitching or format conversions
   - Unified data structures and memory model

3. **Depth-Constrained 3DGS Training**
   - Faster convergence than RGB-only methods
   - Better geometry consistency
   - Higher quality novel view synthesis

4. **Dual Output Quality**
   - High-fidelity 3DGS scenes for visualization
   - Clean triangle meshes for downstream applications
   - Best of both worlds: visual quality + geometric accuracy

5. **Modern Tech Stack**
   - Leverages FastGS advances for training speed
   - Apple Metal/MPS GPU acceleration
   - Mature Rust ecosystem (glam, nalgebra, candle)

6. **Timing Advantage**
   - 3DGS technology recently matured (2023-2024)
   - FastGS and similar work enable practical training speeds
   - Rust ecosystem now robust enough for complex CV/graphics

---

## Target Users

### Primary Users: Action Camera Enthusiasts

**Persona: 李明 (Ming Li) - Outdoor Adventure Recorder**

**Background:**
- Age: 28-35
- Occupation: Outdoor enthusiast, travel blogger, or hobbyist photographer
- Equipment: Owns GoPro, DJI Action, or iPhone with good camera
- Technical Level: Comfortable with consumer software, not a 3D expert

**Current Problem:**
- Records beautiful outdoor scenes (hiking trails, caves, scenic spots) but can only view them as flat videos
- Wants to "walk through" recorded scenes again and share immersive experiences
- Existing 3D scanning solutions are either too expensive (professional LiDAR scanners cost $10K-$100K) or too complex (require technical expertise)
- Cannot easily create 3D souvenirs or measure interesting features in recorded scenes

**Usage Flow:**
1. **Capture**: Records video while exploring outdoor locations using action camera or phone
2. **Process**: Returns home and either:
   - Imports video into PC software for local processing
   - Uploads video via phone to cloud for processing
3. **View & Share**: Views results in browser, downloads 3DGS scenes and meshes
4. **Use Cases**:
   - Immersive viewing of past adventures
   - Creating 3D-printed souvenirs
   - Measuring cave dimensions or trail features
   - Sharing interactive 3D scenes on social media

**Success Criteria (Priority Order):**
1. **High-Quality 3DGS Rendering**: Realistic textures, photorealistic appearance
2. **Sharp Novel Views**: No blur when viewing from new angles, smooth navigation
3. **Convenient Capture**: Simple video recording, no special equipment needed
4. **Clean Mesh Output**: Suitable for 3D printing or further editing

**"Aha!" Moment:**
When Ming first sees his recorded hiking trail transformed into an interactive 3D scene that he can "walk through" from any angle, with photorealistic quality that makes him feel like he's back there.

---

### Secondary Users

#### 1. Home Renovation Workers

**Persona: 王师傅 (Master Wang) - Renovation Contractor**

**Background:**
- Age: 35-50
- Occupation: Home renovation contractor
- Need: Document before/after states, precise measurements

**Usage Scenario:**
- Records client's space before renovation
- Uses 3DGS for visual comparison and client communication
- Relies on accurate scale for measurements and material estimation
- Needs clean meshes for renovation planning

**Key Requirements:**
- **Accurate Scale**: Scene dimensions must be precise for measurement work
- **Realistic Rendering**: Clients need to see realistic before/after comparisons
- **Measurement Tools**: Ability to measure distances, areas, volumes

#### 2. Public Safety & Documentation

**Persona: 张警官 (Officer Zhang) - Crime Scene Investigator**

**Background:**
- Age: 30-45
- Occupation: Public safety officer, accident investigator
- Need: Accurate scene documentation with legal validity

**Usage Scenario:**
- Documents accident scenes, crime scenes, or public safety incidents
- Requires high-fidelity reconstruction for investigation and court evidence
- Needs precise measurements and scale accuracy
- Must preserve scene details for future analysis

**Key Requirements:**
- **Accurate Scale**: Critical for forensic measurements
- **High Fidelity**: Must capture all scene details accurately
- **Archival Quality**: Long-term preservation of scene data
- **Measurement Capability**: Precise distance and area measurements

#### 3. Game Developers

**Persona: 陈开发 (Chen, Game Dev) - Indie Game Developer**

**Background:**
- Age: 25-35
- Occupation: Game developer, 3D artist
- Need: Real-world assets for game environments

**Usage Scenario:**
- Scans real-world locations for game level design
- Uses 3DGS for reference and clean meshes for game assets
- Needs editable, optimized meshes for game engines
- Values both visual quality and geometric cleanliness

**Key Requirements:**
- **Clean Meshes**: Low polygon count, good topology for game engines
- **Realistic Textures**: High-quality 3DGS for reference
- **Editability**: Meshes must be suitable for further modeling work

---

### User Journey

#### Primary User Journey: Ming's First Experience

**1. Discovery (Week 0)**
- Ming sees a friend's immersive 3D scene shared on social media
- Searches for "3D scene from video" and discovers RustScan
- Downloads PC software or mobile app

**2. First Capture (Day 1)**
- Takes RustScan on weekend hiking trip
- Records 2-3 minute video of scenic viewpoint using phone
- Simple recording, no special techniques needed

**3. Processing (Day 1 Evening)**
- Returns home, imports video into RustScan PC software
- Or uploads via phone app to cloud processing
- Processing takes 30-60 minutes (acceptable for offline workflow)
- Receives notification when processing completes

**4. "Aha!" Moment (Day 1)**
- Opens browser viewer, sees his hiking trail in full 3D
- Navigates through scene from any angle - sharp, realistic, no blur
- Realizes: "I can revisit this place anytime, from any viewpoint!"
- Shares interactive link with friends who are amazed

**5. Continued Use (Ongoing)**
- Records every significant trip or location
- Builds personal 3D memory library
- Experiments with 3D printing small souvenirs
- Becomes advocate, shares on outdoor forums

**6. Long-Term Value**
- Accumulates library of 3D memories
- Uses for trip planning (revisiting locations virtually)
- Creates unique content for social media
- Measures trail features for hiking guides

#### Secondary User Journey: Master Wang's Workflow

**1. Client Meeting**
- Shows client 3D scan of their current space
- Discusses renovation plans with immersive visualization

**2. Documentation**
- Records video of space before starting work
- Processes overnight, gets accurate 3D model

**3. Planning**
- Uses measurements from 3DGS scene for material estimation
- Shows client realistic before/after comparisons

**4. Quality Assurance**
- Records after renovation, compares with original scan
- Provides client with both 3D models as deliverables

---

## Success Metrics

### MVP Technical Success Criteria

**Primary Success Indicators:**

1. **End-to-End Pipeline Completion**
   - **Metric**: Pipeline successfully processes video from input to final outputs
   - **Target**: 100% completion rate for valid input videos
   - **Validation**: Input video → Camera poses → 3DGS scene → Triangle mesh
   - **Success Criteria**: All pipeline stages execute without errors

2. **Processing Time**
   - **Metric**: Total processing time from video input to final outputs
   - **Target**: ≤ 30 minutes for typical video (2-3 minutes, 1080p, 30fps)
   - **Measurement**: Wall-clock time from pipeline start to completion
   - **Breakdown**:
     - SLAM processing: ~10-15 minutes
     - 3DGS training: ~10-15 minutes
     - Mesh extraction: ~2-5 minutes

3. **3DGS Rendering Quality**
   - **Metric**: Peak Signal-to-Noise Ratio (PSNR)
   - **Target**: PSNR > 28 dB on test views
   - **Measurement**: Compare rendered novel views against ground truth frames
   - **Success Criteria**: Photorealistic rendering with minimal artifacts

**Secondary Technical Indicators:**

4. **SLAM Tracking Success**
   - **Metric**: Percentage of frames successfully tracked
   - **Target**: > 95% frame tracking success rate
   - **Validation**: Camera poses estimated for all frames without tracking loss

5. **3DGS Training Convergence**
   - **Metric**: Training loss reduction and convergence
   - **Target**: Loss decreases consistently, converges within training time
   - **Indicators**:
     - L1 loss + SSIM loss combined
     - Gaussian count stabilizes (densification/pruning balanced)

6. **Mesh Quality**
   - **Metric**: Clean, watertight mesh output
   - **Target**: 
     - Minimal isolated triangles (< 1% of total)
     - No major holes or artifacts
     - Suitable for 3D printing or game engine import
   - **Validation**: Successfully imports into Blender/Unity without errors

7. **System Stability**
   - **Metric**: Pipeline reliability and error handling
   - **Target**: > 90% success rate across diverse test videos
   - **Validation**: Graceful handling of edge cases (motion blur, low texture, etc.)

**Test Validation Plan:**

- **Test Dataset**: 10-20 diverse video samples
  - Indoor scenes (rooms, corridors)
  - Outdoor scenes (trails, viewpoints)
  - Various lighting conditions
  - Different camera motions (smooth, handheld)

- **Success Threshold**: 
  - 80% of test videos meet all primary success criteria
  - 100% of test videos complete without crashes

**Future Metrics (Post-MVP):**

Once MVP is validated, expand to include:
- User success metrics (completion rate, satisfaction, retention)
- Business metrics (user growth, engagement, revenue)
- Performance optimization metrics (GPU utilization, memory usage)

---

## MVP Scope

### Core Features

**1. Video Input Module**
- **Format Support**: iPhone video formats (MP4/MOV/HEVC)
- **Video Decoding**: Frame extraction with timestamps
- **Camera Intrinsics**: Automatic estimation from video metadata or default values
- **Input Validation**: Check video format, resolution, frame rate

**2. Visual SLAM Module**
- **Feature Extraction**: ORB features with Harris/FAST corner detection
- **Feature Matching**: KNN-based matching with ratio test
- **Camera Tracking**: Per-frame pose estimation using PnP solver
- **Bundle Adjustment**: Gauss-Newton optimization for pose refinement
- **Loop Closing**: BoW-based loop detection and pose graph optimization
- **Output**: Camera trajectory (poses for all frames)

**3. 3D Gaussian Splatting Training Module**
- **Depth-Constrained Training**: Use SLAM depth for faster convergence
- **GPU Acceleration**: Apple Metal/MPS backend via candle-metal
- **Training Pipeline**: 
  - Gaussian initialization from SLAM points
  - Differentiable rendering with tiled rasterization
  - Loss optimization (L1 + SSIM)
  - Densification and pruning
- **Quality Target**: PSNR > 28 dB
- **Output**: Trained 3DGS scene file

**4. Mesh Extraction Module**
- **TSDF Volume Fusion**: Integrate depth maps from multiple views
- **Marching Cubes**: Extract triangle mesh from TSDF volume
- **Post-Processing**:
  - Remove isolated triangles (< 1% of total)
  - Cluster filtering to remove noise
  - Optional normal smoothing
- **Output**: Clean triangle mesh (OBJ/PLY format)

**5. Command-Line Interface**
- **Input**: Video file path
- **Configuration**: Basic parameters (voxel size, training iterations, etc.)
- **Progress Reporting**: Console output showing pipeline stages
- **Output**: 3DGS scene file + mesh file in specified directory

**6. Processing Pipeline Integration**
- **Sequential Execution**: SLAM → 3DGS Training → Mesh Extraction
- **Data Flow**: Seamless data passing between modules
- **Error Handling**: Graceful failure with informative error messages
- **Performance**: Complete processing within 30 minutes for typical video

---

### Out of Scope for MVP

**Explicitly NOT included in MVP:**

1. **Graphical User Interface (GUI)**
   - No 3D visualization window
   - No interactive controls
   - Command-line only for MVP
   - *Rationale*: Focus on core pipeline functionality first

2. **Cloud Processing**
   - No cloud upload/processing capability
   - No mobile app integration
   - Local PC processing only
   - *Rationale*: Simplify deployment and focus on algorithm quality

3. **Measurement Tools**
   - No distance/area measurement features
   - No annotation or markup tools
   - No scale calibration interface
   - *Rationale*: Core reconstruction first, tools later

4. **Real-Time Processing**
   - Offline processing only
   - No live camera feed support
   - No streaming optimization
   - *Rationale*: Quality over speed for MVP

5. **Advanced Features**
   - No IMU integration
   - No multi-map SLAM
   - No semantic segmentation
   - No texture editing
   - *Rationale*: Defer to post-MVP phases

6. **Extended Format Support**
   - iPhone videos only (MP4/MOV/HEVC)
   - No generic video format support
   - No image sequence input
   - *Rationale*: Narrow scope for faster validation

---

### MVP Success Criteria

**Technical Validation:**

1. **Pipeline Completion**
   - Successfully processes 10 diverse test videos
   - 80% meet all technical metrics:
     - Processing time ≤ 30 minutes
     - PSNR > 28 dB
     - Clean mesh output (< 1% isolated triangles)
   - 100% complete without crashes

2. **Quality Validation**
   - 3DGS rendering is photorealistic
   - Novel views are sharp (no blur)
   - Mesh is watertight and suitable for export
   - Successfully imports into Blender/Unity

3. **Robustness Validation**
   - Handles various scene types (indoor/outdoor)
   - Handles different lighting conditions
   - Handles different camera motions (smooth/handheld)
   - Graceful error handling for edge cases

**Go/No-Go Decision Criteria:**

- **GO**: If 80% of test videos meet technical metrics → Proceed to user testing and GUI development
- **NO-GO**: If < 80% success rate → Iterate on core algorithms before expanding scope

**Learning Objectives:**

- Validate depth-constrained 3DGS training approach
- Confirm processing time is acceptable for offline workflow
- Identify bottlenecks for future optimization
- Gather insights for GUI design requirements

---

### Future Vision

**Post-MVP Roadmap (6-12 months):**

**Phase 2: User Experience (3-4 months)**
1. **GUI Development**
   - 3D viewport for real-time visualization
   - Interactive camera controls
   - Progress indicators and status display
   - Export options and settings panel

2. **Workflow Improvements**
   - Batch processing for multiple videos
   - Resume capability for interrupted processing
   - Quality presets (fast/balanced/quality)
   - Automatic parameter tuning

**Phase 3: Performance & Scale (3-4 months)**
3. **Real-Time Processing**
   - Optimize for 30fps processing speed
   - Streaming pipeline architecture
   - Incremental 3DGS updates
   - Live preview during capture

4. **Cloud Integration**
   - Mobile app for video upload
   - Cloud processing service
   - Browser-based viewer
   - Collaborative sharing features

**Phase 4: Advanced Features (6+ months)**
5. **Professional Tools**
   - Measurement and annotation tools
   - Scale calibration interface
   - Export to various formats (FBX, GLTF, USD)
   - Integration with CAD/game engines

6. **Algorithm Enhancements**
   - IMU integration for better tracking
   - Multi-map SLAM for large scenes
   - Semantic segmentation for object extraction
   - Texture editing and enhancement

**Long-Term Vision (2-3 years):**

- **Platform Expansion**: Support for all action cameras and smartphones
- **Ecosystem**: Plugin marketplace for custom processing pipelines
- **Enterprise**: Professional features for architecture, construction, public safety
- **Community**: Open-source core with commercial add-ons
