# RustScan API Reference

## RustMesh API

### Core Types

#### Handles

```rust
pub struct VertexHandle(pub usize);
pub struct EdgeHandle(pub usize);
pub struct HalfedgeHandle(pub usize);
pub struct FaceHandle(pub usize);
```

Handles are lightweight references to mesh elements. They are Copy types and can be used as indices.

#### ArrayKernel

The main mesh data structure using Structure of Arrays (SoA) layout.

```rust
pub struct ArrayKernel {
    // Connectivity
    vertices: Vec<VertexConnectivity>,
    halfedges: Vec<HalfedgeConnectivity>,
    edges: Vec<EdgeConnectivity>,
    faces: Vec<FaceConnectivity>,

    // Geometry
    points: Vec<Vec3>,
    normals: Vec<Vec3>,
    // ...
}
```

### Core Operations

#### Creating a Mesh

```rust
use rustmesh::ArrayKernel;

let mut mesh = ArrayKernel::new();
```

#### Adding Vertices

```rust
use glam::Vec3;

let v0 = mesh.add_vertex(Vec3::new(0.0, 0.0, 0.0));
let v1 = mesh.add_vertex(Vec3::new(1.0, 0.0, 0.0));
let v2 = mesh.add_vertex(Vec3::new(0.0, 1.0, 0.0));
```

#### Adding Faces

```rust
let face = mesh.add_face(&[v0, v1, v2]);
```

#### Querying Connectivity

```rust
// Get vertex position
let pos = mesh.point(v0);

// Get outgoing halfedge
let he = mesh.halfedge(v0);

// Get face vertices
let vertices = mesh.face_vertices(face);

// Iterate around vertex
for vh in mesh.vv_iter(v0) {
    // Process neighbor vertices
}
```

### Circulators

Circulators provide efficient iteration around mesh elements.

```rust
// Vertex-Vertex circulator
for neighbor in mesh.vv_iter(vertex) {
    // Process neighbor vertices
}

// Vertex-Face circulator
for face in mesh.vf_iter(vertex) {
    // Process adjacent faces
}

// Face-Vertex circulator
for vertex in mesh.fv_iter(face) {
    // Process face vertices
}
```

### Mesh Algorithms

#### Decimation

```rust
use rustmesh::Tools::Decimation;

let mut decimator = Decimation::new(&mut mesh);
decimator.decimate(target_vertex_count);
```

#### Subdivision

```rust
use rustmesh::Tools::Subdivision;

// Loop subdivision
Subdivision::loop_subdivide(&mut mesh);

// Catmull-Clark subdivision
Subdivision::catmull_clark(&mut mesh);

// Sqrt3 subdivision
Subdivision::sqrt3(&mut mesh);
```

#### Smoothing

```rust
use rustmesh::Tools::Smoother;

let mut smoother = Smoother::new(&mut mesh);
smoother.smooth(iterations);
```

#### Hole Filling

```rust
use rustmesh::Tools::HoleFilling;

HoleFilling::fill_hole(&mut mesh, boundary_halfedge);
```

### I/O Operations

```rust
use rustmesh::Core::io;

// Load mesh
let mesh = io::read_off("input.off")?;
let mesh = io::read_obj("input.obj")?;
let mesh = io::read_ply("input.ply")?;
let mesh = io::read_stl("input.stl")?;

// Save mesh
io::write_off(&mesh, "output.off")?;
io::write_obj(&mesh, "output.obj")?;
io::write_ply(&mesh, "output.ply")?;
io::write_stl(&mesh, "output.stl")?;
```

## RustSLAM API

### Core Types

#### Camera

```rust
pub struct Camera {
    pub fx: f32,  // Focal length x
    pub fy: f32,  // Focal length y
    pub cx: f32,  // Principal point x
    pub cy: f32,  // Principal point y
    pub width: u32,
    pub height: u32,
}
```

#### Pose

```rust
pub struct Pose {
    pub rotation: Quat,     // Rotation quaternion
    pub translation: Vec3,  // Translation vector
}
```

#### Frame

```rust
pub struct Frame {
    pub id: usize,
    pub timestamp: f64,
    pub image: Image,
    pub keypoints: Vec<KeyPoint>,
    pub descriptors: Vec<Descriptor>,
    pub pose: Pose,
}
```

#### MapPoint

```rust
pub struct MapPoint {
    pub id: usize,
    pub position: Vec3,
    pub normal: Vec3,
    pub descriptor: Descriptor,
    pub observations: Vec<(KeyFrameId, KeyPointId)>,
}
```

### Feature Extraction

#### ORB Features

```rust
use rustslam::features::ORBExtractor;

let extractor = ORBExtractor::new(
    num_features,
    scale_factor,
    num_levels,
);

let (keypoints, descriptors) = extractor.detect_and_compute(&image);
```

#### Pure Rust Features

```rust
use rustslam::features::PureRustExtractor;

let extractor = PureRustExtractor::new();
let keypoints = extractor.detect_harris(&image, threshold);
let keypoints = extractor.detect_fast(&image, threshold);
```

### Feature Matching

```rust
use rustslam::features::KnnMatcher;

let matcher = KnnMatcher::new();
let matches = matcher.knn_match(&desc1, &desc2, k);

// Apply ratio test
let good_matches = matcher.ratio_test(&matches, ratio);
```

### Visual Odometry

```rust
use rustslam::tracker::VisualOdometry;

let mut vo = VisualOdometry::new(camera, config);

// Process frame
let pose = vo.track_frame(&frame);

// Get map points
let map_points = vo.get_map_points();
```

### Bundle Adjustment

```rust
use rustslam::optimizer::BundleAdjustment;

let mut ba = BundleAdjustment::new();
ba.add_keyframe(keyframe);
ba.add_map_point(map_point);
ba.optimize(max_iterations);
```

### Loop Closing

#### Vocabulary

```rust
use rustslam::loop_closing::Vocabulary;

let vocab = Vocabulary::load("vocab.bin")?;
let bow = vocab.transform(&descriptors);
```

#### Loop Detector

```rust
use rustslam::loop_closing::LoopDetector;

let mut detector = LoopDetector::new(vocab);
detector.add_keyframe(keyframe);

if let Some(loop_kf) = detector.detect_loop(current_kf) {
    // Handle loop closure
}
```

### 3D Gaussian Splatting

#### Gaussian

```rust
use rustslam::fusion::Gaussian;

let gaussian = Gaussian {
    position: Vec3::new(x, y, z),
    rotation: Quat::IDENTITY,
    scale: Vec3::ONE,
    opacity: 1.0,
    sh_coeffs: vec![...],
};
```

#### Renderer

```rust
use rustslam::fusion::TiledRenderer;

let renderer = TiledRenderer::new(width, height);
let image = renderer.render(&gaussians, &camera_pose);
```

#### Training

```rust
use rustslam::fusion::CompleteTrainer;

let mut trainer = CompleteTrainer::new(config);
trainer.add_training_view(image, pose);
trainer.train(num_iterations);

let gaussians = trainer.get_gaussians();
```

### Mesh Extraction

#### TSDF Volume

```rust
use rustslam::fusion::TsdfVolume;

let mut tsdf = TsdfVolume::new(
    origin,
    size,
    voxel_size,
    truncation_distance,
);

// Integrate depth frame
tsdf.integrate_depth(
    &depth_image,
    &camera_intrinsics,
    &camera_pose,
);
```

#### Marching Cubes

```rust
use rustslam::fusion::MarchingCubes;

let mc = MarchingCubes::new();
let mesh = mc.extract_mesh(&tsdf);
```

#### High-Level API

```rust
use rustslam::fusion::{MeshExtractor, MeshExtractionConfig};

let mut extractor = MeshExtractor::centered(
    center,
    size,
    voxel_size,
);

// Integrate from Gaussians
extractor.integrate_from_gaussians(
    |idx| depth[idx],
    width,
    height,
    intrinsics,
    &pose,
);

// Extract with post-processing
let config = MeshExtractionConfig {
    min_cluster_size: 100,
    smooth_normals: true,
};
let mesh = extractor.extract_with_config(&config);
```

### Pipeline

#### Real-time SLAM

```rust
use rustslam::pipeline::RealtimePipeline;

let mut pipeline = RealtimePipeline::new(camera, config);

// Start pipeline
pipeline.start();

// Feed frames
pipeline.add_frame(frame);

// Get current pose
let pose = pipeline.get_current_pose();

// Get map
let map = pipeline.get_map();

// Stop pipeline
pipeline.stop();
```

### I/O

#### Dataset Loader

```rust
use rustslam::io::DatasetLoader;

let loader = DatasetLoader::new("path/to/dataset");
for (image, pose) in loader.iter() {
    // Process frame
}
```

#### Video Loader

```rust
use rustslam::io::VideoLoader;

let mut loader = VideoLoader::new("video.mp4")?;
while let Some(frame) = loader.next_frame()? {
    // Process frame
}
```

## Common Patterns

### RustMesh: Mesh Processing Pipeline

```rust
use rustmesh::*;

// Load mesh
let mut mesh = io::read_obj("input.obj")?;

// Smooth
let mut smoother = Smoother::new(&mut mesh);
smoother.smooth(5);

// Decimate
let mut decimator = Decimation::new(&mut mesh);
decimator.decimate(1000);

// Save
io::write_obj(&mesh, "output.obj")?;
```

### RustSLAM: Complete SLAM Pipeline

```rust
use rustslam::*;

// Initialize
let camera = Camera::new(fx, fy, cx, cy, width, height);
let mut vo = VisualOdometry::new(camera, config);
let mut loop_detector = LoopDetector::new(vocab);

// Process frames
for frame in dataset {
    // Track
    let pose = vo.track_frame(&frame);

    // Check for loop
    if let Some(loop_kf) = loop_detector.detect_loop(&frame) {
        // Close loop
        vo.close_loop(loop_kf);
    }
}

// Get results
let trajectory = vo.get_trajectory();
let map_points = vo.get_map_points();
```

### RustSLAM: 3DGS to Mesh

```rust
use rustslam::fusion::*;

// Train Gaussians
let mut trainer = CompleteTrainer::new(config);
for (image, pose) in training_views {
    trainer.add_training_view(image, pose);
}
trainer.train(30000);
let gaussians = trainer.get_gaussians();

// Extract mesh
let mut extractor = MeshExtractor::centered(Vec3::ZERO, 2.0, 0.01);
for (depth, pose) in depth_views {
    extractor.integrate_from_gaussians(
        |idx| depth[idx],
        width, height,
        intrinsics,
        &pose,
    );
}
let mesh = extractor.extract_with_postprocessing();

// Export to RustMesh format
// ... convert to ArrayKernel ...
```

## Error Handling

Both libraries use `Result<T, E>` for error handling:

```rust
use std::error::Error;

fn process_mesh() -> Result<(), Box<dyn Error>> {
    let mesh = io::read_obj("input.obj")?;
    // ... process ...
    io::write_obj(&mesh, "output.obj")?;
    Ok(())
}
```

## Performance Tips

1. **Use release builds**: `cargo build --release`
2. **Enable LTO**: Already configured in Cargo.toml
3. **Use parallel features**: `cargo build --features parallel`
4. **Batch operations**: Process multiple elements at once
5. **Avoid unnecessary allocations**: Reuse buffers when possible
6. **Profile before optimizing**: Use `cargo flamegraph`

## Thread Safety

- RustMesh: Most operations are not thread-safe by default. Use `Arc<Mutex<ArrayKernel>>` for shared access.
- RustSLAM: Pipeline components use channels for thread communication. Individual data structures may require synchronization.
