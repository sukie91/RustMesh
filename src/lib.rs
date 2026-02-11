//! # RustMesh
//! 
//! A Rust port of OpenMesh - a versatile geometric data structure for
//! representing and manipulating polygonal meshes.
//! 
//! ## Overview
//!
//! RustMesh provides a native Rust implementation of mesh data structures
//! and operations, inspired by OpenMesh. It supports:
//! - Vertex, edge, halfedge, and face representations
//! - Polygonal and triangular mesh types
//! - Efficient iteration and circulation
//! - Property management for custom attributes
//!
//! ## Example
//!
//! ```rust
//! use rustmesh::{PolyMesh, Vec3};
//!
//! let mut mesh = PolyMesh::new();
//!
//! // Add vertices
//! let v0 = mesh.add_vertex(Vec3::new(0.0, 0.0, 0.0));
//! let v1 = mesh.add_vertex(Vec3::new(1.0, 0.0, 0.0));
//! let v2 = mesh.add_vertex(Vec3::new(0.0, 1.0, 0.0));
//!
//! // Add a face
//! mesh.add_face(&[v0, v1, v2]);
//!
//! // Iterate over vertices
//! for v in mesh.vertices() {
//!     let point = mesh.point(v);
//!     println!("Vertex at {:?}", point);
//! }
//! ```

// Re-export modules
pub use handles::{VertexHandle, HalfedgeHandle, EdgeHandle, FaceHandle, BaseHandle};
pub use items::{Vertex, Halfedge, Edge, Face};
// SoA kernel for SIMD optimization
pub use soa_kernel::SoAKernel;
pub use attrib_kernel::{AttribKernel, VertexAttributes, HalfedgeAttributes, EdgeAttributes, FaceAttributes};
pub use connectivity::PolyMesh;
pub use connectivity::PolyMeshSoA;
pub use tri_connectivity::TriMesh;
pub use mesh_checker::{MeshChecker, MeshCheckResult, CheckTargets, check_mesh};
pub use smart_handles::{SmartVertex, SmartHalfedge, SmartFace};
pub use decimater::{Decimater, DecimateConfig, DecimateResult, DecimateStats, EdgeLengthCost, decimater};
pub use om_format::{read_om, write_om, detect_om, OmHeader};
pub use hole_filler::{HoleFiller, HoleInfo, FillResult, HoleFillerStats};
pub use dualizer::{Dualizer, DualResult, DualOptions, DualStats, dualizer};
pub use smoother::{Smoother, SmootherConfig, SmootherConfigTrait, SmoothResult, SmoothStats, SmootherType, smoother};
pub use subdivider::{Subdivider, SubdivideConfig, SubdivideResult, SubdivideStats, SubdivideType, subdivider};
pub use test_data::{generate_cube, generate_tetrahedron, generate_pyramid, generate_icosahedron, generate_sphere, generate_torus, generate_grid};
pub use geometry::*;
pub use io::*;
pub use high_perf_mesh::{HighPerfMesh, hperf};
pub use simd_mesh::{SimdMesh, generate_sphere_simd};
pub use simd_ops::{vertex_sum_simd, bounding_box_simd, centroid_simd, surface_area_simd, benchmark_vertex_sum};

// Re-export glam for convenience
pub use glam::Vec3;

mod handles;
mod items;
mod kernel;
mod soa_kernel;
mod attrib_kernel;
mod connectivity;
mod tri_connectivity;
mod mesh_checker;
mod smart_handles;
mod decimater;
mod om_format;
mod hole_filler;
mod dualizer;
mod smoother;
mod subdivider;
mod test_data;
mod geometry;
mod io;
mod high_perf_mesh;
mod simd_mesh;
mod simd_ops;
