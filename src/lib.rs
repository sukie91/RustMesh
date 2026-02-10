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
pub use kernel::{ArrayKernel, PropertyContainer, StatusInfo};
pub use connectivity::PolyMesh;
pub use geometry::*;
pub use io::*;

// Re-export glam for convenience
pub use glam::Vec3;

mod handles;
mod items;
mod kernel;
mod connectivity;
mod geometry;
mod io;
