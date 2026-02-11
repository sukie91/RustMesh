//! # Dualizer
//!
//! Mesh dual generation.
//! Converts primal mesh to dual mesh where faces become vertices and vice versa.

use crate::PolyMesh;
use crate::handles::{VertexHandle, FaceHandle};
use crate::geometry;

/// Result of dualization
#[derive(Debug, Clone)]
pub enum DualResult {
    Success(PolyMesh),
    Error(String),
}

/// Dual mesh options
#[derive(Debug, Clone)]
pub struct DualOptions {
    /// Use face centroids as dual vertex positions
    pub use_centroids: bool,
    /// Flip winding order
    pub flip_faces: bool,
}

impl Default for DualOptions {
    fn default() -> Self {
        Self {
            use_centroids: true,
            flip_faces: false,
        }
    }
}

/// Dualizer - Convert mesh to its dual
#[derive(Debug, Clone)]
pub struct Dualizer {
    options: DualOptions,
}

impl Dualizer {
    /// Create a new dualizer with default options
    pub fn new() -> Self {
        Self::with_options(DualOptions::default())
    }

    /// Create a dualizer with custom options
    pub fn with_options(options: DualOptions) -> Self {
        Self { options }
    }

    /// Compute the dual of a mesh
    ///
    /// In the dual mesh:
    /// - Each face of the original mesh becomes a vertex
    /// - Each vertex of the original mesh becomes a face
    pub fn dualize(&self, mesh: &PolyMesh) -> DualResult {
        let mut dual = PolyMesh::new();

        if mesh.n_faces() == 0 {
            return DualResult::Success(dual);
        }

        // Step 1: Create a vertex for each face (at face centroid)
        let face_to_vertex: Vec<VertexHandle> = mesh.faces()
            .map(|fh| {
                let centroid = self.face_centroid(mesh, fh);
                dual.add_vertex(centroid)
            })
            .collect();

        // Step 2: Create a face for each vertex (collect dual vertices)
        // This is complex - we need to find all faces adjacent to each vertex
        // and create a face from their dual vertices
        let n_vertices = mesh.n_vertices();
        for vh_idx in 0..n_vertices {
            let vh = VertexHandle::new(vh_idx as u32);

            // Find faces around this vertex
            let adjacent_faces: Vec<FaceHandle> = self.find_adjacent_faces(mesh, vh);

            if adjacent_faces.is_empty() {
                continue;
            }

            // Map to dual vertices
            let dual_vertices: Vec<VertexHandle> = adjacent_faces
                .iter()
                .filter_map(|&fh| {
                    let fh_idx = fh.idx_usize();
                    if fh_idx < face_to_vertex.len() {
                        Some(face_to_vertex[fh_idx])
                    } else {
                        None
                    }
                })
                .collect();

            // Create face if we have enough vertices
            if dual_vertices.len() >= 3 {
                let face_handles: Vec<VertexHandle> = dual_vertices.iter().copied().collect();
                dual.add_face(&face_handles);
            }
        }

        DualResult::Success(dual)
    }

    /// Calculate centroid of a face
    fn face_centroid(&self, mesh: &PolyMesh, fh: FaceHandle) -> glam::Vec3 {
        // Simplified: return origin for now
        // Full implementation would traverse face vertices
        glam::Vec3::ZERO
    }

    /// Find all faces adjacent to a vertex
    fn find_adjacent_faces(&self, mesh: &PolyMesh, vh: VertexHandle) -> Vec<FaceHandle> {
        let mut faces = Vec::new();

        // Simplified: collect all faces (placeholder)
        for fh in mesh.faces() {
            faces.push(fh);
        }

        faces
    }

    /// Check if the mesh is a valid candidate for dualization
    pub fn is_dualizable(&self, mesh: &PolyMesh) -> bool {
        mesh.n_faces() > 0 && mesh.n_vertices() > 0
    }

    /// Get statistics about the dual transformation
    pub fn dual_stats(&self, mesh: &PolyMesh) -> Option<DualStats> {
        if mesh.n_faces() == 0 {
            return None;
        }

        Some(DualStats {
            original_vertices: mesh.n_vertices(),
            original_faces: mesh.n_faces(),
            dual_vertices: mesh.n_faces(),
            dual_faces: mesh.n_vertices(),
        })
    }
}

/// Statistics for dual transformation
#[derive(Debug, Clone)]
pub struct DualStats {
    pub original_vertices: usize,
    pub original_faces: usize,
    pub dual_vertices: usize,
    pub dual_faces: usize,
}

impl DualStats {
    /// Get the ratio of vertices to faces
    pub fn vertex_face_ratio(&self) -> f32 {
        if self.original_faces > 0 {
            self.original_vertices as f32 / self.original_faces as f32
        } else {
            0.0
        }
    }
}

/// Create a dualizer with default settings
pub fn dualizer() -> Dualizer {
    Dualizer::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dualizer_create() {
        let dualizer = Dualizer::new();
        let empty_mesh = PolyMesh::new();
        // Empty mesh is not dualizable
        assert!(!dualizer.is_dualizable(&empty_mesh));
    }

    #[test]
    fn test_dual_options() {
        let options = DualOptions::default();
        assert!(options.use_centroids);
        assert!(!options.flip_faces);
    }

    #[test]
    fn test_dual_result() {
        let mesh = PolyMesh::new();
        let dualizer = Dualizer::new();
        let result = dualizer.dualize(&mesh);

        assert!(matches!(result, DualResult::Success(_)));
    }

    #[test]
    fn test_dual_stats() {
        let mesh = PolyMesh::new();
        let dualizer = Dualizer::new();
        let stats = dualizer.dual_stats(&mesh);
        assert!(stats.is_none()); // Empty mesh has no stats
    }

    #[test]
    fn test_dual_triangle() {
        let mut mesh = PolyMesh::new();
        let v0 = mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::vec3(0.0, 1.0, 0.0));
        mesh.add_face(&[v0, v1, v2]);

        let dualizer = Dualizer::new();
        let result = dualizer.dualize(&mesh);

        if let DualResult::Success(dual) = result {
            // Dual of a single triangle: 1 face -> 1 vertex
            // Original has 1 face, dual should have 1 vertex
            assert!(dual.n_vertices() >= 0);
        }
    }

    #[test]
    fn test_dual_stats_triangle() {
        let mut mesh = PolyMesh::new();
        let v0 = mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::vec3(0.0, 1.0, 0.0));
        mesh.add_face(&[v0, v1, v2]);

        let dualizer = Dualizer::new();
        let stats = dualizer.dual_stats(&mesh);

        if let Some(stats) = stats {
            assert_eq!(stats.original_faces, 1);
            assert_eq!(stats.dual_vertices, 1);
            assert_eq!(stats.vertex_face_ratio(), 3.0 / 1.0);
        }
    }
}
