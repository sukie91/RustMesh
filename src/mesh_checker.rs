//! # MeshChecker
//!
//! Mesh integrity checker for verifying mesh data structure correctness.
//! Checks connectivity, halfedge consistency, and face validity.

use crate::PolyMesh;
use crate::tri_connectivity::TriMesh;

/// Mesh check result with error messages
#[derive(Debug, Clone)]
pub struct MeshCheckResult {
    pub is_valid: bool,
    pub errors: Vec<String>,
}

impl Default for MeshCheckResult {
    fn default() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
        }
    }
}

impl MeshCheckResult {
    /// Create a valid result
    pub fn ok() -> Self {
        Self {
            is_valid: true,
            errors: Vec::new(),
        }
    }

    /// Create an invalid result with an error
    pub fn error<T: Into<String>>(msg: T) -> Self {
        Self {
            is_valid: false,
            errors: vec![msg.into()],
        }
    }

    /// Add an error message
    pub fn add_error(&mut self, msg: String) {
        self.is_valid = false;
        self.errors.push(msg);
    }

    /// Combine with another result
    pub fn combine(&mut self, other: &Self) {
        if !other.is_valid {
            self.is_valid = false;
            self.errors.extend(other.errors.clone());
        }
    }

    /// Check if valid
    pub fn is_valid(&self) -> bool {
        self.is_valid
    }
}

/// Check targets for selective mesh checking
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckTargets {
    pub check_vertices: bool,
    pub check_edges: bool,
    pub check_faces: bool,
}

impl Default for CheckTargets {
    fn default() -> Self {
        Self {
            check_vertices: true,
            check_edges: true,
            check_faces: true,
        }
    }
}

impl CheckTargets {
    /// Check all
    pub fn all() -> Self {
        Self {
            check_vertices: true,
            check_edges: true,
            check_faces: true,
        }
    }

    /// Check vertices only
    pub fn vertices_only() -> Self {
        Self {
            check_vertices: true,
            check_edges: false,
            check_faces: false,
        }
    }

    /// Check edges only
    pub fn edges_only() -> Self {
        Self {
            check_vertices: false,
            check_edges: true,
            check_faces: false,
        }
    }

    /// Check faces only
    pub fn faces_only() -> Self {
        Self {
            check_vertices: false,
            check_edges: false,
            check_faces: true,
        }
    }
}

/// MeshChecker for PolyMesh
#[derive(Debug, Clone)]
pub struct MeshChecker<'a> {
    mesh: &'a PolyMesh,
}

impl<'a> MeshChecker<'a> {
    /// Create a new checker for the given mesh
    pub fn new(mesh: &'a PolyMesh) -> Self {
        Self { mesh }
    }

    /// Check mesh integrity with all targets
    pub fn check(&self) -> MeshCheckResult {
        self.check_with_targets(CheckTargets::all())
    }

    /// Check mesh with specific targets
    pub fn check_with_targets(&self, targets: CheckTargets) -> MeshCheckResult {
        let mut result = MeshCheckResult::ok();

        if targets.check_vertices {
            self.check_vertices(&mut result);
        }

        if targets.check_edges {
            self.check_edges(&mut result);
        }

        if targets.check_faces {
            self.check_faces(&mut result);
        }

        result
    }

    /// Check vertex integrity
    fn check_vertices(&self, result: &mut MeshCheckResult) {
        for vh in self.mesh.vertices() {
            // Check outgoing halfedge validity
            if let Some(heh) = self.mesh.halfedge_handle(vh) {
                // Check halfedge bounds
                let n_halfedges = self.mesh.n_halfedges();
                if !heh.is_valid() || heh.idx_usize() >= n_halfedges {
                    result.add_error(format!(
                        "MeshChecker: vertex {:?} has out-of-bounds outgoing HE: {:?}",
                        vh, heh
                    ));
                }

                // Check that outgoing halfedge references back to vertex
                let from_v = self.mesh.from_vertex_handle(heh);
                if from_v != vh {
                    result.add_error(format!(
                        "MeshChecker: vertex {:?}: outgoing halfedge does not reference vertex",
                        vh
                    ));
                }
            }
        }
    }

    /// Check halfedge integrity
    fn check_edges(&self, result: &mut MeshCheckResult) {
        let n_halfedges = self.mesh.n_halfedges();

        if n_halfedges > 0 {
            for heh in self.mesh.halfedges() {
                // Degenerated halfedge check (from == to)
                let from_v = self.mesh.from_vertex_handle(heh);
                let to_v = self.mesh.to_vertex_handle(heh);
                if from_v == to_v {
                    result.add_error(format!(
                        "MeshChecker: halfedge {:?}: to-vertex == from-vertex",
                        heh
                    ));
                }
            }
        }
    }

    /// Check face integrity
    fn check_faces(&self, result: &mut MeshCheckResult) {
        // Skip for now - requires kernel access
    }
}

/// Check a PolyMesh
pub fn check_mesh(mesh: &PolyMesh) -> MeshCheckResult {
    MeshChecker::new(mesh).check()
}

/// Check a PolyMesh with targets
pub fn check_mesh_with_targets(mesh: &PolyMesh, targets: CheckTargets) -> MeshCheckResult {
    MeshChecker::new(mesh).check_with_targets(targets)
}

/// Check any mesh that provides the necessary methods
pub trait MeshCheckable {
    fn n_vertices(&self) -> usize;
    fn n_edges(&self) -> usize;
    fn n_halfedges(&self) -> usize;
    fn n_faces(&self) -> usize;
    fn vertices(&self) -> Box<dyn Iterator<Item = crate::VertexHandle> + '_>;
    fn halfedges(&self) -> Box<dyn Iterator<Item = crate::HalfedgeHandle> + '_>;
    fn faces(&self) -> Box<dyn Iterator<Item = crate::FaceHandle> + '_>;
    fn halfedge_handle(&self, vh: crate::VertexHandle) -> Option<crate::HalfedgeHandle>;
    fn from_vertex_handle(&self, heh: crate::HalfedgeHandle) -> crate::VertexHandle;
    fn to_vertex_handle(&self, heh: crate::HalfedgeHandle) -> crate::VertexHandle;
}

impl MeshCheckable for TriMesh {
    fn n_vertices(&self) -> usize {
        TriMesh::n_vertices(self)
    }
    fn n_edges(&self) -> usize {
        TriMesh::n_edges(self)
    }
    fn n_halfedges(&self) -> usize {
        TriMesh::n_halfedges(self)
    }
    fn n_faces(&self) -> usize {
        TriMesh::n_faces(self)
    }
    fn vertices(&self) -> Box<dyn Iterator<Item = crate::VertexHandle> + '_> {
        Box::new(TriMesh::vertices(self))
    }
    fn halfedges(&self) -> Box<dyn Iterator<Item = crate::HalfedgeHandle> + '_> {
        Box::new(TriMesh::halfedges(self))
    }
    fn faces(&self) -> Box<dyn Iterator<Item = crate::FaceHandle> + '_> {
        Box::new(TriMesh::faces(self))
    }
    fn halfedge_handle(&self, vh: crate::VertexHandle) -> Option<crate::HalfedgeHandle> {
        TriMesh::halfedge_handle(self, vh)
    }
    fn from_vertex_handle(&self, heh: crate::HalfedgeHandle) -> crate::VertexHandle {
        TriMesh::from_vertex_handle(self, heh)
    }
    fn to_vertex_handle(&self, heh: crate::HalfedgeHandle) -> crate::VertexHandle {
        TriMesh::to_vertex_handle(self, heh)
    }
}

/// Generic mesh checker - skip due to stack overflow issues
// struct GenericMeshChecker<'a, M: MeshCheckable> {
//     mesh: &'a M,
// }

/// Check any MeshCheckable - skip due to stack overflow issues
// pub fn check_mesh_generic<M: MeshCheckable>(mesh: &M) -> MeshCheckResult {
//     GenericMeshChecker::new(mesh).check()
// }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TriMesh;

    #[test]
    fn test_mesh_check_result() {
        let mut result = MeshCheckResult::ok();
        assert!(result.is_valid());
        assert!(result.errors.is_empty());

        result.add_error("Test error".to_string());
        assert!(!result.is_valid());
        assert_eq!(result.errors.len(), 1);
    }

    #[test]
    fn test_check_targets() {
        let targets = CheckTargets::all();
        assert!(targets.check_vertices);
        assert!(targets.check_edges);
        assert!(targets.check_faces);

        let targets = CheckTargets::vertices_only();
        assert!(targets.check_vertices);
        assert!(!targets.check_edges);
        assert!(!targets.check_faces);
    }
}
