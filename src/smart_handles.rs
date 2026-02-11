//! # SmartHandles
//!
//! Simple smart handles with fluent API for PolyMesh.

use crate::handles::{VertexHandle, HalfedgeHandle, FaceHandle};
use crate::PolyMesh;

/// Smart vertex handle for PolyMesh
#[derive(Debug, Clone, Copy)]
pub struct SmartVertex<'a> {
    vh: VertexHandle,
    mesh: &'a PolyMesh,
}

impl<'a> SmartVertex<'a> {
    pub fn new(vh: VertexHandle, mesh: &'a PolyMesh) -> Self {
        Self { vh, mesh }
    }
    pub fn point(&self) -> Option<glam::Vec3> {
        self.mesh.point(self.vh)
    }
}

/// Smart halfedge handle for PolyMesh
#[derive(Debug, Clone, Copy)]
pub struct SmartHalfedge<'a> {
    heh: HalfedgeHandle,
    mesh: &'a PolyMesh,
}

impl<'a> SmartHalfedge<'a> {
    pub fn new(heh: HalfedgeHandle, mesh: &'a PolyMesh) -> Self {
        Self { heh, mesh }
    }
    pub fn opposite(&self) -> Self {
        Self::new(self.mesh.opposite_halfedge_handle(self.heh), self.mesh)
    }
    pub fn next(&self) -> Self {
        Self::new(self.mesh.next_halfedge_handle(self.heh), self.mesh)
    }
    pub fn is_boundary(&self) -> bool {
        self.mesh.is_boundary(self.heh)
    }
}

/// Smart face handle for PolyMesh
#[derive(Debug, Clone, Copy)]
pub struct SmartFace<'a> {
    fh: FaceHandle,
    mesh: &'a PolyMesh,
}

impl<'a> SmartFace<'a> {
    pub fn new(fh: FaceHandle, mesh: &'a PolyMesh) -> Self {
        Self { fh, mesh }
    }
    pub fn is_boundary(&self) -> bool {
        if let Some(heh) = self.mesh.face_halfedge_handle(self.fh) {
            self.mesh.is_boundary(heh)
        } else { false }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PolyMesh;

    #[test]
    fn test_smart_polymesh() {
        let mut mesh = PolyMesh::new();
        let v0 = mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::vec3(0.0, 1.0, 0.0));
        mesh.add_face(&[v0, v1, v2]);

        let sv = SmartVertex::new(v0, &mesh);
        assert!(sv.point().is_some());

        if let Some(heh) = mesh.halfedge_handle(v0) {
            let sh = SmartHalfedge::new(heh, &mesh);
            let _opp = sh.opposite();
            let _next = sh.next();
        }

        for fh in mesh.faces() {
            let sf = SmartFace::new(fh, &mesh);
            let _ = sf.is_boundary();
        }
    }
}
