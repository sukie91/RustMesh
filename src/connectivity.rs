//! # PolyConnectivity
//! 
//! Polygonal mesh connectivity implementation.
//! Provides iteration and circulation over mesh elements.

use std::iter::{IntoIterator, Iterator};
use crate::handles::{VertexHandle, HalfedgeHandle, EdgeHandle, FaceHandle};
use crate::kernel::ArrayKernel;
use crate::items::Vertex;

/// Vertex iterator
#[derive(Debug)]
pub struct VertexIter<'a> {
    kernel: &'a ArrayKernel,
    current: usize,
}

impl<'a> VertexIter<'a> {
    fn new(kernel: &'a ArrayKernel) -> Self {
        Self {
            kernel,
            current: 0,
        }
    }
}

impl<'a> Iterator for VertexIter<'a> {
    type Item = VertexHandle;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.kernel.n_vertices() {
            let handle = VertexHandle::new(self.current as i32);
            self.current += 1;
            Some(handle)
        } else {
            None
        }
    }
}

/// Edge iterator
#[derive(Debug)]
pub struct EdgeIter<'a> {
    kernel: &'a ArrayKernel,
    current: usize,
}

impl<'a> EdgeIter<'a> {
    fn new(kernel: &'a ArrayKernel) -> Self {
        Self {
            kernel,
            current: 0,
        }
    }
}

impl<'a> Iterator for EdgeIter<'a> {
    type Item = EdgeHandle;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.kernel.n_edges() {
            let handle = EdgeHandle::new(self.current as i32);
            self.current += 1;
            Some(handle)
        } else {
            None
        }
    }
}

/// Face iterator
#[derive(Debug)]
pub struct FaceIter<'a> {
    kernel: &'a ArrayKernel,
    current: usize,
}

impl<'a> FaceIter<'a> {
    fn new(kernel: &'a ArrayKernel) -> Self {
        Self {
            kernel,
            current: 0,
        }
    }
}

impl<'a> Iterator for FaceIter<'a> {
    type Item = FaceHandle;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.kernel.n_faces() {
            let handle = FaceHandle::new(self.current as i32);
            self.current += 1;
            Some(handle)
        } else {
            None
        }
    }
}

/// Halfedge iterator
#[derive(Debug)]
pub struct HalfedgeIter<'a> {
    kernel: &'a ArrayKernel,
    current: usize,
    total: usize,
}

impl<'a> HalfedgeIter<'a> {
    fn new(kernel: &'a ArrayKernel) -> Self {
        let total = kernel.n_halfedges();
        Self {
            kernel,
            current: 0,
            total,
        }
    }
}

impl<'a> Iterator for HalfedgeIter<'a> {
    type Item = HalfedgeHandle;
    
    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.total {
            let handle = HalfedgeHandle::new(self.current as i32);
            self.current += 1;
            Some(handle)
        } else {
            None
        }
    }
}

/// Circulator for vertices around a vertex (1-ring)
pub struct VertexVertexCirculator<'a> {
    kernel: &'a ArrayKernel,
    center: VertexHandle,
    current: Option<HalfedgeHandle>,
    started: bool,
}

impl<'a> VertexVertexCirculator<'a> {
    fn new(kernel: &'a ArrayKernel, vh: VertexHandle) -> Self {
        let start_heh = kernel.halfedge_handle(vh);
        Self {
            kernel,
            center: vh,
            current: start_heh,
            started: false,
        }
    }
}

impl<'a> Iterator for VertexVertexCirculator<'a> {
    type Item = VertexHandle;
    
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(heh) = self.current {
            let next_vh = self.kernel.to_vertex_handle(heh);
            
            // Move to next halfedge around the vertex
            // For a full implementation, we'd need proper prev/next linkage
            self.current = Some(self.kernel.opposite_halfedge_handle(heh));
            
            // Skip if we've gone full circle
            if Some(next_vh) == self.kernel.halfedge_handle(self.center).map(|h| self.kernel.to_vertex_handle(h)) {
                if self.started {
                    return None;
                }
            }
            self.started = true;
            
            Some(next_vh)
        } else {
            None
        }
    }
}

/// Vertex-face circulator
pub struct VertexFaceCirculator<'a> {
    kernel: &'a ArrayKernel,
    center: VertexHandle,
    current: Option<HalfedgeHandle>,
}

impl<'a> VertexFaceCirculator<'a> {
    fn new(kernel: &'a ArrayKernel, vh: VertexHandle) -> Self {
        let start_heh = kernel.halfedge_handle(vh);
        Self {
            kernel,
            center: vh,
            current: start_heh,
        }
    }
}

impl<'a> Iterator for VertexFaceCirculator<'a> {
    type Item = FaceHandle;
    
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(heh) = self.current {
            // Get face from halfedge
            let fh = self.kernel.face_handle(heh);
            
            // Move to next halfedge around vertex
            // In a full implementation, use proper circulation
            self.current = Some(self.kernel.opposite_halfedge_handle(heh));
            
            fh
        } else {
            None
        }
    }
}

/// Polygonal mesh with full connectivity
#[derive(Debug, Clone, Default)]
pub struct PolyMesh {
    kernel: ArrayKernel,
}

impl PolyMesh {
    /// Create a new empty polygon mesh
    pub fn new() -> Self {
        Self {
            kernel: ArrayKernel::new(),
        }
    }

    /// Clear the mesh
    pub fn clear(&mut self) {
        self.kernel.clear();
    }

    // --- Vertex operations ---

    /// Add a vertex at the given position
    pub fn add_vertex(&mut self, point: glam::Vec3) -> VertexHandle {
        self.kernel.add_vertex(point)
    }

    /// Get vertex position
    pub fn point(&self, vh: VertexHandle) -> Option<glam::Vec3> {
        self.kernel.vertex(vh).map(|v| v.point)
    }

    /// Set vertex position
    pub fn set_point(&mut self, vh: VertexHandle, point: glam::Vec3) {
        if let Some(v) = self.kernel.vertex_mut(vh) {
            v.point = point;
        }
    }

    // --- Edge operations ---

    /// Add an edge between two vertices
    pub fn add_edge(&mut self, v0: VertexHandle, v1: VertexHandle) -> HalfedgeHandle {
        self.kernel.add_edge(v0, v1)
    }

    // --- Face operations ---

    /// Add a face from a list of vertex handles
    pub fn add_face(&mut self, vertices: &[VertexHandle]) -> Option<FaceHandle> {
        if vertices.len() < 3 {
            return None;
        }

        // Create halfedges for each edge of the face
        let n = vertices.len();
        let mut halfedges: Vec<HalfedgeHandle> = Vec::with_capacity(n);

        for i in 0..n {
            let start = vertices[i];
            let end = vertices[(i + 1) % n];
            let he = self.add_edge(end, start); // Halfedge points to end
            halfedges.push(he);
        }

        // Link halfedges into a cycle
        for i in 0..n {
            let curr = halfedges[i];
            let next = halfedges[(i + 1) % n];
            
            // In a full implementation, set next/prev pointers
            // self.kernel.set_next_halfedge_handle(curr, next);
        }

        // Create the face
        let fh = self.kernel.add_face(halfedges[0]);

        // Connect vertices to halfedges
        for (i, &vh) in vertices.iter().enumerate() {
            self.kernel.set_halfedge_handle(vh, halfedges[i]);
        }

        Some(fh)
    }

    // --- Iteration ---

    /// Get an iterator over all vertices
    pub fn vertices(&self) -> VertexIter<'_> {
        VertexIter::new(&self.kernel)
    }

    /// Get an iterator over all edges
    pub fn edges(&self) -> EdgeIter<'_> {
        EdgeIter::new(&self.kernel)
    }

    /// Get an iterator over all faces
    pub fn faces(&self) -> FaceIter<'_> {
        FaceIter::new(&self.kernel)
    }

    /// Get an iterator over all halfedges
    pub fn halfedges(&self) -> HalfedgeIter<'_> {
        HalfedgeIter::new(&self.kernel)
    }

    /// Get a circulator for vertices around a vertex
    pub fn vv_circulator(&self, vh: VertexHandle) -> VertexVertexCirculator<'_> {
        VertexVertexCirculator::new(&self.kernel, vh)
    }

    /// Get a circulator for faces around a vertex
    pub fn vf_circulator(&self, vh: VertexHandle) -> VertexFaceCirculator<'_> {
        VertexFaceCirculator::new(&self.kernel, vh)
    }

    // --- Count queries ---

    /// Get the number of vertices
    pub fn n_vertices(&self) -> usize {
        self.kernel.n_vertices()
    }

    /// Get the number of edges
    pub fn n_edges(&self) -> usize {
        self.kernel.n_edges()
    }

    /// Get the number of faces
    pub fn n_faces(&self) -> usize {
        self.kernel.n_faces()
    }

    /// Get the number of halfedges
    pub fn n_halfedges(&self) -> usize {
        self.kernel.n_halfedges()
    }

    // --- Connectivity queries ---

    /// Get the halfedge handle from a vertex
    pub fn halfedge_handle(&self, vh: VertexHandle) -> Option<HalfedgeHandle> {
        self.kernel.halfedge_handle(vh)
    }

    /// Get the edge handle from a halfedge
    pub fn edge_handle(&self, heh: HalfedgeHandle) -> EdgeHandle {
        self.kernel.edge_handle(heh)
    }

    /// Get the opposite halfedge
    pub fn opposite_halfedge_handle(&self, heh: HalfedgeHandle) -> HalfedgeHandle {
        self.kernel.opposite_halfedge_handle(heh)
    }

    /// Get the face handle from a halfedge
    pub fn face_handle(&self, heh: HalfedgeHandle) -> Option<FaceHandle> {
        self.kernel.face_handle(heh)
    }

    /// Check if a halfedge is a boundary
    pub fn is_boundary(&self, heh: HalfedgeHandle) -> bool {
        self.kernel.is_boundary(heh)
    }

    /// Get the vertices of a face
    pub fn face_vertices(&self, fh: FaceHandle) -> Option<Vec<VertexHandle>> {
        // Simplified implementation: return empty for now
        // A full implementation would traverse the halfedge cycle
        Some(vec![])
    }
}

// Type aliases for compatibility with OpenMesh API
// Note: These require lifetime parameters when used

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_triangle() {
        let mut mesh = PolyMesh::new();
        
        // Add vertices
        let v0 = mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::vec3(0.0, 1.0, 0.0));
        
        // Add face
        let face = mesh.add_face(&[v0, v1, v2]);
        
        assert!(face.is_some());
        assert_eq!(mesh.n_vertices(), 3);
        assert_eq!(mesh.n_faces(), 1);
    }

    #[test]
    fn test_vertex_iteration() {
        let mut mesh = PolyMesh::new();
        
        mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        mesh.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        mesh.add_vertex(glam::vec3(0.0, 1.0, 0.0));
        
        let mut count = 0;
        for v in mesh.vertices() {
            assert!(mesh.point(v).is_some());
            count += 1;
        }
        assert_eq!(count, 3);
    }
}
