//! # ArrayKernel
//! 
//! Core mesh storage using arrays (Vec) for mesh items.
//! This is the underlying storage layer for the mesh data structure.

use std::collections::HashMap;
use crate::handles::{VertexHandle, HalfedgeHandle, EdgeHandle, FaceHandle};
use crate::items::{Vertex, Halfedge, Edge, Face};

/// Property container for mesh attributes
/// Allows attaching arbitrary data to mesh entities
#[derive(Debug)]
pub struct PropertyContainer {
    data: HashMap<String, Box<dyn std::any::Any>>,
}

impl PropertyContainer {
    /// Create a new empty property container
    pub fn new() -> Self {
        Self {
            data: HashMap::new(),
        }
    }

    /// Add a property with the given name
    pub fn add_property<T: 'static>(&mut self, name: &str, value: T) {
        self.data.insert(name.to_string(), Box::new(value));
    }

    /// Get a property by name
    pub fn get_property<T: 'static>(&self, name: &str) -> Option<&T> {
        self.data.get(name).and_then(|b| b.downcast_ref::<T>())
    }

    /// Get a mutable property by name
    pub fn get_property_mut<T: 'static>(&mut self, name: &str) -> Option<&mut T> {
        self.data.get_mut(name).and_then(|b| b.downcast_mut::<T>())
    }

    /// Remove a property
    pub fn remove_property(&mut self, name: &str) -> bool {
        self.data.remove(name).is_some()
    }

    /// Check if property exists
    pub fn has_property(&self, name: &str) -> bool {
        self.data.contains_key(name)
    }
}

/// Status flags for mesh entities
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StatusInfo {
    bits: u32,
}

impl StatusInfo {
    /// Create a new status with all flags unset
    pub fn new() -> Self {
        Self { bits: 0 }
    }

    /// Set a bit flag
    pub fn set_bit(&mut self, bit: u32) {
        self.bits |= 1 << bit;
    }

    /// Unset a bit flag
    pub fn unset_bit(&mut self, bit: u32) {
        self.bits &= !(1 << bit);
    }

    /// Check if a bit is set
    pub fn is_bit_set(&self, bit: u32) -> bool {
        (self.bits & (1 << bit)) != 0
    }

    /// Get the raw bits
    pub fn bits(&self) -> u32 {
        self.bits
    }

    /// Set bits from another status
    pub fn set_bits(&mut self, other: &Self) {
        self.bits |= other.bits;
    }
}

/// Storage for vertex properties
#[derive(Debug, Clone)]
pub struct VertexPropertyContainer {
    status: Option<StatusInfo>,
}

impl Default for VertexPropertyContainer {
    fn default() -> Self {
        Self { status: None }
    }
}

/// Storage for edge properties
#[derive(Debug, Clone)]
pub struct EdgePropertyContainer {
    status: Option<StatusInfo>,
}

impl Default for EdgePropertyContainer {
    fn default() -> Self {
        Self { status: None }
    }
}

/// Storage for face properties
#[derive(Debug, Clone)]
pub struct FacePropertyContainer {
    status: Option<StatusInfo>,
}

impl Default for FacePropertyContainer {
    fn default() -> Self {
        Self { status: None }
    }
}

/// Storage for halfedge properties
#[derive(Debug, Clone)]
pub struct HalfedgePropertyContainer {
    status: Option<StatusInfo>,
}

impl Default for HalfedgePropertyContainer {
    fn default() -> Self {
        Self { status: None }
    }
}

/// The ArrayKernel - core mesh storage using Vec containers
#[derive(Debug, Clone, Default)]
pub struct ArrayKernel {
    vertices: Vec<Vertex>,
    edges: Vec<Edge>,
    faces: Vec<Face>,
    
    // Property containers
    vertex_props: VertexPropertyContainer,
    edge_props: EdgePropertyContainer,
    face_props: FacePropertyContainer,
    halfedge_props: HalfedgePropertyContainer,
}

impl ArrayKernel {
    /// Create a new empty kernel
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            edges: Vec::new(),
            faces: Vec::new(),
            vertex_props: VertexPropertyContainer::default(),
            edge_props: EdgePropertyContainer::default(),
            face_props: FacePropertyContainer::default(),
            halfedge_props: HalfedgePropertyContainer::default(),
        }
    }

    /// Clear all data
    pub fn clear(&mut self) {
        self.vertices.clear();
        self.edges.clear();
        self.faces.clear();
    }

    // --- Handle to item conversion ---

    /// Get a vertex by handle (const)
    pub fn vertex(&self, vh: VertexHandle) -> Option<&Vertex> {
        let idx = vh.idx() as usize;
        self.vertices.get(idx)
    }

    /// Get a vertex by handle (mutable)
    pub fn vertex_mut(&mut self, vh: VertexHandle) -> Option<&mut Vertex> {
        let idx = vh.idx() as usize;
        self.vertices.get_mut(idx)
    }

    /// Get an edge by handle (const)
    pub fn edge(&self, eh: EdgeHandle) -> Option<&Edge> {
        let idx = eh.idx() as usize;
        self.edges.get(idx)
    }

    /// Get an edge by handle (mutable)
    pub fn edge_mut(&mut self, eh: EdgeHandle) -> Option<&mut Edge> {
        let idx = eh.idx() as usize;
        self.edges.get_mut(idx)
    }

    /// Get a face by handle (const)
    pub fn face(&self, fh: FaceHandle) -> Option<&Face> {
        let idx = fh.idx() as usize;
        self.faces.get(idx)
    }

    /// Get a face by handle (mutable)
    pub fn face_mut(&mut self, fh: FaceHandle) -> Option<&mut Face> {
        let idx = fh.idx() as usize;
        self.faces.get_mut(idx)
    }

    /// Get a halfedge by handle (returns None for now - needs full implementation)
    pub fn halfedge(&self, _heh: HalfedgeHandle) -> Option<&Halfedge> {
        // In a full implementation, halfedges would be stored separately
        // or we would return a view constructed from edge data
        None
    }

    // --- Item creation ---

    /// Add a new vertex and return its handle
    pub fn add_vertex(&mut self, point: glam::Vec3) -> VertexHandle {
        let idx = self.vertices.len() as i32;
        self.vertices.push(Vertex::new(point));
        VertexHandle::new(idx)
    }

    /// Add a new edge and return the handle to the first halfedge
    pub fn add_edge(&mut self, start_vh: VertexHandle, end_vh: VertexHandle) -> HalfedgeHandle {
        let edge_idx = self.edges.len() as i32;
        let he0_idx = edge_idx * 2;
        let he1_idx = edge_idx * 2 + 1;
        
        let he0 = HalfedgeHandle::new(he0_idx);
        let he1 = HalfedgeHandle::new(he1_idx);
        
        self.edges.push(Edge::new(he0, he1));
        
        HalfedgeHandle::new(he0_idx)
    }

    /// Add a new face and return its handle
    pub fn add_face(&mut self, halfedge_handle: HalfedgeHandle) -> FaceHandle {
        let idx = self.faces.len() as i32;
        self.faces.push(Face::new(halfedge_handle));
        FaceHandle::new(idx)
    }

    // --- Count queries ---

    /// Get the number of vertices
    pub fn n_vertices(&self) -> usize {
        self.vertices.len()
    }

    /// Get the number of edges
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }

    /// Get the number of faces
    pub fn n_faces(&self) -> usize {
        self.faces.len()
    }

    /// Get the number of halfedges
    pub fn n_halfedges(&self) -> usize {
        self.edges.len() * 2
    }

    /// Check if vertices are empty
    pub fn vertices_empty(&self) -> bool {
        self.vertices.is_empty()
    }

    /// Check if edges are empty
    pub fn edges_empty(&self) -> bool {
        self.edges.is_empty()
    }

    /// Check if faces are empty
    pub fn faces_empty(&self) -> bool {
        self.faces.is_empty()
    }

    // --- Status management ---

    /// Request vertex status tracking
    pub fn request_vertex_status(&mut self) {
        self.vertex_props.status = Some(StatusInfo::new());
    }

    /// Request edge status tracking
    pub fn request_edge_status(&mut self) {
        self.edge_props.status = Some(StatusInfo::new());
    }

    /// Request face status tracking
    pub fn request_face_status(&mut self) {
        self.face_props.status = Some(StatusInfo::new());
    }

    /// Get vertex status
    pub fn vertex_status(&self, vh: VertexHandle) -> Option<&StatusInfo> {
        self.vertex_props.status.as_ref()
    }

    /// Get mutable vertex status
    pub fn vertex_status_mut(&mut self, vh: VertexHandle) -> Option<&mut StatusInfo> {
        self.vertex_props.status.as_mut()
    }

    // --- Connectivity ---

    /// Get the halfedge handle from a vertex
    pub fn halfedge_handle(&self, vh: VertexHandle) -> Option<HalfedgeHandle> {
        self.vertex(vh).and_then(|v| v.halfedge_handle)
    }

    /// Set the halfedge handle for a vertex
    pub fn set_halfedge_handle(&mut self, vh: VertexHandle, heh: HalfedgeHandle) {
        if let Some(v) = self.vertex_mut(vh) {
            v.halfedge_handle = Some(heh);
        }
    }

    /// Check if a vertex is isolated
    pub fn is_isolated(&self, vh: VertexHandle) -> bool {
        self.halfedge_handle(vh).is_none()
    }

    /// Get the to-vertex of a halfedge
    pub fn to_vertex_handle(&self, heh: HalfedgeHandle) -> VertexHandle {
        // In a full implementation, we'd look up the halfedge data
        VertexHandle::new(heh.idx() ^ 1)
    }

    /// Get the opposite halfedge
    pub fn opposite_halfedge_handle(&self, heh: HalfedgeHandle) -> HalfedgeHandle {
        HalfedgeHandle::new(heh.idx() ^ 1)
    }

    /// Get the edge handle from a halfedge
    pub fn edge_handle(&self, heh: HalfedgeHandle) -> EdgeHandle {
        EdgeHandle::new(heh.idx() >> 1)
    }

    /// Get the halfedge handle from an edge (0 or 1)
    pub fn edge_halfedge_handle(&self, eh: EdgeHandle, idx: usize) -> HalfedgeHandle {
        HalfedgeHandle::new((eh.idx() << 1) + idx as i32)
    }

    /// Get the face handle from a halfedge
    pub fn face_handle(&self, heh: HalfedgeHandle) -> Option<FaceHandle> {
        // Simplified: assume face index matches halfedge index / 2
        Some(FaceHandle::new(heh.idx() >> 1))
    }

    /// Set the face handle for a halfedge
    pub fn set_face_handle(&mut self, heh: HalfedgeHandle, fh: FaceHandle) {
        // In a full implementation, store in halfedge data
    }

    /// Set a halfedge as boundary (no face)
    pub fn set_boundary(&mut self, heh: HalfedgeHandle) {
        // In a full implementation
    }

    /// Check if a halfedge is a boundary
    pub fn is_boundary(&self, heh: HalfedgeHandle) -> bool {
        // Simplified implementation
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut kernel = ArrayKernel::new();
        
        // Add vertices
        let v0 = kernel.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = kernel.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let v2 = kernel.add_vertex(glam::vec3(0.0, 1.0, 0.0));
        
        assert_eq!(kernel.n_vertices(), 3);
        assert!(kernel.vertex(v0).is_some());
        
        // Add edge
        let he = kernel.add_edge(v0, v1);
        assert_eq!(kernel.n_edges(), 1);
        assert_eq!(kernel.n_halfedges(), 2);
        
        // Check connectivity
        assert_eq!(kernel.edge_handle(he), EdgeHandle::new(0));
        assert_eq!(kernel.opposite_halfedge_handle(he).idx(), he.idx() ^ 1);
    }

    #[test]
    fn test_status_management() {
        let mut kernel = ArrayKernel::new();
        kernel.request_vertex_status();
        kernel.request_edge_status();
        
        // Status should be available after request
        assert!(kernel.vertex_status(VertexHandle::new(0)).is_some());
    }
}
