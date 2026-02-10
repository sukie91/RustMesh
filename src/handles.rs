//! # Handles
//! 
//! Handle types for mesh entities (Vertex, Edge, Halfedge, Face).
//! Handles are lightweight references to mesh elements using integer indices.

use std::fmt;

/// Base handle type for all mesh entities
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BaseHandle {
    idx: i32,
}

impl BaseHandle {
    /// Create a new handle with the given index (default: invalid)
    pub fn new(idx: i32) -> Self {
        Self { idx }
    }

    /// Get the underlying index
    pub fn idx(&self) -> i32 {
        self.idx
    }

    /// Check if the handle is valid (index >= 0)
    pub fn is_valid(&self) -> bool {
        self.idx >= 0
    }

    /// Invalidate the handle
    pub fn invalidate(&mut self) {
        self.idx = -1;
    }

    /// Reset to invalid state
    pub fn reset(&mut self) {
        self.invalidate();
    }
}

impl Default for BaseHandle {
    fn default() -> Self {
        Self::new(-1)
    }
}

impl fmt::Display for BaseHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.idx)
    }
}

/// Handle referencing a vertex entity
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VertexHandle(BaseHandle);

impl VertexHandle {
    /// Create a new vertex handle
    pub fn new(idx: i32) -> Self {
        Self(BaseHandle::new(idx))
    }

    /// Get the underlying index
    pub fn idx(&self) -> i32 {
        self.0.idx()
    }

    /// Check if valid
    pub fn is_valid(&self) -> bool {
        self.0.is_valid()
    }

    /// Invalidate the handle
    pub fn invalidate(&mut self) {
        self.0.invalidate();
    }
}

impl Default for VertexHandle {
    fn default() -> Self {
        Self(BaseHandle::default())
    }
}

impl From<i32> for VertexHandle {
    fn from(idx: i32) -> Self {
        Self::new(idx)
    }
}

impl From<usize> for VertexHandle {
    fn from(idx: usize) -> Self {
        Self::new(idx as i32)
    }
}

/// Handle referencing a halfedge entity
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HalfedgeHandle(BaseHandle);

impl HalfedgeHandle {
    /// Create a new halfedge handle
    pub fn new(idx: i32) -> Self {
        Self(BaseHandle::new(idx))
    }

    /// Get the underlying index
    pub fn idx(&self) -> i32 {
        self.0.idx()
    }

    /// Check if valid
    pub fn is_valid(&self) -> bool {
        self.0.is_valid()
    }

    /// Invalidate the handle
    pub fn invalidate(&mut self) {
        self.0.invalidate();
    }

    /// Get the opposite halfedge index (xor 1)
    pub fn opposite(&self) -> Self {
        Self::new(self.idx() ^ 1)
    }
}

impl Default for HalfedgeHandle {
    fn default() -> Self {
        Self(BaseHandle::default())
    }
}

impl From<i32> for HalfedgeHandle {
    fn from(idx: i32) -> Self {
        Self::new(idx)
    }
}

impl From<usize> for HalfedgeHandle {
    fn from(idx: usize) -> Self {
        Self::new(idx as i32)
    }
}

/// Handle referencing an edge entity
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EdgeHandle(BaseHandle);

impl EdgeHandle {
    /// Create a new edge handle
    pub fn new(idx: i32) -> Self {
        Self(BaseHandle::new(idx))
    }

    /// Get the underlying index
    pub fn idx(&self) -> i32 {
        self.0.idx()
    }

    /// Check if valid
    pub fn is_valid(&self) -> bool {
        self.0.is_valid()
    }

    /// Invalidate the handle
    pub fn invalidate(&mut self) {
        self.0.invalidate();
    }
}

impl Default for EdgeHandle {
    fn default() -> Self {
        Self(BaseHandle::default())
    }
}

impl From<i32> for EdgeHandle {
    fn from(idx: i32) -> Self {
        Self::new(idx)
    }
}

impl From<usize> for EdgeHandle {
    fn from(idx: usize) -> Self {
        Self::new(idx as i32)
    }
}

/// Handle referencing a face entity
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FaceHandle(BaseHandle);

impl FaceHandle {
    /// Create a new face handle
    pub fn new(idx: i32) -> Self {
        Self(BaseHandle::new(idx))
    }

    /// Get the underlying index
    pub fn idx(&self) -> i32 {
        self.0.idx()
    }

    /// Check if valid
    pub fn is_valid(&self) -> bool {
        self.0.is_valid()
    }

    /// Invalidate the handle
    pub fn invalidate(&mut self) {
        self.0.invalidate();
    }
}

impl Default for FaceHandle {
    fn default() -> Self {
        Self(BaseHandle::default())
    }
}

impl From<i32> for FaceHandle {
    fn from(idx: i32) -> Self {
        Self::new(idx)
    }
}

impl From<usize> for FaceHandle {
    fn from(idx: usize) -> Self {
        Self::new(idx as i32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handle_validity() {
        let valid = VertexHandle::new(0);
        let invalid = VertexHandle::default();
        
        assert!(valid.is_valid());
        assert!(!invalid.is_valid());
    }

    #[test]
    fn test_halfedge_opposite() {
        let he = HalfedgeHandle::new(5);
        assert_eq!(he.opposite().idx(), 4);
        
        let he2 = HalfedgeHandle::new(4);
        assert_eq!(he2.opposite().idx(), 5);
    }
}
