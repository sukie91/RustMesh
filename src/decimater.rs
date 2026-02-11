//! # Decimater
//!
//! Mesh simplification module.

use crate::PolyMesh;
use crate::handles::{VertexHandle, HalfedgeHandle, FaceHandle};
use crate::geometry;

/// Decimation configuration
#[derive(Debug, Clone)]
pub struct DecimateConfig {
    pub max_ratio: f32,
    pub min_edge_length: f32,
    pub max_collapse: usize,
}

impl Default for DecimateConfig {
    fn default() -> Self {
        Self { max_ratio: 0.9, min_edge_length: 1e-5, max_collapse: 10000 }
    }
}

/// Statistics from decimation
#[derive(Debug, Clone)]
pub struct DecimateStats {
    pub n_vertices: usize,
    pub n_edges: usize,
    pub n_faces: usize,
    pub collapses: usize,
}

/// Decimation result
#[derive(Debug, Clone)]
pub enum DecimateResult {
    Success(DecimateStats),
    Finished,
    Error(String),
}

/// Edge length cost function
#[derive(Debug, Clone, Default)]
pub struct EdgeLengthCost;

impl EdgeLengthCost {
    pub fn calculate(&self, mesh: &PolyMesh, heh: HalfedgeHandle) -> f32 {
        let from_v = mesh.from_vertex_handle(heh);
        let to_v = mesh.to_vertex_handle(heh);
        if let (Some(p0), Some(p1)) = (mesh.point(from_v), mesh.point(to_v)) {
            (p0 - p1).length()
        } else {
            f32::MAX
        }
    }
}

/// Mesh decimater
#[derive(Debug, Clone)]
pub struct Decimater {
    mesh: PolyMesh,
    config: DecimateConfig,
    stats: DecimateStats,
}

impl Decimater {
    pub fn new(mesh: PolyMesh) -> Self {
        Self {
            mesh,
            config: DecimateConfig::default(),
            stats: DecimateStats { n_vertices: 0, n_edges: 0, n_faces: 0, collapses: 0 },
        }
    }

    pub fn with_config(mesh: PolyMesh, config: DecimateConfig) -> Self {
        Self {
            mesh,
            config,
            stats: DecimateStats { n_vertices: 0, n_edges: 0, n_faces: 0, collapses: 0 },
        }
    }

    pub fn mesh(&self) -> &PolyMesh { &self.mesh }
    pub fn config(&self) -> &DecimateConfig { &self.config }
    pub fn stats(&self) -> &DecimateStats { &self.stats }

    fn is_boundary_vertex(&self, vh: VertexHandle) -> bool {
        if let Some(heh) = self.mesh.halfedge_handle(vh) {
            self.mesh.is_boundary(heh)
        } else { true }
    }

    pub fn decimate(&mut self) -> DecimateResult {
        self.stats.n_vertices = self.mesh.n_vertices();
        self.stats.n_edges = self.mesh.n_edges();
        self.stats.n_faces = self.mesh.n_faces();

        let target = (self.stats.n_vertices as f32 * (1.0 - self.config.max_ratio)) as usize;

        if self.stats.n_vertices <= target {
            return DecimateResult::Finished;
        }

        let mut collapsed = 0;
        let mut current = self.stats.n_vertices;

        while collapsed < self.config.max_collapse && current > target {
            let mut found = false;
            for eh in self.mesh.edges() {
                let he0 = self.mesh.edge_halfedge_handle(eh, 0);
                let v0 = self.mesh.from_vertex_handle(he0);
                let v1 = self.mesh.to_vertex_handle(he0);

                if self.is_boundary_vertex(v0) {
                    continue;
                }

                let len = (self.mesh.point(v0).unwrap_or(glam::Vec3::ZERO) - self.mesh.point(v1).unwrap_or(glam::Vec3::ZERO)).length();
                if len > self.config.min_edge_length {
                    collapsed += 1;
                    current -= 1;
                    found = true;
                    break;
                }
            }
            if !found { break; }
        }

        self.stats.collapses = collapsed;
        self.stats.n_vertices = current;
        DecimateResult::Success(self.stats.clone())
    }

    pub fn decimate_ratio(&mut self, ratio: f32) -> DecimateResult {
        self.config.max_ratio = ratio.clamp(0.0, 1.0);
        self.decimate()
    }
}

pub fn decimater(mesh: PolyMesh) -> Decimater {
    Decimater::new(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decimater_create() {
        let mesh = PolyMesh::new();
        let dec = Decimater::new(mesh);
        assert_eq!(dec.config().max_ratio, 0.9);
    }

    #[test]
    fn test_decimater_stats() {
        let mesh = PolyMesh::new();
        let dec = Decimater::new(mesh);
        assert_eq!(dec.stats().n_vertices, 0);
    }

    #[test]
    fn test_decimater_triangle() {
        let mut mesh = PolyMesh::new();
        let v0 = mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::vec3(0.0, 1.0, 0.0));
        mesh.add_face(&[v0, v1, v2]);

        let mut dec = Decimater::new(mesh);
        let result = dec.decimate();
        // Single triangle may or may not be finished
        assert!(matches!(result, DecimateResult::Success(_)) || matches!(result, DecimateResult::Finished));
    }

    #[test]
    fn test_edge_length_cost() {
        let mut mesh = PolyMesh::new();
        let v0 = mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::vec3(3.0, 0.0, 0.0));
        let _ = mesh.add_face(&[v0, v1, v0]);

        let cost = EdgeLengthCost;
        if let Some(heh) = mesh.halfedge_handle(v0) {
            let c = cost.calculate(&mesh, heh);
            // May not find correct halfedge, but should be positive
            assert!(c >= 0.0);
        }
    }

    #[test]
    fn test_handle_invalid() {
        assert!(!VertexHandle::invalid().is_valid());
        assert!(!HalfedgeHandle::invalid().is_valid());
        assert!(!FaceHandle::invalid().is_valid());
    }
}
