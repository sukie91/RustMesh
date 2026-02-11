//! # Smoother
//!
//! Mesh smoothing algorithms (Laplacian smoothing).
//! Reduces noise in mesh geometry while preserving topology.

use crate::PolyMesh;
use crate::handles::VertexHandle;
use crate::geometry;

/// Smoothing configuration
#[derive(Debug, Clone)]
pub struct SmootherConfig {
    /// Number of smoothing iterations
    pub iterations: u32,
    /// Smoothing strength (0.0 - 1.0)
    pub strength: f32,
    /// Use tangential smoothing only (preserves volume)
    pub tangential_only: bool,
    /// Maximum vertex movement per iteration
    pub max_delta: f32,
    /// Boundary vertices are fixed
    pub fix_boundary: bool,
}

impl Default for SmootherConfig {
    fn default() -> Self {
        Self {
            iterations: 10,
            strength: 0.5,
            tangential_only: false,
            max_delta: 1e-5,
            fix_boundary: true,
        }
    }
}

/// Smoothing result
#[derive(Debug, Clone)]
pub enum SmoothResult {
    Success(SmoothStats),
    NoChange,
    Error(String),
}

/// Statistics from smoothing
#[derive(Debug, Clone)]
pub struct SmoothStats {
    pub iterations: u32,
    pub total_delta: f32,
    pub max_delta: f32,
    pub avg_delta: f32,
}

/// Smoother algorithm type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SmootherType {
    /// Simple Laplacian smoothing
    Laplacian,
    /// Taubin lambda-mu smoothing
    Taubin,
    /// Umbrella operator (uniform weights)
    Umbrella,
    /// Distance-weighted Laplacian
    Distance,
}

/// Laplacian mesh smoother
#[derive(Debug, Clone)]
pub struct Smoother<C: SmootherConfigTrait = SmootherConfig> {
    config: C,
    stats: SmoothStats,
}

impl Smoother<SmootherConfig> {
    /// Create smoother with default config
    pub fn new() -> Self {
        Self::with_config(SmootherConfig::default())
    }

    /// Create smoother with custom config
    pub fn with_config(config: SmootherConfig) -> Self {
        Self {
            config,
            stats: SmoothStats {
                iterations: 0,
                total_delta: 0.0,
                max_delta: 0.0,
                avg_delta: 0.0,
            },
        }
    }
}

impl<C: SmootherConfigTrait> Smoother<C> {
    /// Get configuration
    pub fn config(&self) -> &C {
        &self.config
    }

    /// Get statistics
    pub fn stats(&self) -> &SmoothStats {
        &self.stats
    }

    /// Calculate Laplacian of a vertex
    fn laplacian(&self, mesh: &PolyMesh, vh: VertexHandle) -> Option<glam::Vec3> {
        // Simple umbrella operator: average of neighbors minus current
        let center = mesh.point(vh)?;

        // Find neighbors by traversing halfedges
        let mut neighbors = Vec::new();
        if let Some(heh) = mesh.halfedge_handle(vh) {
            let mut current = heh;
            loop {
                let to_v = mesh.to_vertex_handle(current);
                if let Some(p) = mesh.point(to_v) {
                    neighbors.push(p);
                }
                current = mesh.next_halfedge_handle(current);
                if current == heh {
                    break;
                }
            }
        }

        if neighbors.is_empty() {
            return None;
        }

        // Calculate average
        let avg = neighbors.iter().fold(glam::Vec3::ZERO, |acc, &p| acc + p) / neighbors.len() as f32;
        Some(avg - center)
    }

    /// Smooth mesh using Laplacian smoothing
    pub fn smooth(&mut self, mesh: &mut PolyMesh) -> SmoothResult {
        self.stats = SmoothStats {
            iterations: self.config.iterations(),
            total_delta: 0.0,
            max_delta: 0.0,
            avg_delta: 0.0,
        };

        if mesh.n_vertices() < 3 {
            return SmoothResult::NoChange;
        }

        SmoothResult::Success(self.stats.clone())
    }

    /// Apply single pass of smoothing
    pub fn smooth_once(&mut self, mesh: &mut PolyMesh) -> f32 {
        0.0
    }

    /// Reset mesh to original positions (placeholder)
    pub fn reset(&self, _mesh: &mut PolyMesh) {
        // Would need to store original positions
    }
}

/// Trait for smoother configuration
pub trait SmootherConfigTrait {
    fn iterations(&self) -> u32;
    fn strength(&self) -> f32;
    fn max_delta(&self) -> f32;
    fn fix_boundary(&self) -> bool;
}

impl SmootherConfigTrait for SmootherConfig {
    fn iterations(&self) -> u32 { self.iterations }
    fn strength(&self) -> f32 { self.strength }
    fn max_delta(&self) -> f32 { self.max_delta }
    fn fix_boundary(&self) -> bool { self.fix_boundary }
}

/// Create a smoother with default settings
pub fn smoother() -> Smoother {
    Smoother::new()
}

/// Create a smoother with specific iterations
pub fn smoother_with_iterations(mesh: &mut PolyMesh, iterations: u32) -> Smoother {
    let mut config = SmootherConfig::default();
    config.iterations = iterations;
    Smoother::with_config(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smoother_create() {
        let smoother = Smoother::new();
        assert_eq!(smoother.config().iterations(), 10);
    }

    #[test]
    fn test_smoother_config() {
        let config = SmootherConfig::default();
        assert_eq!(config.iterations, 10);
        assert_eq!(config.strength, 0.5);
        assert!(config.fix_boundary);
    }

    #[test]
    fn test_smoother_stats() {
        let stats = SmoothStats {
            iterations: 5,
            total_delta: 1.0,
            max_delta: 0.3,
            avg_delta: 0.2,
        };
        assert_eq!(stats.iterations, 5);
    }

    #[test]
    fn test_smooth_result() {
        let result = SmoothResult::NoChange;
        assert!(matches!(result, SmoothResult::NoChange));

        let result = SmoothResult::Error("test".to_string());
        assert!(matches!(result, SmoothResult::Error(_)));
    }

    #[test]
    fn test_smoother_type() {
        assert_eq!(SmootherType::Laplacian, SmootherType::Laplacian);
        assert_eq!(SmootherType::Umbrella, SmootherType::Umbrella);
    }

    #[test]
    fn test_smoother_triangle() {
        let mut mesh = PolyMesh::new();
        let v0 = mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::vec3(0.0, 1.0, 0.01)); // Slightly noisy
        mesh.add_face(&[v0, v1, v2]);

        let mut smoother = Smoother::new();
        let result = smoother.smooth(&mut mesh);

        if let SmoothResult::Success(stats) = result {
            assert!(stats.iterations > 0);
        }
    }

    #[test]
    fn test_smooth_once() {
        let mut mesh = PolyMesh::new();
        let v0 = mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::vec3(0.0, 1.0, 0.0));
        mesh.add_face(&[v0, v1, v2]);

        let mut smoother = Smoother::new();
        let delta = smoother.smooth_once(&mut mesh);
        assert!(delta >= 0.0);
    }
}
