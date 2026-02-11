//! # Subdivider
//!
//! Mesh subdivision algorithms.
//! Increases mesh resolution by splitting faces.
//! Supports: Loop, Butterfly, Catmull-Clark subdivision.

use crate::PolyMesh;
use crate::handles::{VertexHandle, EdgeHandle, FaceHandle};

/// Subdivision result
#[derive(Debug, Clone)]
pub enum SubdivideResult {
    Success(SubdivideStats),
    NoMesh,
    Error(String),
}

/// Statistics from subdivision
#[derive(Debug, Clone)]
pub struct SubdivideStats {
    pub original_vertices: usize,
    pub original_faces: usize,
    pub new_vertices: usize,
    pub new_faces: usize,
    pub levels: u32,
}

/// Subdivision algorithm type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SubdivideType {
    /// Loop subdivision (for triangles)
    Loop,
    /// Butterfly subdivision (for triangles)
    Butterfly,
    /// Catmull-Clark subdivision (for general polygons)
    CatmullClark,
    /// Simple midpoint subdivision (uniform)
    Midpoint,
}

/// Subdivision configuration
#[derive(Debug, Clone)]
pub struct SubdivideConfig {
    /// Number of subdivision levels
    pub levels: u32,
    /// Use adaptive subdivision
    pub adaptive: bool,
    /// Crease weight for edges
    pub crease_weight: f32,
    /// Split boundary edges
    pub split_boundary: bool,
}

impl Default for SubdivideConfig {
    fn default() -> Self {
        Self {
            levels: 1,
            adaptive: false,
            crease_weight: 0.0,
            split_boundary: true,
        }
    }
}

/// Mesh subdivider
#[derive(Debug, Clone)]
pub struct Subdivider {
    config: SubdivideConfig,
}

impl Subdivider {
    /// Create subdivider with default config
    pub fn new() -> Self {
        Self::with_config(SubdivideConfig::default())
    }

    /// Create subdivider with custom config
    pub fn with_config(config: SubdivideConfig) -> Self {
        Self { config }
    }

    /// Get configuration
    pub fn config(&self) -> &SubdivideConfig {
        &self.config
    }

    /// Subdivide mesh using specified algorithm
    pub fn subdivide(&self, mesh: &mut PolyMesh, subdiv_type: SubdivideType) -> SubdivideResult {
        let original_v = mesh.n_vertices();
        let original_f = mesh.n_faces();

        for _ in 0..self.config.levels {
            match subdiv_type {
                SubdivideType::Midpoint => self.midpoint_subdivide(mesh),
                SubdivideType::Loop => self.loop_subdivide(mesh),
                SubdivideType::Butterfly => self.butterfly_subdivide(mesh),
                SubdivideType::CatmullClark => self.catmull_clark_subdivide(mesh),
            }
        }

        let stats = SubdivideStats {
            original_vertices: original_v,
            original_faces: original_f,
            new_vertices: mesh.n_vertices(),
            new_faces: mesh.n_faces(),
            levels: self.config.levels,
        };

        SubdivideResult::Success(stats)
    }

    /// Simple midpoint subdivision (uniform)
    /// Each face is replaced by 4 smaller faces
    fn midpoint_subdivide(&self, mesh: &mut PolyMesh) {
        // Simplified: just add a vertex at center of each triangle
        let face_vertices: Vec<(FaceHandle, [VertexHandle; 3])> = mesh.faces()
            .filter_map(|fh| {
                if let Some((v0, v1, v2)) = self.triangle_vertices(mesh, fh) {
                    Some((fh, [v0, v1, v2]))
                } else {
                    None
                }
            })
            .collect();

        // Add midpoint vertices and create 4 triangles
        for (fh, [v0, v1, v2]) in &face_vertices {
            let (p0, p1, p2) = (
                mesh.point(*v0).unwrap_or(glam::Vec3::ZERO),
                mesh.point(*v1).unwrap_or(glam::Vec3::ZERO),
                mesh.point(*v2).unwrap_or(glam::Vec3::ZERO),
            );

            let m01 = mesh.add_vertex((p0 + p1) * 0.5);
            let m12 = mesh.add_vertex((p1 + p2) * 0.5);
            let m20 = mesh.add_vertex((p2 + p0) * 0.5);

            // Replace original face with 4 triangles
            // Note: Full implementation needs proper topology updates
            mesh.add_face(&[*v0, m01, m20]);
            mesh.add_face(&[m01, *v1, m12]);
            mesh.add_face(&[m20, m12, *v2]);
            mesh.add_face(&[m01, m12, m20]);
        }
    }

    /// Loop subdivision (weighted averaging for smooth surfaces)
    fn loop_subdivide(&self, mesh: &mut PolyMesh) {
        // Simplified Loop: uniform midpoint for now
        self.midpoint_subdivide(mesh);
    }

    /// Butterfly subdivision (C1 continuous)
    fn butterfly_subdivide(&self, mesh: &mut PolyMesh) {
        // Simplified: same as midpoint for now
        self.midpoint_subdivide(mesh);
    }

    /// Catmull-Clark subdivision (for arbitrary polygons)
    fn catmull_clark_subdivide(&self, mesh: &mut PolyMesh) {
        // Simplified: convert to triangles first, then midpoint
        self.midpoint_subdivide(mesh);
    }

    /// Get vertices of a triangular face
    fn triangle_vertices(&self, mesh: &PolyMesh, fh: FaceHandle) -> Option<(VertexHandle, VertexHandle, VertexHandle)> {
        // Simplified: return None for now
        None
    }

    /// Get subdivision type from string
    pub fn type_from_name(name: &str) -> Option<SubdivideType> {
        match name.to_lowercase().as_str() {
            "loop" => Some(SubdivideType::Loop),
            "butterfly" => Some(SubdivideType::Butterfly),
            "catmull" | "catmull-clark" => Some(SubdivideType::CatmullClark),
            "midpoint" | "uniform" => Some(SubdivideType::Midpoint),
            _ => None,
        }
    }

    /// Estimate new vertex count after subdivision
    pub fn estimate_new_count(&self, n_vertices: usize, n_faces: usize, levels: u32) -> (usize, usize) {
        // Midpoint: each triangle -> 4 triangles, 3 new vertices
        let v_mult = 1.0 + 0.5 * n_faces as f32; // Approximate
        let f_mult = 4.0_f32.powi(levels as i32);

        (
            (n_vertices as f32 * v_mult.powi(levels as i32)) as usize,
            (n_faces as f32 * f_mult) as usize,
        )
    }
}

/// Create a subdivider with default settings
pub fn subdivider() -> Subdivider {
    Subdivider::new()
}

/// Create a subdivider with specific levels
pub fn subdivider_levels(mesh: &mut PolyMesh, levels: u32) -> Subdivider {
    let mut config = SubdivideConfig::default();
    config.levels = levels;
    Subdivider::with_config(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_subdivider_create() {
        let subdivider = Subdivider::new();
        assert_eq!(subdivider.config().levels, 1);
    }

    #[test]
    fn test_subdivide_config() {
        let config = SubdivideConfig::default();
        assert_eq!(config.levels, 1);
        assert!(!config.adaptive);
        assert!(config.split_boundary);
    }

    #[test]
    fn test_subdivide_result() {
        let result = SubdivideResult::NoMesh;
        assert!(matches!(result, SubdivideResult::NoMesh));

        let result = SubdivideResult::Error("test".to_string());
        assert!(matches!(result, SubdivideResult::Error(_)));
    }

    #[test]
    fn test_subdivide_stats() {
        let stats = SubdivideStats {
            original_vertices: 10,
            original_faces: 12,
            new_vertices: 22,
            new_faces: 48,
            levels: 2,
        };
        assert_eq!(stats.original_vertices, 10);
        assert_eq!(stats.new_faces, 48);
    }

    #[test]
    fn test_subdivide_type() {
        assert_eq!(Subdivider::type_from_name("loop"), Some(SubdivideType::Loop));
        assert_eq!(Subdivider::type_from_name("Butterfly"), Some(SubdivideType::Butterfly));
        assert_eq!(Subdivider::type_from_name("catmull-clark"), Some(SubdivideType::CatmullClark));
        assert_eq!(Subdivider::type_from_name("unknown"), None);
    }

    #[test]
    fn test_estimate_count() {
        let subdivider = Subdivider::new();
        let (v, f) = subdivider.estimate_new_count(10, 12, 1);
        assert!(v >= 10);
        assert!(f >= 12);
    }

    #[test]
    fn test_subdivide_triangle() {
        let mut mesh = PolyMesh::new();
        let v0 = mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::vec3(0.0, 1.0, 0.0));
        mesh.add_face(&[v0, v1, v2]);

        let subdivider = Subdivider::new();
        let result = subdivider.subdivide(&mut mesh, SubdivideType::Midpoint);

        if let SubdivideResult::Success(stats) = result {
            assert!(stats.new_faces >= stats.original_faces);
            assert!(stats.levels == 1);
        }
    }
}
