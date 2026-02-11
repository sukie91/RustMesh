//! # High-Performance Mesh Data Structure
//!
//! This module provides a zero-overhead mesh structure that matches C++ performance.
//! Key optimizations:
//! - Contiguous memory layout (SoA - Structure of Arrays)
//! - No heap allocation per element
//! - Raw pointer access for maximum performance
//! - SIMD-friendly data layout

use std::ptr;

/// Structure of Arrays (SoA) layout for maximum cache efficiency
#[derive(Debug, Clone)]
pub struct HighPerfMesh {
    // Vertex data - separate arrays for better SIMD
    x: Vec<f32>,
    y: Vec<f32>,
    z: Vec<f32>,
    
    // Connectivity
    faces: Vec<[u32; 3]>,  // Triangular faces
    
    // Stats
    n_vertices: usize,
    n_faces: usize,
}

impl HighPerfMesh {
    /// Create a new empty mesh
    #[inline]
    pub fn new() -> Self {
        Self {
            x: Vec::new(),
            y: Vec::new(),
            z: Vec::new(),
            faces: Vec::new(),
            n_vertices: 0,
            n_faces: 0,
        }
    }

    /// Reserve capacity
    #[inline]
    pub fn reserve(&mut self, n_vertices: usize, n_faces: usize) {
        self.x.reserve(n_vertices);
        self.y.reserve(n_vertices);
        self.z.reserve(n_vertices);
        self.faces.reserve(n_faces);
    }

    /// Add a vertex
    #[inline]
    pub fn add_vertex(&mut self, px: f32, py: f32, pz: f32) -> u32 {
        let idx = self.n_vertices as u32;
        self.x.push(px);
        self.y.push(py);
        self.z.push(pz);
        self.n_vertices += 1;
        idx
    }

    /// Add a triangular face
    #[inline]
    pub fn add_triangle(&mut self, v0: u32, v1: u32, v2: u32) -> u32 {
        let idx = self.n_faces as u32;
        self.faces.push([v0, v1, v2]);
        self.n_faces += 1;
        idx
    }

    /// Get vertex count
    #[inline]
    pub fn n_vertices(&self) -> usize {
        self.n_vertices
    }

    /// Get face count
    #[inline]
    pub fn n_faces(&self) -> usize {
        self.n_faces
    }

    // ========== Zero-overhead iteration methods ==========

    /// Iterate vertices by index (fastest)
    #[inline]
    pub fn vertex_indices(&self) -> VertexIndexIter<'_> {
        VertexIndexIter::new(self)
    }

    /// Iterate faces by index (fastest)
    #[inline]
    pub fn face_indices(&self) -> FaceIndexIter<'_> {
        FaceIndexIter::new(self)
    }

    /// Get vertex as raw pointer (for bulk processing)
    #[inline]
    pub fn vertex_ptr(&self) -> *const f32 {
        self.x.as_ptr()
    }

    /// Get x coordinates
    #[inline]
    pub fn x_ptr(&self) -> *const f32 {
        self.x.as_ptr()
    }

    /// Get y coordinates
    #[inline]
    pub fn y_ptr(&self) -> *const f32 {
        self.y.as_ptr()
    }

    /// Get z coordinates
    #[inline]
    pub fn z_ptr(&self) -> *const f32 {
        self.z.as_ptr()
    }

    /// Get face data
    #[inline]
    pub fn faces_ptr(&self) -> *const [u32; 3] {
        self.faces.as_ptr()
    }

    /// Compute bounding box
    #[inline]
    pub fn bounding_box(&self) -> (f32, f32, f32, f32, f32, f32) {
        if self.n_vertices == 0 {
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        }

        let mut min_x = self.x[0];
        let mut min_y = self.y[0];
        let mut min_z = self.z[0];
        let mut max_x = min_x;
        let mut max_y = min_y;
        let mut max_z = min_z;

        for i in 0..self.n_vertices {
            let x = self.x[i];
            let y = self.y[i];
            let z = self.z[i];

            if x < min_x { min_x = x; }
            if x > max_x { max_x = x; }
            if y < min_y { min_y = y; }
            if y > max_y { max_y = y; }
            if z < min_z { min_z = z; }
            if z > max_z { max_z = z; }
        }

        (min_x, min_y, min_z, max_x, max_y, max_z)
    }

    /// Compute centroid
    #[inline]
    pub fn centroid(&self) -> (f32, f32, f32) {
        if self.n_vertices == 0 {
            return (0.0, 0.0, 0.0);
        }

        let mut sum_x = 0.0f32;
        let mut sum_y = 0.0f32;
        let mut sum_z = 0.0f32;

        // Use pointer arithmetic for speed
        let px = self.x.as_ptr();
        let py = self.y.as_ptr();
        let pz = self.z.as_ptr();

        for i in 0..self.n_vertices {
            unsafe {
                sum_x += *px.add(i);
                sum_y += *py.add(i);
                sum_z += *pz.add(i);
            }
        }

        let n = self.n_vertices as f32;
        (sum_x / n, sum_y / n, sum_z / n)
    }

    /// Compute surface area (sum of triangle areas)
    #[inline]
    pub fn surface_area(&self) -> f32 {
        let mut area = 0.0f32;
        let faces_ptr = self.faces.as_ptr();
        let px = self.x.as_ptr();
        let py = self.y.as_ptr();
        let pz = self.z.as_ptr();

        for i in 0..self.n_faces {
            unsafe {
                let f = &*faces_ptr.add(i);
                let v0 = f[0] as usize;
                let v1 = f[1] as usize;
                let v2 = f[2] as usize;

                // Triangle vertices
                let x0 = *px.add(v0);
                let y0 = *py.add(v0);
                let z0 = *pz.add(v0);

                let x1 = *px.add(v1);
                let y1 = *py.add(v1);
                let z1 = *pz.add(v1);

                let x2 = *px.add(v2);
                let y2 = *py.add(v2);
                let z2 = *pz.add(v2);

                // Edge vectors
                let e1x = x1 - x0;
                let e1y = y1 - y0;
                let e1z = z1 - z0;

                let e2x = x2 - x0;
                let e2y = y2 - y0;
                let e2z = z2 - z0;

                // Cross product
                let cx = e1y * e2z - e1z * e2y;
                let cy = e1z * e2x - e1x * e2z;
                let cz = e1x * e2y - e1y * e2x;

                // Area
                area += (cx * cx + cy * cy + cz * cz).sqrt() * 0.5;
            }
        }

        area
    }
}

/// Vertex index iterator
#[derive(Debug)]
pub struct VertexIndexIter<'a> {
    mesh: &'a HighPerfMesh,
    current: usize,
    end: usize,
}

impl<'a> VertexIndexIter<'a> {
    #[inline]
    pub fn new(mesh: &'a HighPerfMesh) -> Self {
        Self {
            mesh,
            current: 0,
            end: mesh.n_vertices(),
        }
    }
}

impl<'a> Iterator for VertexIndexIter<'a> {
    type Item = u32;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.end {
            let idx = self.current as u32;
            self.current += 1;
            Some(idx)
        } else {
            None
        }
    }
}

/// Face index iterator
#[derive(Debug)]
pub struct FaceIndexIter<'a> {
    mesh: &'a HighPerfMesh,
    current: usize,
    end: usize,
}

impl<'a> FaceIndexIter<'a> {
    #[inline]
    pub fn new(mesh: &'a HighPerfMesh) -> Self {
        Self {
            mesh,
            current: 0,
            end: mesh.n_faces(),
        }
    }
}

impl<'a> Iterator for FaceIndexIter<'a> {
    type Item = u32;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current < self.end {
            let idx = self.current as u32;
            self.current += 1;
            Some(idx)
        } else {
            None
        }
    }
}

/// Performance benchmark helper
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub name: String,
    pub iterations: usize,
    pub total_ns: f64,
    pub per_item_ns: f64,
}

impl BenchmarkResult {
    #[inline]
    pub fn new(name: String, iterations: usize, total_ns: f64, n_items: usize) -> Self {
        Self {
            name,
            iterations,
            total_ns,
            per_item_ns: total_ns / n_items as f64,
        }
    }

    #[inline]
    pub fn mps(&self) -> f64 {
        self.iterations as f64 / (self.total_ns / 1_000_000.0)
    }
}

/// High-performance utilities
pub mod hperf {
    use super::*;

    /// Generate a sphere mesh
    pub fn generate_sphere(radius: f32, segments: usize, rings: usize) -> HighPerfMesh {
        let mut mesh = HighPerfMesh::new();
        mesh.reserve((segments + 1) * (rings + 1), segments * rings * 2);

        // Generate vertices
        for r in 0..=rings {
            let phi = std::f32::consts::PI * r as f32 / rings as f32;
            for s in 0..=segments {
                let theta = 2.0 * std::f32::consts::PI * s as f32 / segments as f32;

                let x = radius * phi.sin() * theta.cos();
                let y = radius * phi.cos();
                let z = radius * phi.sin() * theta.sin();

                mesh.add_vertex(x, y, z);
            }
        }

        // Generate faces
        for r in 0..rings {
            for s in 0..segments {
                let current = r * (segments + 1) + s;
                let next = current + segments + 1;

                mesh.add_triangle(
                    current as u32,
                    (current + 1) as u32,
                    (next + 1) as u32,
                );
                mesh.add_triangle(
                    current as u32,
                    (next + 1) as u32,
                    next as u32,
                );
            }
        }

        mesh
    }

    /// Benchmark vertex iteration
    pub fn benchmark_vertex_iteration(mesh: &HighPerfMesh, runs: usize) -> BenchmarkResult {
        let mut sum = 0.0f32;
        let start = std::time::Instant::now();

        for _ in 0..runs {
            for i in 0..mesh.n_vertices() {
                unsafe {
                    sum += *mesh.x_ptr().add(i);
                }
            }
        }

        let total_ns = start.elapsed().as_nanos() as f64;
        BenchmarkResult::new(
            "Vertex Iteration".to_string(),
            runs * mesh.n_vertices(),
            total_ns,
            mesh.n_vertices(),
        )
    }

    /// Benchmark bounding box computation
    pub fn benchmark_bounding_box(mesh: &HighPerfMesh, runs: usize) -> BenchmarkResult {
        let start = std::time::Instant::now();

        for _ in 0..runs {
            let _ = mesh.bounding_box();
        }

        let total_ns = start.elapsed().as_nanos() as f64;
        BenchmarkResult::new(
            "Bounding Box".to_string(),
            runs,
            total_ns,
            mesh.n_vertices(),
        )
    }

    /// Benchmark centroid computation
    pub fn benchmark_centroid(mesh: &HighPerfMesh, runs: usize) -> BenchmarkResult {
        let start = std::time::Instant::now();

        for _ in 0..runs {
            let _ = mesh.centroid();
        }

        let total_ns = start.elapsed().as_nanos() as f64;
        BenchmarkResult::new(
            "Centroid".to_string(),
            runs,
            total_ns,
            mesh.n_vertices(),
        )
    }

    /// Benchmark surface area
    pub fn benchmark_surface_area(mesh: &HighPerfMesh, runs: usize) -> BenchmarkResult {
        let start = std::time::Instant::now();

        for _ in 0..runs {
            let _ = mesh.surface_area();
        }

        let total_ns = start.elapsed().as_nanos() as f64;
        BenchmarkResult::new(
            "Surface Area".to_string(),
            runs,
            total_ns,
            mesh.n_faces(),
        )
    }
}
