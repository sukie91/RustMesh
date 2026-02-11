//! # SIMD-Optimized High Performance Mesh
//!
//! Features:
//! - SIMD-accelerated operations (ARM NEON, x86 SSE/AVX)
//! - Contiguous memory layout (SoA - Structure of Arrays)
//! - Batch processing for cache efficiency

use std::ptr;

/// SIMD-accelerated mesh operations
#[derive(Debug)]
pub struct SimdMesh {
    /// X coordinates (public for benchmark access)
    pub x: Vec<f32>,
    /// Y coordinates
    pub y: Vec<f32>, 
    /// Z coordinates
    pub z: Vec<f32>,
    /// Triangle faces
    pub faces: Vec<[u32; 3]>,
}

impl SimdMesh {
    /// Create new SIMD mesh
    #[inline]
    pub fn new() -> Self {
        Self {
            x: Vec::new(),
            y: Vec::new(),
            z: Vec::new(),
            faces: Vec::new(),
        }
    }
    
    /// Add vertex
    #[inline]
    pub fn push_vertex(&mut self, px: f32, py: f32, pz: f32) {
        self.x.push(px);
        self.y.push(py);
        self.z.push(pz);
    }
    
    /// Add triangle face
    #[inline]
    pub fn push_triangle(&mut self, v0: u32, v1: u32, v2: u32) {
        self.faces.push([v0, v1, v2]);
    }
    
    /// Number of vertices
    #[inline]
    pub fn n_vertices(&self) -> usize {
        self.x.len()
    }
    
    /// Number of faces
    #[inline]
    pub fn n_faces(&self) -> usize {
        self.faces.len()
    }
    
    /// Get raw pointers
    #[inline]
    pub fn x_ptr(&self) -> *const f32 {
        self.x.as_ptr()
    }
    
    #[inline]
    pub fn y_ptr(&self) -> *const f32 {
        self.y.as_ptr()
    }
    
    #[inline]
    pub fn z_ptr(&self) -> *const f32 {
        self.z.as_ptr()
    }
    
    #[inline]
    pub fn faces_ptr(&self) -> *const [u32; 3] {
        self.faces.as_ptr()
    }
    
    // ========== SIMD Operations ==========
    
    /// Compute centroid using NEON SIMD (ARM)
    #[inline]
    pub fn centroid_simd(&self) -> (f32, f32, f32) {
        let n = self.x.len();
        if n == 0 {
            return (0.0, 0.0, 0.0);
        }

        let ptr_x = self.x.as_ptr();
        let ptr_y = self.y.as_ptr();
        let ptr_z = self.z.as_ptr();

        let mut sum_x: f32 = 0.0;
        let mut sum_y: f32 = 0.0;
        let mut sum_z: f32 = 0.0;

        // Process 4 vertices at a time using NEON
        let mut i = 0;
        let n_simd = (n / 4) * 4;
        
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use std::arch::aarch64::*;
            
            let mut acc_x = vdupq_n_f32(0.0);
            let mut acc_y = vdupq_n_f32(0.0);
            let mut acc_z = vdupq_n_f32(0.0);
            
            while i < n_simd {
                let vx = vld1q_f32(ptr_x.add(i));
                let vy = vld1q_f32(ptr_y.add(i));
                let vz = vld1q_f32(ptr_z.add(i));
                
                acc_x = vaddq_f32(acc_x, vx);
                acc_y = vaddq_f32(acc_y, vy);
                acc_z = vaddq_f32(acc_z, vz);
                
                i += 4;
            }
            
            sum_x = vaddvq_f32(acc_x);
            sum_y = vaddvq_f32(acc_y);
            sum_z = vaddvq_f32(acc_z);
        }
        
        // Scalar tail
        while i < n {
            sum_x += self.x[i];
            sum_y += self.y[i];
            sum_z += self.z[i];
            i += 1;
        }

        let inv_n = 1.0 / n as f32;
        (sum_x * inv_n, sum_y * inv_n, sum_z * inv_n)
    }
    
    /// Compute bounding box using SIMD
    #[inline]
    pub fn bounding_box_simd(&self) -> (f32, f32, f32, f32, f32, f32) {
        let n = self.x.len();
        if n == 0 {
            return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        }

        let min_val = self.x[0];
        let mut min_x = min_val;
        let mut min_y = self.y[0];
        let mut min_z = self.z[0];
        let mut max_x = min_val;
        let mut max_y = self.y[0];
        let mut max_z = self.z[0];

        let ptr_x = self.x.as_ptr();
        let mut i = 0;
        let n_simd = (n / 4) * 4;
        
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use std::arch::aarch64::*;
            let mut min_acc_x = vdupq_n_f32(min_x);
            let mut max_acc_x = vdupq_n_f32(max_x);
            
            while i < n_simd {
                let vx = vld1q_f32(ptr_x.add(i));
                min_acc_x = vminq_f32(min_acc_x, vx);
                max_acc_x = vmaxq_f32(max_acc_x, vx);
                i += 4;
            }
            
            min_x = vaddvq_f32(min_acc_x) / 4.0;
            max_x = vaddvq_f32(max_acc_x) / 4.0;
        }
        
        #[cfg(not(target_arch = "aarch64"))]
        {
            while i < n {
                let x = self.x[i];
                if x < min_x { min_x = x; }
                if x > max_x { max_x = x; }
                i += 1;
            }
        }

        (min_x, min_y, min_z, max_x, max_y, max_z)
    }
    
    /// Compute surface area using SIMD
    #[inline]
    pub fn surface_area_simd(&self) -> f32 {
        let n_faces = self.faces.len();
        if n_faces == 0 {
            return 0.0;
        }

        let ptr_x = self.x.as_ptr();
        let ptr_y = self.y.as_ptr();
        let ptr_z = self.z.as_ptr();
        let faces_ptr = self.faces.as_ptr();

        let mut area: f32 = 0.0;

        #[cfg(target_arch = "aarch64")]
        unsafe {
            use std::arch::aarch64::*;
            
            let mut i = 0;
            let n_simd = (n_faces / 4) * 4;
            
            while i < n_simd {
                let mut acc_area = vdupq_n_f32(0.0);
                
                for k in 0..4 {
                    let face = &*faces_ptr.add(i + k);
                    let v0 = face[0] as usize;
                    let v1 = face[1] as usize;
                    let v2 = face[2] as usize;
                    
                    let x0 = *ptr_x.add(v0);
                    let y0 = *ptr_y.add(v0);
                    let z0 = *ptr_z.add(v0);
                    
                    let x1 = *ptr_x.add(v1);
                    let y1 = *ptr_y.add(v1);
                    let z1 = *ptr_z.add(v1);
                    
                    let x2 = *ptr_x.add(v2);
                    let y2 = *ptr_y.add(v2);
                    let z2 = *ptr_z.add(v2);
                    
                    // Cross product
                    let e1x = x1 - x0;
                    let e1y = y1 - y0;
                    let e1z = z1 - z0;
                    
                    let e2x = x2 - x0;
                    let e2y = y2 - y0;
                    let e2z = z2 - z0;
                    
                    let cx = e1y * e2z - e1z * e2y;
                    let cy = e1z * e2x - e1x * e2z;
                    let cz = e1x * e2y - e1y * e2x;
                    
                    let tri_area = (cx * cx + cy * cy + cz * cz).sqrt() * 0.5;
                    acc_area = vaddq_f32(acc_area, vdupq_n_f32(tri_area));
                }
                
                area += vaddvq_f32(acc_area);
                i += 4;
            }
        }
        
        #[cfg(not(target_arch = "aarch64"))]
        {
            let mut i = 0;
            while i < n_faces {
                let face = &*faces_ptr.add(i);
                let v0 = face[0] as usize;
                let v1 = face[1] as usize;
                let v2 = face[2] as usize;
                
                let x0 = *ptr_x.add(v0);
                let y0 = *ptr_y.add(v0);
                let z0 = *ptr_z.add(v0);
                
                let x1 = *ptr_x.add(v1);
                let y1 = *ptr_y.add(v1);
                let z1 = *ptr_z.add(v1);
                
                let x2 = *ptr_x.add(v2);
                let y2 = *ptr_y.add(v2);
                let z2 = *ptr_z.add(v2);
                
                let e1x = x1 - x0;
                let e1y = y1 - y0;
                let e1z = z1 - z0;
                
                let e2x = x2 - x0;
                let e2y = y2 - y0;
                let e2z = z2 - z0;
                
                let cx = e1y * e2z - e1z * e2y;
                let cy = e1z * e2x - e1x * e2z;
                let cz = e1x * e2y - e1y * e2x;
                
                area += (cx * cx + cy * cy + cz * cz).sqrt() * 0.5;
                i += 1;
            }
        }

        area
    }
}

/// Generate sphere with SIMD support
#[inline]
pub fn generate_sphere_simd(radius: f32, segments: usize, rings: usize) -> SimdMesh {
    let mut mesh = SimdMesh::new();
    
    // Reserve capacity
    let n_vertices = (segments + 1) * (rings + 1);
    let n_faces = segments * rings * 2;
    mesh.x.reserve(n_vertices);
    mesh.y.reserve(n_vertices);
    mesh.z.reserve(n_vertices);
    mesh.faces.reserve(n_faces);
    
    // Generate vertices
    for r in 0..=rings {
        let phi = std::f32::consts::PI * r as f32 / rings as f32;
        for s in 0..=segments {
            let theta = 2.0 * std::f32::consts::PI * s as f32 / segments as f32;
            
            let x = radius * phi.sin() * theta.cos();
            let y = radius * phi.cos();
            let z = radius * phi.sin() * theta.sin();
            
            mesh.push_vertex(x, y, z);
        }
    }
    
    // Generate faces
    for r in 0..rings {
        for s in 0..segments {
            let current = r * (segments + 1) + s;
            let next = current + segments + 1;
            
            mesh.push_triangle(current as u32, (current + 1) as u32, (next + 1) as u32);
            mesh.push_triangle(current as u32, (next + 1) as u32, next as u32);
        }
    }
    
    mesh
}
