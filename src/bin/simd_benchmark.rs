// SIMD Performance Benchmark

use std::time::Instant;
use std::hint::black_box;

fn main() {
    println!("========================================");
    println!("RustMesh SIMD Performance Benchmark");
    println!("========================================");

    // Generate test mesh
    let segments = 512;
    let rings = 512;

    println!("Generating SIMD test mesh ({}x{})...", segments, rings);
    
    let orig_mesh = rustmesh::generate_sphere(100.0, segments, rings);
    let n_vertices = orig_mesh.n_vertices();
    let n_faces = orig_mesh.n_faces();
    
    let mut hp_mesh = rustmesh::HighPerfMesh::new();
    for v in orig_mesh.vertices() {
        if let Some(p) = orig_mesh.point(v) {
            hp_mesh.add_vertex(p.x, p.y, p.z);
        }
    }
    
    println!("  Vertices: {}", n_vertices);
    println!("  Faces: {}", n_faces);

    let simd_mesh = rustmesh::generate_sphere_simd(100.0, segments, rings);
    let runs = 50;

    // Centroid
    let mut times1 = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let _ = hp_mesh.centroid();
        times1.push(start.elapsed().as_nanos() as f64);
    }
    let avg1 = times1.iter().sum::<f64>() / runs as f64;

    let mut times2 = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let _ = simd_mesh.centroid_simd();
        times2.push(start.elapsed().as_nanos() as f64);
    }
    let avg2 = times2.iter().sum::<f64>() / runs as f64;

    // Vertex Sum SIMD
    let mut times3 = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let mut sum_x = 0.0f32;
        let mut sum_y = 0.0f32;
        let mut sum_z = 0.0f32;
        
        let ptr_x = simd_mesh.x.as_ptr();
        let ptr_y = simd_mesh.y.as_ptr();
        let ptr_z = simd_mesh.z.as_ptr();
        let n = simd_mesh.n_vertices();
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
        
        #[cfg(not(target_arch = "aarch64"))]
        {
            while i < n {
                sum_x += simd_mesh.x[i];
                sum_y += simd_mesh.y[i];
                sum_z += simd_mesh.z[i];
                i += 1;
            }
        }
        
        black_box((sum_x, sum_y, sum_z));
        times3.push(start.elapsed().as_nanos() as f64);
    }
    let avg3 = times3.iter().sum::<f64>() / runs as f64;

    // Surface Area
    let mut times4 = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let _ = hp_mesh.surface_area();
        times4.push(start.elapsed().as_nanos() as f64);
    }
    let avg4 = times4.iter().sum::<f64>() / runs as f64;

    let mut times5 = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let _ = simd_mesh.surface_area_simd();
        times5.push(start.elapsed().as_nanos() as f64);
    }
    let avg5 = times5.iter().sum::<f64>() / runs as f64;

    println!("\n[1] Centroid Computation:");
    println!("  Scalar (pointer): {:.0} ns", avg1);
    println!("  SIMD (NEON):     {:.0} ns", avg2);
    if avg1 > 0.0 { println!("  SIMD speedup: {:.1}x", avg1 / avg2); }

    println!("\n[2] Vertex Sum (NEON SIMD):");
    println!("  {:.0} ns", avg3);

    println!("\n[3] Surface Area:");
    println!("  Scalar: {:.0} ns", avg4);
    println!("  SIMD:   {:.0} ns", avg5);
    if avg4 > 0.0 { println!("  SIMD speedup: {:.1}x", avg4 / avg5); }

    println!("\n========================================");
    println!("OpenMesh Comparison (extrapolated)");
    println!("========================================");

    let openmesh_centroid = 30000.0 / 1000.0 * n_vertices as f64;
    let openmesh_iter = 291.0 / 1000.0 * n_vertices as f64;

    println!("\n[1] Centroid:");
    println!("  OpenMesh (est): {:.0} ns", openmesh_centroid);
    println!("  RustMesh SIMD:  {:.0} ns", avg2);
    println!("  Gap: {:.1}x", avg2 / openmesh_centroid);

    println!("\n[2] Vertex Iteration:");
    println!("  OpenMesh: {:.0} ns", openmesh_iter);
    println!("  RustMesh SIMD: {:.0} ns", avg3);
    println!("  Gap: {:.1}x", avg3 / openmesh_iter);

    println!("\n========================================");
    println!("Summary");
    println!("========================================");

    let avg_openmesh = (openmesh_iter + openmesh_centroid) / 2.0;
    let avg_rust = (avg2 + avg3 + avg5) / 3.0;
    let overall_ratio = avg_rust / avg_openmesh;
    
    println!("\nOverall gap vs OpenMesh: {:.1}x", overall_ratio);

    if overall_ratio < 2.0 {
        println!("RustMesh SIMD is competitive!");
    } else {
        println!("Continue optimization (gap: {:.1}x)", overall_ratio);
    }
}
