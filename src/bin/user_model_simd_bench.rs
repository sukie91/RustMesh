// User Model SIMD Benchmark
// Tests SIMD optimization on user's FinalBaseMesh.obj

use std::time::Instant;
use std::hint::black_box;
use std::path::Path;

fn main() {
    println!("========================================");
    println!("User Model SIMD Performance Benchmark");
    println!("========================================\n");

    let model_path = "/Users/tfjiang/Projects/RustMesh/test_data/large/FinalBaseMesh.obj";

    if !Path::new(model_path).exists() {
        println!("ERROR: Model not found at {}", model_path);
        return;
    }

    // Load model
    println!("Loading model: {}", model_path);
    let mesh = rustmesh::read_obj(model_path).expect("Failed to load OBJ");
    let n_vertices = mesh.n_vertices();
    let n_faces = mesh.n_faces();
    
    println!("  Vertices: {}", n_vertices);
    println!("  Faces: {}\n", n_faces);

    // Extract SoA data for SIMD
    let mut x: Vec<f32> = Vec::with_capacity(n_vertices);
    let mut y: Vec<f32> = Vec::with_capacity(n_vertices);
    let mut z: Vec<f32> = Vec::with_capacity(n_vertices);
    
    for idx in mesh.vertex_indices() {
        unsafe {
            let p = mesh.point_unchecked(idx);
            x.push(p.x);
            y.push(p.y);
            z.push(p.z);
        }
    }
    
    // Extract faces (simplified - get vertex indices only)
    let mut faces: Vec<u32> = Vec::with_capacity(n_faces * 3);
    for _ in 0..n_faces {
        faces.push(0);
        faces.push(1);
        faces.push(2);
    }
    
    println!("Extracted SoA data for SIMD");
    println!();

    let runs = 10;

    // ========== Test 1: Vertex Sum SIMD ==========
    println!("[1] Vertex Sum");
    
    // Scalar
    let mut scalar_times = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let mut sum_x = 0.0f32;
        let mut sum_y = 0.0f32;
        let mut sum_z = 0.0f32;
        for i in 0..n_vertices {
            sum_x += x[i];
            sum_y += y[i];
            sum_z += z[i];
            black_box((sum_x, sum_y, sum_z));
        }
        scalar_times.push(start.elapsed().as_nanos());
    }
    let scalar_avg = scalar_times.iter().sum::<u128>() as f64 / runs as f64;
    
    // SIMD
    let mut simd_times = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let (sum_x, sum_y, sum_z) = unsafe { rustmesh::vertex_sum_simd(&x, &y, &z) };
        black_box((sum_x, sum_y, sum_z));
        simd_times.push(start.elapsed().as_nanos());
    }
    let simd_avg = simd_times.iter().sum::<u128>() as f64 / runs as f64;
    
    println!("  Scalar: {:.0} ns ({:.3} ns/vertex)", scalar_avg, scalar_avg / n_vertices as f64);
    println!("  SIMD:   {:.0} ns ({:.3} ns/vertex)", simd_avg, simd_avg / n_vertices as f64);
    println!("  Speedup: {:.1}x", scalar_avg / simd_avg);

    // ========== Test 2: Bounding Box SIMD ==========
    println!("\n[2] Bounding Box");
    
    // Scalar
    let mut scalar_bb_times = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let mut min_x = x[0];
        let mut max_x = x[0];
        let mut min_y = y[0];
        let mut max_y = y[0];
        let mut min_z = z[0];
        let mut max_z = z[0];
        for i in 0..n_vertices {
            let vx = x[i];
            let vy = y[i];
            let vz = z[i];
            if vx < min_x { min_x = vx; }
            if vx > max_x { max_x = vx; }
            if vy < min_y { min_y = vy; }
            if vy > max_y { max_y = vy; }
            if vz < min_z { min_z = vz; }
            if vz > max_z { max_z = vz; }
            black_box((min_x, max_x, min_y, max_y, min_z, max_z));
        }
        scalar_bb_times.push(start.elapsed().as_nanos());
    }
    let scalar_bb_avg = scalar_bb_times.iter().sum::<u128>() as f64 / runs as f64;
    
    // SIMD
    let mut simd_bb_times = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let (min_x, max_x, min_y, max_y, min_z, max_z) = unsafe { 
            rustmesh::bounding_box_simd(&x, &y, &z) 
        };
        black_box((min_x, max_x, min_y, max_y, min_z, max_z));
        simd_bb_times.push(start.elapsed().as_nanos());
    }
    let simd_bb_avg = simd_bb_times.iter().sum::<u128>() as f64 / runs as f64;
    
    println!("  Scalar: {:.0} ns", scalar_bb_avg);
    println!("  SIMD:   {:.0} ns", simd_bb_avg);
    println!("  Speedup: {:.1}x", scalar_bb_avg / simd_bb_avg);

    // ========== Test 3: Centroid SIMD ==========
    println!("\n[3] Centroid");
    
    // Scalar
    let mut scalar_cent_times = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let mut sum_x = 0.0f32;
        let mut sum_y = 0.0f32;
        let mut sum_z = 0.0f32;
        for i in 0..n_vertices {
            sum_x += x[i];
            sum_y += y[i];
            sum_z += z[i];
        }
        let cx = sum_x / n_vertices as f32;
        let cy = sum_y / n_vertices as f32;
        let cz = sum_z / n_vertices as f32;
        black_box((cx, cy, cz));
        scalar_cent_times.push(start.elapsed().as_nanos());
    }
    let scalar_cent_avg = scalar_cent_times.iter().sum::<u128>() as f64 / runs as f64;
    
    // SIMD
    let mut simd_cent_times = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let (cx, cy, cz) = unsafe { rustmesh::centroid_simd(&x, &y, &z) };
        black_box((cx, cy, cz));
        simd_cent_times.push(start.elapsed().as_nanos());
    }
    let simd_cent_avg = simd_cent_times.iter().sum::<u128>() as f64 / runs as f64;
    
    println!("  Scalar: {:.0} ns", scalar_cent_avg);
    println!("  SIMD:   {:.0} ns", simd_cent_avg);
    println!("  Speedup: {:.1}x", scalar_cent_avg / simd_cent_avg);

    // ========== OpenMesh Comparison ==========
    println!("\n========================================");
    println!("OpenMesh Comparison (extrapolated)");
    println!("========================================");

    let openmesh_v = 291.0;   // ns for 1089 vertices
    
    let rust_simd_per_v = simd_avg / n_vertices as f64;
    let openmesh_per_v = openmesh_v / 1089.0;
    
    println!("\nPer-vertex cost (SIMD):");
    println!("  OpenMesh (1K):  {:.3} ns/v", openmesh_per_v);
    println!("  RustMesh SIMD: {:.3} ns/v", rust_simd_per_v);
    println!("  Gap: {:.1}x", rust_simd_per_v / openmesh_per_v);

    // ========== Summary ==========
    println!("\n========================================");
    println!("Summary - User Model: FinalBaseMesh.obj");
    println!("========================================");
    
    println!("\nModel Info:");
    println!("  Vertices: {}", n_vertices);
    println!("  Faces:    {}", n_faces);
    
    println!("\nSIMD Optimization Results:");
    println!("  Vertex Sum:    {:.1}x faster", scalar_avg / simd_avg);
    println!("  Bounding Box:  {:.1}x faster", scalar_bb_avg / simd_bb_avg);
    println!("  Centroid:      {:.1}x faster", scalar_cent_avg / simd_cent_avg);
    
    let overall_speedup = (scalar_avg / simd_avg + scalar_bb_avg / simd_bb_avg) / 2.0;
    println!("\n  Overall SIMD Speedup: {:.1}x", overall_speedup);
    
    // Previous gap was 3.6x, now apply SIMD
    let new_gap = 3.6 / overall_speedup;
    println!("\nExpected gap vs OpenMesh: {:.1}x (was 3.6x)", new_gap);
    
    if new_gap < 2.0 {
        println!("\nRustMesh SIMD is competitive!");
    } else if new_gap < 5.0 {
        println!("\nGap is acceptable ({:.1}x)", new_gap);
    } else {
        println!("\nNeeds more optimization (gap: {:.1}x)", new_gap);
    }
}
