// Optimized Performance Benchmark

use std::time::Instant;
use std::hint::black_box;

fn main() {
    println!("========================================");
    println!("RustMesh Optimized Performance Benchmark");
    println!("========================================\n");

    let segments = 512;
    let rings = 512;
    
    println!("Generating mesh ({}x{})...", segments, rings);
    let mesh = rustmesh::generate_sphere(100.0, segments, rings);
    let n_vertices = mesh.n_vertices();
    let n_faces = mesh.n_faces();
    
    println!("  Vertices: {}", n_vertices);
    println!("  Faces: {}\n", n_faces);
    
    let runs = 10;
    
    // ========== Test 1: Standard API (baseline) ==========
    println!("[1] Standard API (VertexHandle + Option)");
    
    let mut times1: Vec<u128> = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let mut total = 0.0f32;
        for v in mesh.vertices() {
            if let Some(p) = mesh.point(v) {
                total += p.x + p.y + p.z;
            }
            black_box(total);
        }
        times1.push(start.elapsed().as_nanos());
    }
    let avg1: f64 = times1.iter().map(|&t| t as f64).sum::<f64>() / runs as f64;
    println!("  Average: {:.0} ns ({:.3} ns/v)", avg1, avg1 / n_vertices as f64);
    
    // ========== Test 2: Index Iterator (no Handle) ==========
    println!("\n[2] Index Iterator (usize, no Handle)");
    
    let mut times2: Vec<u128> = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let mut total = 0.0f32;
        for idx in mesh.vertex_indices() {
            unsafe {
                let p = mesh.point_unchecked(idx);
                total += p.x + p.y + p.z;
            }
            black_box(total);
        }
        times2.push(start.elapsed().as_nanos());
    }
    let avg2: f64 = times2.iter().map(|&t| t as f64).sum::<f64>() / runs as f64;
    println!("  Average: {:.0} ns ({:.3} ns/v)", avg2, avg2 / n_vertices as f64);
    println!("  Speedup vs standard: {:.1}x", avg1 / avg2);
    
    // ========== Test 3: Raw pointer (fastest) ==========
    println!("\n[3] Raw Pointer (unsafe, no checks)");
    
    let mut times3: Vec<u128> = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let mut total = 0.0f32;
        let ptr = unsafe { mesh.vertex_ptr() };
        for i in 0..n_vertices {
            unsafe {
                total += (*ptr.add(i)).point.x;
            }
            black_box(total);
        }
        times3.push(start.elapsed().as_nanos());
    }
    let avg3: f64 = times3.iter().map(|&t| t as f64).sum::<f64>() / runs as f64;
    println!("  Average: {:.0} ns ({:.3} ns/v)", avg3, avg3 / n_vertices as f64);
    println!("  Speedup vs standard: {:.1}x", avg1 / avg3);
    
    // ========== Test 4: Just counting ==========
    println!("\n[4] Just Counting (baseline)");
    
    let mut times4: Vec<u128> = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let mut count = 0;
        for _ in mesh.vertex_indices() {
            count += 1;
            black_box(count);
        }
        times4.push(start.elapsed().as_nanos());
    }
    let avg4: f64 = times4.iter().map(|&t| t as f64).sum::<f64>() / runs as f64;
    println!("  Average: {:.0} ns ({:.3} ns/v)", avg4, avg4 / n_vertices as f64);
    
    // ========== Test 5: Face iteration (index) ==========
    println!("\n[5] Face Iteration (index iterator)");
    
    let mut times5: Vec<u128> = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let mut count = 0;
        for _ in mesh.face_indices() {
            count += 1;
            black_box(count);
        }
        times5.push(start.elapsed().as_nanos());
    }
    let avg5: f64 = times5.iter().map(|&t| t as f64).sum::<f64>() / runs as f64;
    println!("  Average: {:.0} ns ({:.3} ns/f)", avg5, avg5 / n_faces as f64);
    
    // ========== Comparison with OpenMesh ==========
    println!("\n========================================");
    println!("OpenMesh Comparison");
    println!("========================================");
    
    let openmesh_v = 291.0;  // ns for 1089 vertices
    let openmesh_f = 84.0;   // ns for 2048 faces
    
    let rust_per_v = avg2 / n_vertices as f64;
    let rust_per_f = avg5 / n_faces as f64;
    let openmesh_per_v = openmesh_v / 1089.0;
    let openmesh_per_f = openmesh_f / 2048.0;
    
    println!("\nPer-vertex cost:");
    println!("  OpenMesh (1K):    {:.3} ns/v", openmesh_per_v);
    println!("  RustMesh Index:   {:.3} ns/v", rust_per_v);
    println!("  Gap: {:.1}x", rust_per_v / openmesh_per_v);
    
    println!("\nPer-face cost:");
    println!("  OpenMesh (2K):    {:.3} ns/f", openmesh_per_f);
    println!("  RustMesh Index:   {:.3} ns/f", rust_per_f);
    println!("  Gap: {:.1}x", rust_per_f / openmesh_per_f);
    
    // ========== Summary ==========
    println!("\n========================================");
    println!("Summary");
    println!("========================================");
    
    println!("\nOptimization Results:");
    println!("  Standard API:    {:.0} ns/v (baseline)", avg1 / n_vertices as f64);
    println!("  Index Iterator: {:.0} ns/v ({:.1}x faster)", avg2 / n_vertices as f64, avg1 / avg2);
    println!("  Raw Pointer:    {:.0} ns/v ({:.1}x faster)", avg3 / n_vertices as f64, avg1 / avg3);
    
    println!("\nGap vs OpenMesh:");
    let gap = (rust_per_v / openmesh_per_v + rust_per_f / openmesh_per_f) / 2.0;
    println!("  Average gap: {:.1}x", gap);
    
    if gap < 3.0 {
        println!("\nRustMesh is competitive!");
    } else if gap < 5.0 {
        println!("\nGap is acceptable ({:.1}x)", gap);
    } else {
        println!("\nNeed more optimization (gap: {:.1}x)", gap);
    }
}
