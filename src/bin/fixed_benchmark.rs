// Fixed Benchmark: 确保编译器不优化掉计算

use std::time::Instant;
use std::hint::black_box;

fn main() {
    println!("========================================");
    println!("RustMesh Fixed Performance Benchmark");
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
    
    // ========== Test 1: Vertex iteration ==========
    println!("[1] Vertex Iteration");
    
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
        let elapsed = start.elapsed();
        times1.push(elapsed.as_nanos());
        println!("  Run {}: {} ns", times1.len(), elapsed.as_nanos());
    }
    let avg1: f64 = times1.iter().map(|&t| t as f64).sum::<f64>() / runs as f64;
    println!("  Average: {:.0} ns ({:.3} ns/vertex)", avg1, avg1 / n_vertices as f64);
    
    // ========== Test 2: Vertex count ==========
    println!("\n[2] Vertex Count (baseline)");
    
    let mut times2: Vec<u128> = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let mut count = 0;
        for _ in mesh.vertices() {
            count += 1;
            black_box(count);
        }
        let elapsed = start.elapsed();
        times2.push(elapsed.as_nanos());
    }
    let avg2: f64 = times2.iter().map(|&t| t as f64).sum::<f64>() / runs as f64;
    println!("  Average: {:.0} ns ({:.3} ns/vertex)", avg2, avg2 / n_vertices as f64);
    
    // ========== Test 3: Face iteration ==========
    println!("\n[3] Face Iteration");
    
    let mut times3: Vec<u128> = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let mut count = 0;
        for _ in mesh.faces() {
            count += 1;
            black_box(count);
        }
        let elapsed = start.elapsed();
        times3.push(elapsed.as_nanos());
    }
    let avg3: f64 = times3.iter().map(|&t| t as f64).sum::<f64>() / runs as f64;
    println!("  Average: {:.0} ns ({:.3} ns/face)", avg3, avg3 / n_faces as f64);
    
    // ========== Comparison ==========
    println!("\n========================================");
    println!("OpenMesh Comparison");
    println!("========================================");
    
    let openmesh_v = 291.0;  // ns for 1089 vertices
    let openmesh_f = 84.0;   // ns for 2048 faces
    
    let rust_per_v = avg1 / n_vertices as f64;
    let rust_per_f = avg3 / n_faces as f64;
    let openmesh_per_v = openmesh_v / 1089.0;
    let openmesh_per_f = openmesh_f / 2048.0;
    
    println!("\nPer-vertex cost:");
    println!("  OpenMesh (1K):  {:.3} ns/v", openmesh_per_v);
    println!("  RustMesh (262K): {:.3} ns/v", rust_per_v);
    println!("  Gap: {:.1}x", rust_per_v / openmesh_per_v);
    
    println!("\nPer-face cost:");
    println!("  OpenMesh (2K): {:.3} ns/f", openmesh_per_f);
    println!("  RustMesh (524K): {:.3} ns/f", rust_per_f);
    println!("  Gap: {:.1}x", rust_per_f / openmesh_per_f);
    
    println!("\n========================================");
    println!("Conclusion");
    println!("========================================");
    
    let v_gap = rust_per_v / openmesh_per_v;
    let f_gap = rust_per_f / openmesh_per_f;
    
    if v_gap < 2.0 && f_gap < 2.0 {
        println!("RustMesh is competitive with OpenMesh!");
    } else if v_gap < 5.0 {
        println!("RustMesh is {:.1}x slower than OpenMesh", v_gap.max(f_gap));
    } else {
        println!("RustMesh is {:.1}x slower than OpenMesh", v_gap.max(f_gap));
    }
}
