// User Model Performance Benchmark
// Tests RustMesh vs OpenMesh on user's FinalBaseMesh.obj

use std::time::Instant;
use std::hint::black_box;
use std::path::Path;

fn main() {
    println!("========================================");
    println!("User Model Performance Benchmark");
    println!("========================================\n");

    let model_path = "/Users/tfjiang/Projects/RustMesh/test_data/large/FinalBaseMesh.obj";

    if !Path::new(model_path).exists() {
        println!("ERROR: Model not found at {}", model_path);
        return;
    }

    // ========== Load Model ==========
    println!("Loading model: {}", model_path);
    
    let load_start = Instant::now();
    let mesh = rustmesh::read_obj(model_path).expect("Failed to load OBJ");
    let load_time = load_start.elapsed();
    
    let n_vertices = mesh.n_vertices();
    let n_faces = mesh.n_faces();
    
    println!("  Loaded in: {:?}", load_time);
    println!("  Vertices: {}", n_vertices);
    println!("  Faces: {}\n", n_faces);

    let runs = 10;

    // ========== Test 1: Vertex Iteration ==========
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
        times1.push(start.elapsed().as_nanos());
    }
    let avg1: f64 = times1.iter().map(|&t| t as f64).sum::<f64>() / runs as f64;
    println!("  Average: {:.0} ns ({:.3} ns/vertex)", avg1, avg1 / n_vertices as f64);

    // ========== Test 2: Index Iterator (optimized) ==========
    println!("\n[2] Vertex Index Iterator (optimized)");
    
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
    println!("  Average: {:.0} ns ({:.3} ns/vertex)", avg2, avg2 / n_vertices as f64);
    if avg1 > 0.0 {
        println!("  Speedup vs standard: {:.1}x", avg1 / avg2);
    }

    // ========== Test 3: Raw Pointer (fastest) ==========
    println!("\n[3] Raw Pointer (fastest)");
    
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
    println!("  Average: {:.0} ns ({:.3} ns/vertex)", avg3, avg3 / n_vertices as f64);
    if avg1 > 0.0 {
        println!("  Speedup vs standard: {:.1}x", avg1 / avg3);
    }

    // ========== Test 4: Face Iteration ==========
    println!("\n[4] Face Iteration");
    
    let mut times4: Vec<u128> = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let mut count = 0;
        for _ in mesh.faces() {
            count += 1;
            black_box(count);
        }
        times4.push(start.elapsed().as_nanos());
    }
    let avg4: f64 = times4.iter().map(|&t| t as f64).sum::<f64>() / runs as f64;
    println!("  Average: {:.0} ns ({:.3} ns/face)", avg4, avg4 / n_faces as f64);

    // ========== Test 5: Bounding Box ==========
    println!("\n[5] Bounding Box (scalar)");
    
    let mut times5: Vec<u128> = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let mut min_x = f32::MAX;
        let mut max_x = f32::MIN;
        let mut min_y = f32::MAX;
        let mut max_y = f32::MIN;
        let mut min_z = f32::MAX;
        let mut max_z = f32::MIN;
        
        for v in mesh.vertices() {
            if let Some(p) = mesh.point(v) {
                if p.x < min_x { min_x = p.x; }
                if p.x > max_x { max_x = p.x; }
                if p.y < min_y { min_y = p.y; }
                if p.y > max_y { max_y = p.y; }
                if p.z < min_z { min_z = p.z; }
                if p.z > max_z { max_z = p.z; }
            }
            black_box((min_x, max_x, min_y, max_y, min_z, max_z));
        }
        times5.push(start.elapsed().as_nanos());
    }
    let avg5: f64 = times5.iter().map(|&t| t as f64).sum::<f64>() / runs as f64;
    println!("  Average: {:.0} ns", avg5);

    // ========== Test 6: Centroid ==========
    println!("\n[6] Centroid");
    
    let mut times6: Vec<u128> = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let mut sum_x = 0.0f32;
        let mut sum_y = 0.0f32;
        let mut sum_z = 0.0f32;
        
        for v in mesh.vertices() {
            if let Some(p) = mesh.point(v) {
                sum_x += p.x;
                sum_y += p.y;
                sum_z += p.z;
            }
        }
        let cx = sum_x / n_vertices as f32;
        let cy = sum_y / n_vertices as f32;
        let cz = sum_z / n_vertices as f32;
        black_box((cx, cy, cz));
        times6.push(start.elapsed().as_nanos());
    }
    let avg6: f64 = times6.iter().map(|&t| t as f64).sum::<f64>() / runs as f64;
    println!("  Average: {:.0} ns", avg6);

    // ========== OpenMesh Comparison ==========
    println!("\n========================================");
    println!("OpenMesh Comparison (extrapolated)");
    println!("========================================");

    // OpenMesh baseline (1K vertices, 2K faces)
    let openmesh_v = 291.0;   // ns for 1089 vertices
    let openmesh_f = 84.0;    // ns for 2048 faces
    let openmesh_bb = 700.0;  // ns estimate for bounding box
    
    let rust_per_v = avg3 / n_vertices as f64;  // Use best RustMesh result
    let rust_per_f = avg4 / n_faces as f64;
    let openmesh_per_v = openmesh_v / 1089.0;
    let openmesh_per_f = openmesh_f / 2048.0;
    
    println!("\nPer-vertex cost:");
    println!("  OpenMesh (1K):  {:.3} ns/v", openmesh_per_v);
    println!("  RustMesh ({}K): {:.3} ns/v", n_vertices / 1000, rust_per_v);
    println!("  Gap: {:.1}x", rust_per_v / openmesh_per_v);
    
    println!("\nPer-face cost:");
    println!("  OpenMesh (2K): {:.3} ns/f", openmesh_per_f);
    println!("  RustMesh ({}K): {:.3} ns/f", n_faces / 1000, rust_per_f);
    println!("  Gap: {:.1}x", rust_per_f / openmesh_per_f);

    // ========== Summary ==========
    println!("\n========================================");
    println!("Summary - User Model: FinalBaseMesh.obj");
    println!("========================================");
    
    println!("\nModel Info:");
    println!("  Vertices: {}", n_vertices);
    println!("  Faces:    {}", n_faces);
    
    println!("\nPerformance:");
    println!("  Vertex Iteration: {:.1}x", rust_per_v / openmesh_per_v);
    println!("  Face Iteration:   {:.1}x", rust_per_f / openmesh_per_f);
    
    let avg_gap = (rust_per_v / openmesh_per_v + rust_per_f / openmesh_per_f) / 2.0;
    println!("\n  Average gap: {:.1}x", avg_gap);
    
    if avg_gap < 1.5 {
        println!("\nRustMesh is competitive with OpenMesh!");
    } else if avg_gap < 3.0 {
        println!("\nRustMesh is {:.1}x slower (acceptable)", avg_gap);
    } else {
        println!("\nRustMesh is {:.1}x slower (needs optimization)", avg_gap);
    }

    // ========== Per-Operation Breakdown ==========
    println!("\n========================================");
    println!("Detailed Breakdown");
    println!("========================================");
    
    println!("\nOperation            | Time      | Per-Item | OpenMesh | Gap");
    println!("---------------------|-----------|----------|----------|-----");
    println!("Vertex (standard)    | {:7.0} μs | {:.3} ns | {:.3} ns | {:.1}x", 
        avg1 / 1000.0, avg1 / n_vertices as f64, openmesh_per_v, rust_per_v / openmesh_per_v);
    println!("Vertex (index iter)  | {:7.0} μs | {:.3} ns | {:.3} ns | {:.1}x", 
        avg2 / 1000.0, avg2 / n_vertices as f64, openmesh_per_v, (avg2 / n_vertices as f64) / openmesh_per_v);
    println!("Vertex (raw pointer) | {:7.0} μs | {:.3} ns | {:.3} ns | {:.1}x", 
        avg3 / 1000.0, avg3 / n_vertices as f64, openmesh_per_v, (avg3 / n_vertices as f64) / openmesh_per_v);
    println!("Face iteration       | {:7.0} μs | {:.3} ns | {:.3} ns | {:.1}x", 
        avg4 / 1000.0, avg4 / n_faces as f64, openmesh_per_f, (avg4 / n_faces as f64) / openmesh_per_f);
    println!("Bounding Box         | {:7.0} μs | -        | -        | -", 
        avg5 / 1000.0);
    println!("Centroid             | {:7.0} μs | -        | -        | -", 
        avg6 / 1000.0);
}
