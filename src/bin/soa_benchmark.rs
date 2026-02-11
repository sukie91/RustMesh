// SoA Layout Performance Benchmark
// Compare AoS (Array of Structures) vs SoA (Structure of Arrays)

use std::time::Instant;
use std::hint::black_box;

fn main() {
    println!("========================================");
    println!("RustMesh AoS vs SoA Performance Benchmark");
    println!("========================================\n");

    let segments = 512;
    let rings = 512;
    
    // Generate test mesh using standard PolyMesh (AoS)
    println!("Generating test mesh ({}x{})...", segments, rings);
    let orig_mesh = rustmesh::generate_sphere(100.0, segments, rings);
    let n_vertices = orig_mesh.n_vertices();
    let n_faces = orig_mesh.n_faces();
    
    println!("  Vertices: {}", n_vertices);
    println!("  Faces: {}\n", n_faces);
    
    // Create SoA mesh from PolyMesh
    println!("Converting to PolyMeshSoA (SoA layout)...");
    let mut soa_mesh = rustmesh::PolyMeshSoA::new();
    
    for idx in orig_mesh.vertex_indices() {
        unsafe {
            let p = orig_mesh.point_unchecked(idx);
            soa_mesh.add_vertex(p);
        }
    }
    
    for fh in orig_mesh.faces() {
        // Skip face iteration for now - need to handle connectivity
    }
    
    println!("  Conversion complete\n");
    
    let runs = 10;
    
    // ========== Test 1: Vertex Iteration ==========
    println!("[1] Vertex Iteration (sum x+y+z)");
    
    // AoS (standard)
    let mut aos_times = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let mut total = 0.0f32;
        for v in orig_mesh.vertices() {
            if let Some(p) = orig_mesh.point(v) {
                total += p.x + p.y + p.z;
            }
            black_box(total);
        }
        aos_times.push(start.elapsed().as_nanos());
    }
    let aos_avg = aos_times.iter().sum::<u128>() as f64 / runs as f64;
    
    // SoA (index iterator + unchecked)
    let mut soa_times = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let mut total = 0.0f32;
        let n = soa_mesh.n_vertices();
        for i in 0..n {
            unsafe {
                let p = soa_mesh.point_unchecked(i);
                total += p.x + p.y + p.z;
            }
            black_box(total);
        }
        soa_times.push(start.elapsed().as_nanos());
    }
    let soa_avg = soa_times.iter().sum::<u128>() as f64 / runs as f64;
    
    println!("  AoS (PolyMesh):  {:.0} ns ({:.3} ns/v)", aos_avg, aos_avg / n_vertices as f64);
    println!("  SoA (PolyMeshSoA): {:.0} ns ({:.3} ns/v)", soa_avg, soa_avg / n_vertices as f64);
    if aos_avg > 0.0 {
        println!("  Speedup: {:.1}x", aos_avg / soa_avg);
    }
    
    // ========== Test 2: Direct Coordinate Access ==========
    println!("\n[2] Direct Coordinate Access (x only)");
    
    // AoS (standard)
    let mut aos_x_times = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let mut total = 0.0f32;
        for v in orig_mesh.vertices() {
            if let Some(p) = orig_mesh.point(v) {
                total += p.x;
            }
            black_box(total);
        }
        aos_x_times.push(start.elapsed().as_nanos());
    }
    let aos_x_avg = aos_x_times.iter().sum::<u128>() as f64 / runs as f64;
    
    // SoA (direct x slice)
    let mut soa_x_times = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let mut total = 0.0f32;
        let x = soa_mesh.x();
        for i in 0..x.len() {
            total += x[i];
            black_box(total);
        }
        soa_x_times.push(start.elapsed().as_nanos());
    }
    let soa_x_avg = soa_x_times.iter().sum::<u128>() as f64 / runs as f64;
    
    println!("  AoS (PolyMesh):  {:.0} ns", aos_x_avg);
    println!("  SoA (PolyMeshSoA): {:.0} ns", soa_x_avg);
    if aos_x_avg > 0.0 {
        println!("  Speedup: {:.1}x", aos_x_avg / soa_x_avg);
    }
    
    // ========== Test 3: Bounding Box ==========
    println!("\n[3] Bounding Box");
    
    // AoS (standard)
    let mut aos_bb_times = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let mut min_x = f32::MAX;
        let mut max_x = f32::MIN;
        let mut min_y = f32::MAX;
        let mut max_y = f32::MIN;
        let mut min_z = f32::MAX;
        let mut max_z = f32::MIN;
        
        for v in orig_mesh.vertices() {
            if let Some(p) = orig_mesh.point(v) {
                if p.x < min_x { min_x = p.x; }
                if p.x > max_x { max_x = p.x; }
                if p.y < min_y { min_y = p.y; }
                if p.y > max_y { max_y = p.y; }
                if p.z < min_z { min_z = p.z; }
                if p.z > max_z { max_z = p.z; }
            }
            black_box((min_x, max_x, min_y, max_y, min_z, max_z));
        }
        aos_bb_times.push(start.elapsed().as_nanos());
    }
    let aos_bb_avg = aos_bb_times.iter().sum::<u128>() as f64 / runs as f64;
    
    // SoA (optimized method)
    let mut soa_bb_times = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let bb = soa_mesh.bounding_box();
        black_box(bb);
        soa_bb_times.push(start.elapsed().as_nanos());
    }
    let soa_bb_avg = soa_bb_times.iter().sum::<u128>() as f64 / runs as f64;
    
    println!("  AoS (PolyMesh):  {:.0} ns", aos_bb_avg);
    println!("  SoA (PolyMeshSoA): {:.0} ns", soa_bb_avg);
    if aos_bb_avg > 0.0 {
        println!("  Speedup: {:.1}x", aos_bb_avg / soa_bb_avg);
    }
    
    // ========== Test 4: Centroid ==========
    println!("\n[4] Centroid");
    
    // AoS
    let mut aos_cent_times = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let mut sum_x = 0.0f32;
        let mut sum_y = 0.0f32;
        let mut sum_z = 0.0f32;
        for v in orig_mesh.vertices() {
            if let Some(p) = orig_mesh.point(v) {
                sum_x += p.x;
                sum_y += p.y;
                sum_z += p.z;
            }
        }
        let cx = sum_x / n_vertices as f32;
        let cy = sum_y / n_vertices as f32;
        let cz = sum_z / n_vertices as f32;
        black_box((cx, cy, cz));
        aos_cent_times.push(start.elapsed().as_nanos());
    }
    let aos_cent_avg = aos_cent_times.iter().sum::<u128>() as f64 / runs as f64;
    
    // SoA
    let mut soa_cent_times = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let cent = soa_mesh.centroid();
        black_box(cent);
        soa_cent_times.push(start.elapsed().as_nanos());
    }
    let soa_cent_avg = soa_cent_times.iter().sum::<u128>() as f64 / runs as f64;
    
    println!("  AoS (PolyMesh):  {:.0} ns", aos_cent_avg);
    println!("  SoA (PolyMeshSoA): {:.0} ns", soa_cent_avg);
    if aos_cent_avg > 0.0 {
        println!("  Speedup: {:.1}x", aos_cent_avg / soa_cent_avg);
    }
    
    // ========== Summary ==========
    println!("\n========================================");
    println!("Summary - AoS vs SoA");
    println!("========================================");
    
    println!("\nPerformance Comparison:");
    println!("  Vertex Iteration:   {:.1}x", aos_avg / soa_avg);
    println!("  X Coordinate Only:  {:.1}x", aos_x_avg / soa_x_avg);
    println!("  Bounding Box:       {:.1}x", aos_bb_avg / soa_bb_avg);
    println!("  Centroid:          {:.1}x", aos_cent_avg / soa_cent_avg);
    
    let overall_speedup = (aos_avg / soa_avg + aos_bb_avg / soa_bb_avg) / 2.0;
    println!("\n  Overall Speedup:    {:.1}x", overall_speedup);
    
    if overall_speedup > 1.0 {
        println!("\nSoA layout is faster!");
    } else {
        println!("\nAoS layout may be sufficient.");
    }
    
    // ========== OpenMesh Comparison ==========
    println!("\n========================================");
    println!("OpenMesh Comparison");
    println!("========================================");
    
    let openmesh_v = 291.0;  // ns for 1089 vertices
    let rust_soa_per_v = soa_avg / n_vertices as f64;
    let openmesh_per_v = openmesh_v / 1089.0;
    
    println!("\nPer-vertex cost (SoA):");
    println!("  OpenMesh (1K):  {:.3} ns/v", openmesh_per_v);
    println!("  RustMesh SoA:   {:.3} ns/v", rust_soa_per_v);
    println!("  Gap: {:.1}x", rust_soa_per_v / openmesh_per_v);
    
    if rust_soa_per_v / openmesh_per_v < 2.0 {
        println!("\nRustMesh SoA is competitive!");
    } else if rust_soa_per_v / openmesh_per_v < 5.0 {
        println!("\nGap is acceptable ({:.1}x)", rust_soa_per_v / openmesh_per_v);
    } else {
        println!("\nNeeds more optimization (gap: {:.1}x)", rust_soa_per_v / openmesh_per_v);
    }
}
