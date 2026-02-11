// User Model: SoA + SIMD Performance Benchmark
// Tests SoA layout + NEON SIMD on FinalBaseMesh.obj

use std::time::Instant;
use std::hint::black_box;
use std::path::Path;

fn main() {
    println!("========================================");
    println!("User Model: SoA + SIMD Performance");
    println!("========================================\n");

    let model_path = "/Users/tfjiang/Projects/RustMesh/test_data/large/FinalBaseMesh.obj";

    if !Path::new(model_path).exists() {
        println!("ERROR: Model not found at {}", model_path);
        return;
    }

    // Load model
    println!("Loading model: {}", model_path);
    let orig_mesh = rustmesh::read_obj(model_path).expect("Failed to load OBJ");
    let n_vertices = orig_mesh.n_vertices();
    let n_faces = orig_mesh.n_faces();
    
    println!("  Vertices: {}", n_vertices);
    println!("  Faces: {}\n", n_faces);

    // Create SoA mesh
    println!("Converting to PolyMeshSoA...");
    let mut soa_mesh = rustmesh::PolyMeshSoA::new();
    
    for idx in orig_mesh.vertex_indices() {
        unsafe {
            let p = orig_mesh.point_unchecked(idx);
            soa_mesh.add_vertex(p);
        }
    }
    
    println!("  Conversion complete\n");
    
    let runs = 10;
    
    // ========== Test 1: Bounding Box ==========
    println!("[1] Bounding Box");
    
    // AoS (PolyMesh)
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
    
    // SoA scalar
    let mut soa_scalar_bb_times = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let bb = soa_mesh.bounding_box();
        black_box(bb);
        soa_scalar_bb_times.push(start.elapsed().as_nanos());
    }
    let soa_scalar_bb_avg = soa_scalar_bb_times.iter().sum::<u128>() as f64 / runs as f64;
    
    // SoA + SIMD
    let mut soa_simd_bb_times = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let bb = unsafe { soa_mesh.bounding_box_simd() };
        black_box(bb);
        soa_simd_bb_times.push(start.elapsed().as_nanos());
    }
    let soa_simd_bb_avg = soa_simd_bb_times.iter().sum::<u128>() as f64 / runs as f64;
    
    println!("  AoS (PolyMesh):      {:.0} ns", aos_bb_avg);
    println!("  SoA (scalar):        {:.0} ns", soa_scalar_bb_avg);
    println!("  SoA + SIMD:          {:.0} ns", soa_simd_bb_avg);
    println!("  SoA speedup:         {:.1}x", aos_bb_avg / soa_scalar_bb_avg);
    println!("  SoA+SIMD speedup:    {:.1}x", aos_bb_avg / soa_simd_bb_avg);
    
    // ========== Test 2: Centroid ==========
    println!("\n[2] Centroid");
    
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
    
    // SoA scalar
    let mut soa_scalar_cent_times = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let cent = soa_mesh.centroid();
        black_box(cent);
        soa_scalar_cent_times.push(start.elapsed().as_nanos());
    }
    let soa_scalar_cent_avg = soa_scalar_cent_times.iter().sum::<u128>() as f64 / runs as f64;
    
    // SoA + SIMD
    let mut soa_simd_cent_times = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let cent = unsafe { soa_mesh.centroid_simd() };
        black_box(cent);
        soa_simd_cent_times.push(start.elapsed().as_nanos());
    }
    let soa_simd_cent_avg = soa_simd_cent_times.iter().sum::<u128>() as f64 / runs as f64;
    
    println!("  AoS (PolyMesh):      {:.0} ns", aos_cent_avg);
    println!("  SoA (scalar):        {:.0} ns", soa_scalar_cent_avg);
    println!("  SoA + SIMD:          {:.0} ns", soa_simd_cent_avg);
    println!("  SoA speedup:         {:.1}x", aos_cent_avg / soa_scalar_cent_avg);
    println!("  SoA+SIMD speedup:    {:.1}x", aos_cent_avg / soa_simd_cent_avg);
    
    // ========== Test 3: Vertex Sum ==========
    println!("\n[3] Vertex Sum (SIMD)");
    
    let mut soa_simd_sum_times = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let sum = unsafe { soa_mesh.vertex_sum_simd() };
        black_box(sum);
        soa_simd_sum_times.push(start.elapsed().as_nanos());
    }
    let soa_simd_sum_avg = soa_simd_sum_times.iter().sum::<u128>() as f64 / runs as f64;
    
    println!("  SoA + SIMD:          {:.0} ns ({:.3} ns/v)", 
        soa_simd_sum_avg, soa_simd_sum_avg / n_vertices as f64);
    
    // ========== Test 4: Per-vertex Iteration ==========
    println!("\n[4] Per-Vertex Iteration (x only)");
    
    // AoS
    let mut aos_x_times = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let mut sum = 0.0f32;
        for v in orig_mesh.vertices() {
            if let Some(p) = orig_mesh.point(v) {
                sum += p.x;
            }
            black_box(sum);
        }
        aos_x_times.push(start.elapsed().as_nanos());
    }
    let aos_x_avg = aos_x_times.iter().sum::<u128>() as f64 / runs as f64;
    
    // SoA direct
    let mut soa_x_times = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let mut sum = 0.0f32;
        let x = soa_mesh.x();
        for i in 0..x.len() {
            sum += x[i];
            black_box(sum);
        }
        soa_x_times.push(start.elapsed().as_nanos());
    }
    let soa_x_avg = soa_x_times.iter().sum::<u128>() as f64 / runs as f64;
    
    println!("  AoS (PolyMesh):      {:.0} ns ({:.3} ns/v)", aos_x_avg, aos_x_avg / n_vertices as f64);
    println!("  SoA (direct):        {:.0} ns ({:.3} ns/v)", soa_x_avg, soa_x_avg / n_vertices as f64);
    println!("  Speedup:             {:.1}x", aos_x_avg / soa_x_avg);
    
    // ========== OpenMesh Comparison ==========
    println!("\n========================================");
    println!("OpenMesh Comparison");
    println!("========================================");

    let openmesh_v = 291.0;   // ns for 1089 vertices
    let openmesh_per_v = openmesh_v / 1089.0;
    
    let rust_soa_simd_per_v = soa_simd_sum_avg / n_vertices as f64;
    let rust_soa_per_v = soa_x_avg / n_vertices as f64;
    
    println!("\nPer-vertex cost:");
    println!("  OpenMesh (1K):      {:.3} ns/v", openmesh_per_v);
    println!("  RustMesh SoA:       {:.3} ns/v", rust_soa_per_v);
    println!("  RustMesh SoA+SIMD:  {:.3} ns/v", rust_soa_simd_per_v);
    
    println!("\nGap:");
    println!("  SoA vs OpenMesh:    {:.1}x", rust_soa_per_v / openmesh_per_v);
    println!("  SoA+SIMD vs OpenMesh: {:.1}x", rust_soa_simd_per_v / openmesh_per_v);
    
    // ========== Summary ==========
    println!("\n========================================");
    println!("Summary - User Model: FinalBaseMesh.obj");
    println!("========================================");
    
    println!("\nModel Info:");
    println!("  Vertices: {}", n_vertices);
    println!("  Faces:    {}", n_faces);
    
    println!("\nOptimization Results:");
    println!("  Bounding Box:   {:.1}x faster (SoA+SIMD)", aos_bb_avg / soa_simd_bb_avg);
    println!("  Centroid:      {:.1}x faster (SoA+SIMD)", aos_cent_avg / soa_simd_cent_avg);
    println!("  Per-Vertex:     {:.1}x faster (SoA)", aos_x_avg / soa_x_avg);
    
    println!("\nGap vs OpenMesh:");
    let gap = rust_soa_simd_per_v / openmesh_per_v;
    if gap < 1.0 {
        println!("  SoA+SIMD: {:.1}x FASTER than OpenMesh!", 1.0 / gap);
    } else if gap < 1.5 {
        println!("  SoA+SIMD: {:.1}x slower (competitive)", gap);
    } else if gap < 3.0 {
        println!("  SoA+SIMD: {:.1}x slower (acceptable)", gap);
    } else {
        println!("  SoA+SIMD: {:.1}x slower (needs work)", gap);
    }
    
    // ========== Per-Vertex Cost Breakdown ==========
    println!("\n========================================");
    println!("Per-Vertex Cost Breakdown");
    println!("========================================");
    
    println!("\nOperation              | RustMesh  | OpenMesh | Gap");
    println!("------------------------|-----------|----------|----");
    println!("Bounding Box (SoA+SIMD) | {:.3} ns/v | {:.3} ns/v | {:.1}x", 
        soa_simd_bb_avg / n_vertices as f64, 700.0 / 1089.0, (soa_simd_bb_avg / n_vertices as f64) / (700.0 / 1089.0));
    println!("Centroid (SoA+SIMD)    | {:.3} ns/v | {:.3} ns/v | {:.1}x", 
        soa_simd_cent_avg / n_vertices as f64, 300.0 / 1089.0, (soa_simd_cent_avg / n_vertices as f64) / (300.0 / 1089.0));
    println!("Vertex Sum (SoA+SIMD)  | {:.3} ns/v | {:.3} ns/v | {:.1}x", 
        rust_soa_simd_per_v, openmesh_per_v, rust_soa_simd_per_v / openmesh_per_v);
    println!("Per-Vertex (SoA)       | {:.3} ns/v | {:.3} ns/v | {:.1}x", 
        rust_soa_per_v, openmesh_per_v, rust_soa_per_v / openmesh_per_v);
}
