// SoA + SIMD Performance Benchmark
// Compare: AoS vs SoA vs SoA+SIMD

use std::time::Instant;
use std::hint::black_box;

fn main() {
    println!("========================================");
    println!("RustMesh Performance: AoS vs SoA vs SoA+SIMD");
    println!("========================================\n");

    let segments = 512;
    let rings = 512;
    
    println!("Generating test mesh ({}x{})...", segments, rings);
    let orig_mesh = rustmesh::generate_sphere(100.0, segments, rings);
    let n_vertices = orig_mesh.n_vertices();
    let n_faces = orig_mesh.n_faces();
    
    println!("  Vertices: {}", n_vertices);
    println!("  Faces: {}\n", n_faces);
    
    // Create SoA mesh
    let mut soa_mesh = rustmesh::PolyMeshSoA::new();
    for idx in orig_mesh.vertex_indices() {
        unsafe {
            let p = orig_mesh.point_unchecked(idx);
            soa_mesh.add_vertex(p);
        }
    }
    
    let runs = 10;
    
    // ========== Test 1: Bounding Box ==========
    println!("[1] Bounding Box Comparison");
    
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
    
    // SoA (scalar)
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
    println!("\n[2] Centroid Comparison");
    
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
    println!("\n[3] Vertex Sum Comparison");
    
    // SoA + SIMD
    let mut soa_simd_sum_times = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let sum = unsafe { soa_mesh.vertex_sum_simd() };
        black_box(sum);
        soa_simd_sum_times.push(start.elapsed().as_nanos());
    }
    let soa_simd_sum_avg = soa_simd_sum_times.iter().sum::<u128>() as f64 / runs as f64;
    
    println!("  SoA + SIMD:          {:.0} ns ({:.3} ns/v)", soa_simd_sum_avg, soa_simd_sum_avg / n_vertices as f64);
    
    // ========== Summary ==========
    println!("\n========================================");
    println!("Summary - Performance Evolution");
    println!("========================================");
    
    println!("\nBounding Box:");
    println!("  AoS:        {:.0} ns (baseline)", aos_bb_avg);
    println!("  SoA:        {:.0} ns ({:.1}x)", soa_scalar_bb_avg, aos_bb_avg / soa_scalar_bb_avg);
    println!("  SoA + SIMD: {:.0} ns ({:.1}x)", soa_simd_bb_avg, aos_bb_avg / soa_simd_bb_avg);
    
    println!("\nCentroid:");
    println!("  AoS:        {:.0} ns (baseline)", aos_cent_avg);
    println!("  SoA:        {:.0} ns ({:.1}x)", soa_scalar_cent_avg, aos_cent_avg / soa_scalar_cent_avg);
    println!("  SoA + SIMD: {:.0} ns ({:.1}x)", soa_simd_cent_avg, aos_cent_avg / soa_simd_cent_avg);
    
    // ========== OpenMesh Comparison ==========
    println!("\n========================================");
    println!("OpenMesh Comparison");
    println!("========================================");
    
    let openmesh_v = 291.0;  // ns for 1089 vertices
    let openmesh_per_v = openmesh_v / 1089.0;
    let rust_soa_simd_per_v = soa_simd_sum_avg / n_vertices as f64;
    
    println!("\nPer-vertex cost (SoA + SIMD):");
    println!("  OpenMesh (1K):  {:.3} ns/v", openmesh_per_v);
    println!("  RustMesh SoA+SIMD: {:.3} ns/v", rust_soa_simd_per_v);
    println!("  Gap: {:.1}x", rust_soa_simd_per_v / openmesh_per_v);
    
    let overall_gap = rust_soa_simd_per_v / openmesh_per_v;
    if overall_gap < 1.5 {
        println!("\nRustMesh SoA + SIMD is competitive!");
    } else if overall_gap < 3.0 {
        println!("\nGap is acceptable ({:.1}x)", overall_gap);
    } else {
        println!("\nNeeds more optimization (gap: {:.1}x)", overall_gap);
    }
    
    // ========== Performance Milestones ==========
    println!("\n========================================");
    println!("Performance Milestones");
    println!("========================================");
    
    println!("\nOriginal (wrong):     <1 ns/v (fake)");
    println!("Fixed (bug):         1.93 ns/v (7.2x slow)");
    println!("Index Iterator:      0.91 ns/v (3.4x slow)");
    println!("Raw Pointer:         0.76 ns/v (2.8x slow)");
    println!("SIMD (262K):         0.33 ns/v (1.2x slow)");
    println!("SoA (262K):          0.82 ns/v (3.1x slow)");
    println!("SoA + SIMD (262K):   {:.3} ns/v ({:.1}x slow)", 
        rust_soa_simd_per_v, rust_soa_simd_per_v / openmesh_per_v);
}
