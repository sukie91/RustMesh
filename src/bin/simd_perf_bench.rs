// SIMD Performance Benchmark

use std::hint::black_box;
use std::time::Instant;

fn main() {
    println!("========================================");
    println!("RustMesh SIMD Performance Benchmark");
    println!("========================================\n");

    let n = 263_169;
    
    println!("Generating SoA test data ({} vertices)...\n", n);
    
    // Create SoA layout data
    let x: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
    let y: Vec<f32> = (0..n).map(|i| (i as f32) * 0.002 + 1.0).collect();
    let z: Vec<f32> = (0..n).map(|i| (i as f32) * 0.003 + 2.0).collect();
    
    // Create faces
    let faces: Vec<u32> = (0..524_287 * 3).map(|i| (i % n) as u32).collect();
    
    let runs = 10;
    
    // ========== Test 1: Vertex Sum ==========
    println!("[1] Vertex Sum (3 coords)");
    
    // Scalar baseline
    let mut scalar_times = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let mut sum_x = 0.0f32;
        let mut sum_y = 0.0f32;
        let mut sum_z = 0.0f32;
        for i in 0..n {
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
    
    println!("  Scalar: {:.0} ns ({:.3} ns/vertex)", scalar_avg, scalar_avg / n as f64);
    println!("  SIMD:   {:.0} ns ({:.3} ns/vertex)", simd_avg, simd_avg / n as f64);
    println!("  Speedup: {:.1}x", scalar_avg / simd_avg);
    
    // ========== Test 2: Bounding Box ==========
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
        for i in 0..n {
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
    
    println!("  Scalar: {:.0} ns ({:.3} ns/vertex)", scalar_bb_avg, scalar_bb_avg / n as f64);
    println!("  SIMD:   {:.0} ns ({:.3} ns/vertex)", simd_bb_avg, simd_bb_avg / n as f64);
    println!("  Speedup: {:.1}x", scalar_bb_avg / simd_bb_avg);
    
    // ========== Test 3: Centroid ==========
    println!("\n[3] Centroid");
    
    // Scalar
    let mut scalar_cent_times = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let mut sum_x = 0.0f32;
        let mut sum_y = 0.0f32;
        let mut sum_z = 0.0f32;
        for i in 0..n {
            sum_x += x[i];
            sum_y += y[i];
            sum_z += z[i];
        }
        let cx = sum_x / n as f32;
        let cy = sum_y / n as f32;
        let cz = sum_z / n as f32;
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
    
    println!("  Scalar: {:.0} ns ({:.3} ns/vertex)", scalar_cent_avg, scalar_cent_avg / n as f64);
    println!("  SIMD:   {:.0} ns ({:.3} ns/vertex)", simd_cent_avg, simd_cent_avg / n as f64);
    println!("  Speedup: {:.1}x", scalar_cent_avg / simd_cent_avg);
    
    // ========== Test 4: Surface Area ==========
    println!("\n[4] Surface Area ({} faces)", n / 2);
    
    // Scalar
    let mut scalar_area_times = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let mut area = 0.0f32;
        for i in 0..faces.len() / 3 {
            let i0 = faces[i * 3] as usize;
            let i1 = faces[i * 3 + 1] as usize;
            let i2 = faces[i * 3 + 2] as usize;
            
            let ax = x[i0];
            let ay = y[i0];
            let az = z[i0];
            let bx = x[i1];
            let by = y[i1];
            let bz = z[i1];
            let cx = x[i2];
            let cy = y[i2];
            let cz = z[i2];
            
            let bax = bx - ax;
            let bay = by - ay;
            let baz = bz - az;
            let cax = cx - ax;
            let cay = cy - ay;
            let caz = cz - az;
            
            let cx1 = bay * caz - baz * cay;
            let cy1 = baz * cax - bax * caz;
            let cz1 = bax * cay - bay * cax;
            
            area += 0.5 * (cx1 * cx1 + cy1 * cy1 + cz1 * cz1).sqrt();
            black_box(area);
        }
        scalar_area_times.push(start.elapsed().as_nanos());
    }
    let scalar_area_avg = scalar_area_times.iter().sum::<u128>() as f64 / runs as f64;
    
    // SIMD
    let mut simd_area_times = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let area = unsafe { rustmesh::surface_area_simd(&x, &y, &z, &faces) };
        black_box(area);
        simd_area_times.push(start.elapsed().as_nanos());
    }
    let simd_area_avg = simd_area_times.iter().sum::<u128>() as f64 / runs as f64;
    
    println!("  Scalar: {:.0} ns", scalar_area_avg);
    println!("  SIMD:   {:.0} ns", simd_area_avg);
    println!("  Speedup: {:.1}x", scalar_area_avg / simd_area_avg);
    
    // ========== Comparison with OpenMesh ==========
    println!("\n========================================");
    println!("OpenMesh Comparison");
    println!("========================================");
    
    let openmesh_v = 291.0;  // ns for 1089 vertices
    
    // Extrapolate OpenMesh to our size
    let rust_per_v = simd_avg / n as f64;
    let openmesh_per_v = openmesh_v / 1089.0;
    
    println!("\nPer-vertex cost (SIMD):");
    println!("  OpenMesh (1K):  {:.3} ns/v", openmesh_per_v);
    println!("  RustMesh SIMD: {:.3} ns/v", rust_per_v);
    println!("  Gap: {:.1}x", rust_per_v / openmesh_per_v);
    
    // ========== Summary ==========
    println!("\n========================================");
    println!("Summary");
    println!("========================================");
    
    println!("\nSIMD Optimization Results:");
    println!("  Vertex Sum:    {:.1}x faster", scalar_avg / simd_avg);
    println!("  Bounding Box:  {:.1}x faster", scalar_bb_avg / simd_bb_avg);
    println!("  Centroid:      {:.1}x faster", scalar_cent_avg / simd_cent_avg);
    println!("  Surface Area:  {:.1}x faster", scalar_area_avg / simd_area_avg);
    
    let overall_speedup = (scalar_avg / simd_avg + scalar_bb_avg / simd_bb_avg) / 2.0;
    
    println!("\nOverall SIMD Speedup: {:.1}x", overall_speedup);
    
    // Previous gap was 3.4x, now apply SIMD
    let new_gap = 3.4 / overall_speedup;
    println!("\nExpected gap vs OpenMesh: {:.1}x (was 3.4x)", new_gap);
    
    if new_gap < 2.0 {
        println!("\nRustMesh SIMD is competitive!");
    } else {
        println!("\nContinue optimization (gap: {:.1}x)", new_gap);
    }
}
