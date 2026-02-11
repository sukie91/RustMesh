// High-Performance Mesh Benchmark
// Compare: Original RustMesh vs HighPerfMesh vs OpenMesh

use std::time::Instant;
use std::hint::black_box;

fn main() {
    println!("========================================");
    println!("RustMesh 高性能 Mesh 对比测试");
    println!("原始 vs 高性能 vs OpenMesh");
    println!("========================================\n");

    // Generate meshes
    let segments = 512;
    let rings = 512;

    println!("生成测试网格 ({}x{})...", segments, rings);
    
    // Generate original mesh
    let orig_mesh = rustmesh::generate_sphere(100.0, segments, rings);
    let n_vertices = orig_mesh.n_vertices();
    let n_faces = orig_mesh.n_faces();
    
    println!("  顶点数: {}", n_vertices);
    println!("  面片数: {}\n", n_faces);

    // Generate high-perf mesh
    let hp_mesh = rustmesh::HighPerfMesh::new();
    
    let runs = 50;

    // ========== Test 1: Vertex Iteration ==========
    println!("[1] 顶点遍历测试");
    println!("----------------------------------------");

    // Original mesh
    let mut times1 = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let mut sum = 0.0f32;
        for v in orig_mesh.vertices() {
            if let Some(p) = orig_mesh.point(v) {
                sum += p.x;
            }
            black_box(sum);
        }
        times1.push(start.elapsed().as_nanos() as f64);
    }
    let avg1 = times1.iter().sum::<f64>() / runs as f64;
    println!("  原始 RustMesh: {:.0} ns ({:.3} ns/顶点)", avg1, avg1 / n_vertices as f64);

    // High-perf mesh - need to create one
    let mut hp = rustmesh::HighPerfMesh::new();
    hp.reserve(n_vertices, n_faces);
    
    // Copy vertices
    for v in orig_mesh.vertices() {
        if let Some(p) = orig_mesh.point(v) {
            hp.add_vertex(p.x, p.y, p.z);
        }
    }
    
    let mut times2 = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let mut sum = 0.0f32;
        let ptr = hp.x_ptr();
        for i in 0..hp.n_vertices() {
            unsafe {
                sum += *ptr.add(i);
            }
            black_box(sum);
        }
        times2.push(start.elapsed().as_nanos() as f64);
    }
    let avg2 = times2.iter().sum::<f64>() / runs as f64;
    println!("  高性能 Mesh:  {:.0} ns ({:.3} ns/顶点)", avg2, avg2 / n_vertices as f64);
    println!("  vs 原始: {:.1}x 更快", avg1 / avg2);

    // ========== Test 2: Bounding Box ==========
    println!("\n[2] 包围盒计算");
    println!("----------------------------------------");

    // Original mesh
    let mut times3 = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let mut min_x = 0.0f32;
        let mut max_x = 0.0f32;
        for v in orig_mesh.vertices() {
            if let Some(p) = orig_mesh.point(v) {
                if p.x < min_x { min_x = p.x; }
                if p.x > max_x { max_x = p.x; }
            }
        }
        black_box((min_x, max_x));
        times3.push(start.elapsed().as_nanos() as f64);
    }
    let avg3 = times3.iter().sum::<f64>() / runs as f64;
    println!("  原始 RustMesh: {:.0} ns", avg3);

    // High-perf mesh
    let mut times4 = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let (min_x, _, _, max_x, _, _) = hp.bounding_box();
        black_box((min_x, max_x));
        times4.push(start.elapsed().as_nanos() as f64);
    }
    let avg4 = times4.iter().sum::<f64>() / runs as f64;
    println!("  高性能 Mesh:  {:.0} ns", avg4);
    println!("  vs 原始: {:.1}x 更快", avg3 / avg4);

    // ========== Test 3: Centroid ==========
    println!("\n[3] 质心计算");
    println!("----------------------------------------");

    // Original mesh
    let mut times5 = Vec::new();
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
        let n = orig_mesh.n_vertices() as f32;
        black_box((sum_x / n, sum_y / n, sum_z / n));
        times5.push(start.elapsed().as_nanos() as f64);
    }
    let avg5 = times5.iter().sum::<f64>() / runs as f64;
    println!("  原始 RustMesh: {:.0} ns", avg5);

    // High-perf mesh
    let mut times6 = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let _ = hp.centroid();
        times6.push(start.elapsed().as_nanos() as f64);
    }
    let avg6 = times6.iter().sum::<f64>() / runs as f64;
    println!("  高性能 Mesh:  {:.0} ns", avg6);
    println!("  vs 原始: {:.1}x 更快", avg5 / avg6);

    // ========== OpenMesh Comparison ==========
    println!("\n========================================");
    println!("OpenMesh 对比 (外推到 {} 顶点)", n_vertices);
    println!("========================================");

    let openmesh_v_iter = 291.0 / 1000.0 * n_vertices as f64;
    let openmesh_bbox = 50000.0 / 1000.0 * n_vertices as f64;  // Estimate
    let openmesh_centroid = 30000.0 / 1000.0 * n_vertices as f64;  // Estimate

    println!("\n[1] 顶点遍历:");
    println!("  OpenMesh (估计): {:.0} ns", openmesh_v_iter);
    println!("  RustMesh 高性能: {:.0} ns", avg2);
    let ratio1 = avg2 / openmesh_v_iter;
    println!("  差距: {:.1}x", ratio1);

    println!("\n[2] 包围盒:");
    println!("  OpenMesh (估计): {:.0} ns", openmesh_bbox);
    println!("  RustMesh 高性能: {:.0} ns", avg4);
    let ratio2 = avg4 / openmesh_bbox;
    println!("  差距: {:.1}x", ratio2);

    println!("\n[3] 质心:");
    println!("  OpenMesh (估计): {:.0} ns", openmesh_centroid);
    println!("  RustMesh 高性能: {:.0} ns", avg6);
    let ratio3 = avg6 / openmesh_centroid;
    println!("  差距: {:.1}x", ratio3);

    // ========== Summary ==========
    println!("\n========================================");
    println!("性能总结");
    println!("========================================");

    println!("\n原始 vs 高性能优化:");
    println!("  顶点遍历: {:.1}x 提升", avg1 / avg2);
    println!("  包围盒:   {:.1}x 提升", avg3 / avg4);
    println!("  质心:     {:.1}x 提升", avg5 / avg6);

    println!("\n高性能 vs OpenMesh:");
    let avg_openmesh = (openmesh_v_iter + openmesh_bbox + openmesh_centroid) / 3.0;
    let avg_hp = (avg2 + avg4 + avg6) / 3.0;
    let avg_ratio = avg_hp / avg_openmesh;
    println!("  平均差距: {:.1}x", avg_ratio);

    println!("\n结论:");
    if avg_ratio < 1.5 {
        println!("  RustMesh 高性能结构接近 OpenMesh!");
    } else if avg_ratio < 3.0 {
        println!("  差距 {:.1}x，可以接受", avg_ratio);
    } else {
        println!("  差距 {:.1}x，需要继续优化", avg_ratio);
    }
}
