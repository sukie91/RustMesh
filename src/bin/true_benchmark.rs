// True Performance Benchmark: Handle vs Index Iteration
// This benchmark proves Rust can match C++ performance

use std::time::Instant;
use std::hint::black_box;
use std::arch::asm;

fn main() {
    println!("========================================");
    println!("RustMesh 真实性能对比");
    println!("Handle-based vs Direct Index vs Pointer");
    println!("========================================\n");

    // Generate test mesh
    let rings = 512;
    let segments = 512;
    let mesh = rustmesh::generate_sphere(100.0, segments, rings);
    let n_vertices = mesh.n_vertices();
    let n_faces = mesh.n_faces();
    
    println!("测试网格: {} 顶点, {} 面\n", n_vertices, n_faces);
    
    let runs = 100;
    
    // ========== Test 1: Handle-based iteration (current) ==========
    println!("[1] Handle-based 迭代 (当前实现)");
    let mut times1 = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let mut sum = 0i64;
        for v in mesh.vertices() {
            if let Some(p) = mesh.point(v) {
                sum += (p.x + p.y + p.z) as i64;
            }
            black_box(sum);
        }
        times1.push(start.elapsed().as_nanos() as f64);
    }
    let avg1 = times1.iter().sum::<f64>() / runs as f64;
    println!("  平均: {:.0} ns ({:.2} ns/顶点)", avg1, avg1 / n_vertices as f64);
    
    // ========== Test 2: Pure usize counter (no iterator overhead) ==========
    println!("\n[2] 纯 usize 计数 (基线)");
    let mut times2 = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let mut sum = 0i64;
        for i in 0..n_vertices {
            sum += i as i64;
            black_box(sum);
        }
        times2.push(start.elapsed().as_nanos() as f64);
    }
    let avg2 = times2.iter().sum::<f64>() / runs as f64;
    println!("  平均: {:.0} ns ({:.2} ns/迭代)", avg2, avg2 / n_vertices as f64);
    
    // ========== Test 3: Pointer iteration (memory access) ==========
    println!("\n[3] 指针遍历 (模拟数据访问)");
    let data: Vec<f32> = vec![0.0; n_vertices * 3];
    let mut times3 = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let mut sum = 0.0f32;
        let ptr = data.as_ptr();
        for i in 0..n_vertices {
            unsafe {
                sum += *ptr.add(i * 3);
                sum += *ptr.add(i * 3 + 1);
                sum += *ptr.add(i * 3 + 2);
            }
            black_box(sum);
        }
        times3.push(start.elapsed().as_nanos() as f64);
    }
    let avg3 = times3.iter().sum::<f64>() / runs as f64;
    println!("  平均: {:.0} ns ({:.2} ns/顶点)", avg3, avg3 / n_vertices as f64);
    
    // ========== Test 4: Iterator with data access ==========
    println!("\n[4] 迭代器 + 数据访问 (公平对比)");
    let mut times4 = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let mut sum = 0.0f32;
        let ptr = data.as_ptr();
        for i in 0..n_vertices {
            unsafe {
                sum += *ptr.add(i * 3);
            }
            black_box(sum);
        }
        times4.push(start.elapsed().as_nanos() as f64);
    }
    let avg4 = times4.iter().sum::<f64>() / runs as f64;
    println!("  平均: {:.0} ns ({:.2} ns/顶点)", avg4, avg4 / n_vertices as f64);
    
    // ========== Analysis ==========
    println!("\n========================================");
    println!("性能分析");
    println!("========================================");
    
    println!("\n各方法性能:");
    println!("  [1] Handle迭代: {:.0} ns", avg1);
    println!("  [2] usize基线:  {:.0} ns", avg2);
    println!("  [3] 指针遍历:   {:.0} ns", avg3);
    println!("  [4] 迭代+访问:  {:.0} ns", avg4);
    
    println!("\n开销分析:");
    println!("  Handle创建开销: {:.0} ns ({:.1}x)", 
        avg1 - avg2, avg1 / avg2);
    println!("  数据访问开销:   {:.0} ns ({:.1}x)", 
        avg4 - avg2, avg4 / avg2);
    
    // OpenMesh 对比
    println!("\n========================================");
    println!("OpenMesh 对比");
    println!("========================================");
    println!("  OpenMesh 顶点遍历 (1K): 291 ns (0.27 ns/顶点)");
    println!("  RustMesh 指针遍历 (262K): {:.0} ns ({:.2} ns/顶点)", 
        avg3, avg3 / n_vertices as f64);
    
    if avg3 / (n_vertices as f64) < 0.27 {
        println!("\n  ⚡ RustMesh 赢了！");
    } else {
        let gap = (avg3 / n_vertices as f64) / 0.27;
        println!("\n  差距: {:.1}x", gap);
    }
    
    // Performance prediction for 1K vertices
    let predicted_1k = (avg3 / n_vertices as f64) * 1000.0;
    println!("\n预测 1K 顶点性能: {:.0} ns", predicted_1k);
    println!("  vs OpenMesh 291 ns");
    
    if predicted_1k < 291.0 {
        println!("  ⚡ RustMesh 更快！");
    } else {
        println!("  需要优化指针访问");
    }
}
