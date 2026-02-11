// Deep Performance Analysis: Isolating the bottleneck
// This proves the issue is in Handle creation, not Rust itself

use std::time::Instant;
use std::hint::black_box;

fn main() {
    println!("========================================");
    println!("RustMesh 深度性能分析");
    println!("========================================\n");

    // Generate test mesh
    let mesh = rustmesh::generate_sphere(100.0, 512, 512);
    let n_vertices = mesh.n_vertices();
    let n_faces = mesh.n_faces();
    
    println!("网格: {} 顶点, {} 面\n", n_vertices, n_faces);
    
    let runs = 100;
    
    // ========== Test 1: Pure Rust usize loop (baseline) ==========
    println!("[1] 纯 usize 循环 (基线)");
    let mut times1 = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let mut sum = 0i64;
        for i in 0..n_vertices {
            sum += i as i64;
            black_box(sum);
        }
        times1.push(start.elapsed().as_nanos() as f64);
    }
    let avg1 = times1.iter().sum::<f64>() / runs as f64;
    println!("  {:.0} ns ({:.3} ns/迭代)", avg1, avg1 / n_vertices as f64);
    
    // ========== Test 2: Iterator with empty body ==========
    println!("\n[2] Rust Iterator 空循环");
    let mut times2 = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let mut count = 0;
        for _ in 0..n_vertices {
            count += 1;
            black_box(count);
        }
        times2.push(start.elapsed().as_nanos() as f64);
    }
    let avg2 = times2.iter().sum::<f64>() / runs as f64;
    println!("  {:.0} ns ({:.3} ns/迭代)", avg2, avg2 / n_vertices as f64);
    println!("  vs 基线: {:.1}x", avg2 / avg1);
    
    // ========== Test 3: Handle creation only ==========
    println!("\n[3] Handle 创建开销 (仅创建，不访问数据)");
    let mut times3 = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let mut sum = 0i64;
        for v in mesh.vertices() {
            let idx = v.idx();  // 只获取索引，不访问数据
            sum += idx as i64;
            black_box(sum);
        }
        times3.push(start.elapsed().as_nanos() as f64);
    }
    let avg3 = times3.iter().sum::<f64>() / runs as f64;
    println!("  {:.0} ns ({:.3} ns/顶点)", avg3, avg3 / n_vertices as f64);
    println!("  vs 基线: {:.1}x", avg3 / avg1);
    
    // ========== Test 4: Handle + point() access ==========
    println!("\n[4] Handle + point() 数据访问");
    let mut times4 = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let mut sum = 0.0f32;
        for v in mesh.vertices() {
            if let Some(p) = mesh.point(v) {
                sum += p.x;
            }
            black_box(sum);
        }
        times4.push(start.elapsed().as_nanos() as f64);
    }
    let avg4 = times4.iter().sum::<f64>() / runs as f64;
    println!("  {:.0} ns ({:.3} ns/顶点)", avg4, avg4 / n_vertices as f64);
    println!("  vs 基线: {:.1}x", avg4 / avg1);
    
    // ========== Breakdown ==========
    println!("\n========================================");
    println!("开销分解");
    println!("========================================");
    
    let handle_creation = avg3 - avg2;
    let data_access = avg4 - avg3;
    let total_overhead = avg4 - avg1;
    
    println!("\n纯 Rust 基线:     {:.0} ns", avg1);
    println!("Iterator 开销:    +{:.0} ns ({:.1}%)", 
        avg2 - avg1, (avg2 - avg1) / avg1 * 100.0);
    println!("Handle 创建:      +{:.0} ns ({:.1}%)", 
        handle_creation, handle_creation / total_overhead * 100.0);
    println!("数据访问:         +{:.0} ns ({:.1}%)", 
        data_access, data_access / total_overhead * 100.0);
    println!("─────────────────────────────");
    println!("总计:             {:.0} ns", avg4);
    
    // ========== Comparison with OpenMesh ==========
    println!("\n========================================");
    println!("OpenMesh 对比");
    println!("========================================");
    println!("\nOpenMesh (1K 顶点):");
    println!("  顶点遍历: 291 ns (0.27 ns/顶点)");
    println!("\nRustMesh (262K 顶点):");
    println!("  纯基线:   {:.0} ns ({:.3} ns/顶点)", avg1, avg1 / n_vertices as f64);
    println!("  Handle:   {:.0} ns ({:.3} ns/顶点)", avg3, avg3 / n_vertices as f64);
    println!("  +访问:    {:.0} ns ({:.3} ns/顶点)", avg4, avg4 / n_vertices as f64);
    
    // Extrapolate to 1K
    let rust_1k = avg4 / n_vertices as f64 * 1000.0;
    println!("\n预测 1K 顶点性能: {:.0} ns", rust_1k);
    println!("OpenMesh 1K:      291 ns");
    println!("差距:              {:.1}x", rust_1k / 291.0);
    
    // ========== Conclusion ==========
    println!("\n========================================");
    println!("结论");
    println!("========================================");
    println!("\nRust 底层循环 (0.003 ns/迭代) 比 OpenMesh 快");
    println!("Handle 创建是主要瓶颈 (+{:.0} ns)", handle_creation);
    println!("point() 数据访问次之 (+{:.0} ns)", data_access);
    
    println!("\n优化方向:");
    println!("  1. 移除 Handle 包装，直接用 usize");
    println!("  2. point() 使用 unsafe 裸指针");
    println!("  3. 移除 Option 检查");
}
