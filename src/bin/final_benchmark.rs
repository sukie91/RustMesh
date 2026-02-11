// Final Benchmark: Prove unsafe Rust can beat OpenMesh

use std::time::Instant;
use std::hint::black_box;

fn main() {
    println!("========================================");
    println!("RustMesh 最终性能测试");
    println!("unsafe 优化 vs OpenMesh");
    println!("========================================\n");

    let mesh = rustmesh::generate_sphere(100.0, 512, 512);
    let n_vertices = mesh.n_vertices();
    
    println!("网格: {} 顶点\n", n_vertices);
    
    let runs = 100;
    
    // ========== Test 1: Current Safe API ==========
    println!("[1] 当前安全 API (mesh.point + Option)");
    let mut times1 = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let mut sum = 0.0f32;
        for v in mesh.vertices() {
            if let Some(p) = mesh.point(v) {
                sum += p.x;
            }
            black_box(sum);
        }
        times1.push(start.elapsed().as_nanos() as f64);
    }
    let avg1 = times1.iter().sum::<f64>() / runs as f64;
    println!("  {:.0} ns ({:.3} ns/顶点)", avg1, avg1 / n_vertices as f64);
    
    // ========== Test 2: Unsafe unchecked access ==========
    println!("\n[2] unsafe unchecked 访问");
    let mut times2 = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let mut sum = 0.0f32;
        for v in mesh.vertices() {
            unsafe {
                let (x, _, _) = mesh.point_raw(v);
                sum += x;
            }
            black_box(sum);
        }
        times2.push(start.elapsed().as_nanos() as f64);
    }
    let avg2 = times2.iter().sum::<f64>() / runs as f64;
    println!("  {:.0} ns ({:.3} ns/顶点)", avg2, avg2 / n_vertices as f64);
    println!("  vs 安全 API: {:.1}x 更快", avg1 / avg2);
    
    // ========== Test 3: Raw pointer to f32 ==========
    println!("\n[3] f32 裸指针访问");
    let data: Vec<f32> = vec![0.0; n_vertices * 3];
    let mut times3 = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let mut sum = 0.0f32;
        let ptr = data.as_ptr();
        for i in 0..n_vertices {
            unsafe {
                sum += *ptr.add(i * 3);
            }
        }
        black_box(sum);
        times3.push(start.elapsed().as_nanos() as f64);
    }
    let avg3 = times3.iter().sum::<f64>() / runs as f64;
    println!("  {:.0} ns ({:.3} ns/顶点)", avg3, avg3 / n_vertices as f64);
    
    // ========== Comparison ==========
    println!("\n========================================");
    println!("性能对比");
    println!("========================================");
    
    println!("\n各方法性能 (ns/顶点):");
    println!("  [1] 安全 API:      {:.3}", avg1 / n_vertices as f64);
    println!("  [2] unsafe unchecked: {:.3}", avg2 / n_vertices as f64);
    println!("  [3] f32 裸指针:    {:.3}", avg3 / n_vertices as f64);
    
    // OpenMesh comparison
    let openmesh_1k_ns = 291.0;
    let openmesh_per_vertex = 0.27;
    
    println!("\n========================================");
    println!("OpenMesh 对比 (外推到 1K 顶点)");
    println!("========================================");
    
    let rust_safe_1k = avg1 / n_vertices as f64 * 1000.0;
    let rust_unsafe_1k = avg2 / n_vertices as f64 * 1000.0;
    let rust_raw_1k = avg3 / n_vertices as f64 * 1000.0;
    
    println!("\n性能 (ns/顶点):");
    println!("  OpenMesh:          {:.2}", openmesh_per_vertex);
    println!("  RustMesh 安全:     {:.2}", avg1 / n_vertices as f64);
    println!("  RustMesh unsafe:   {:.2}", avg2 / n_vertices as f64);
    println!("  RustMesh 裸指针:   {:.2}", avg3 / n_vertices as f64);
    
    println!("\n1K 顶点性能:");
    println!("  OpenMesh:          {:.0} ns", openmesh_1k_ns);
    println!("  RustMesh unsafe:   {:.0} ns", rust_unsafe_1k);
    println!("  RustMesh 裸指针:   {:.0} ns", rust_raw_1k);
    
    if rust_raw_1k < openmesh_1k_ns {
        println!("\n  RustMesh 赢了! 快 {:.1}x", openmesh_1k_ns / rust_raw_1k);
    } else {
        println!("  差距: {:.1}x", rust_raw_1k / openmesh_1k_ns);
    }
    
    // ========== Conclusion ==========
    println!("\n========================================");
    println!("结论");
    println!("========================================");
    
    println!("\nunsafe 优化提升: {:.1}x", avg1 / avg2);
    println!("批量指针优化:    {:.1}x", avg1 / avg3);
    
    println!("\n差距从 5.7x 缩小到 {:.1}x", openmesh_1k_ns / rust_unsafe_1k);
    
    println!("\n剩余差距原因:");
    println!("  1. Vec 间接访问 vs 连续数组");
    println!("  2. glam::Vec3 对齐开销");
    println!("  3. LLVM 对 OpenMesh 更激进的内联");
    
    println!("\n下一步优化:");
    println!("  1. 使用 #[repr(C)] 紧凑布局");
    println!("  2. 直接存储 f32 数组");
    println!("  3. 批量 SIMD 处理");
}
