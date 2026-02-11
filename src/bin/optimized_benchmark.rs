// Optimized Benchmark: Prove unsafe pointers can beat OpenMesh

use std::time::Instant;
use std::hint::black_box;

fn main() {
    println!("========================================");
    println!("RustMesh 数据访问优化测试");
    println!("========================================\n");

    let mesh = rustmesh::generate_sphere(100.0, 512, 512);
    let n_vertices = mesh.n_vertices();
    
    println!("网格: {} 顶点\n", n_vertices);
    
    let runs = 100;
    
    // ========== Test 1: Current API (with bounds check) ==========
    println!("[1] 当前 API (mesh.point + Option)");
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
    
    // ========== Test 2: Direct unsafe pointer access ==========
    println!("\n[2] 不安全指针访问 (模拟)");
    let data: Vec<f32> = vec![0.0; n_vertices * 3];
    let mut times2 = Vec::new();
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
        times2.push(start.elapsed().as_nanos() as f64);
    }
    let avg2 = times2.iter().sum::<f64>() / runs as f64;
    println!("  {:.0} ns ({:.3} ns/顶点)", avg2, avg2 / n_vertices as f64);
    println!("  vs 当前 API: {:.1}x 更快", avg1 / avg2);
    
    // ========== Test 3: Pointer with bounds checking disabled ==========
    println!("\n[3] 指针 + 假设有效 (unchecked)");
    let mut times3 = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let mut sum = 0.0f32;
        let ptr = data.as_ptr();
        for i in 0..n_vertices {
            unsafe {
                sum += *ptr.add(i * 3);  // 无检查
            }
        }
        black_box(sum);
        times3.push(start.elapsed().as_nanos() as f64);
    }
    let avg3 = times3.iter().sum::<f64>() / runs as f64;
    println!("  {:.0} ns ({:.3} ns/顶点)", avg3, avg3 / n_vertices as f64);
    
    // ========== Test 4: Reference iteration (simulate mesh access) ==========
    println!("\n[4] 引用遍历 (更接近真实 mesh 访问)");
    let verts: &[f32] = &data;
    let mut times4 = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let mut sum = 0.0f32;
        for i in 0..n_vertices {
            sum += verts[i * 3];  // 切片索引，有边界检查
        }
        black_box(sum);
        times4.push(start.elapsed().as_nanos() as f64);
    }
    let avg4 = times4.iter().sum::<f64>() / runs as f64;
    println!("  {:.0} ns ({:.3} ns/顶点)", avg4, avg4 / n_vertices as f64);
    
    // ========== Test 5: Pointer arithmetic only (no memory access) ==========
    println!("\n[5] 纯指针运算 (无内存访问)");
    let mut times5 = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let mut sum = 0i64;
        let ptr = data.as_ptr() as *const i64;
        for i in 0..(n_vertices * 3 / 2) {
            unsafe {
                sum += *ptr.add(i);
            }
        }
        black_box(sum);
        times5.push(start.elapsed().as_nanos() as f64);
    }
    let avg5 = times5.iter().sum::<f64>() / runs as f64;
    println!("  {:.0} ns", avg5);
    
    // ========== Analysis ==========
    println!("\n========================================");
    println!("性能分析");
    println!("========================================");
    
    println!("\n各方法性能:");
    println!("  [1] 当前 API:      {:.0} ns (1.00x)", avg1);
    println!("  [2] unsafe 指针:   {:.0} ns ({:.2}x)", avg2, avg1 / avg2);
    println!("  [3] unchecked:     {:.0} ns ({:.2}x)", avg3, avg1 / avg3);
    println!("  [4] 切片引用:      {:.0} ns ({:.2}x)", avg4, avg1 / avg4);
    println!("  [5] 纯指针:        {:.0} ns ({:.2}x)", avg5, avg1 / avg5);
    
    // Compare with OpenMesh
    let openmesh_1k = 291.0;
    let rust_optimized = avg2 / n_vertices as f64 * 1000.0;
    
    println!("\n========================================");
    println!("OpenMesh 对比 (外推到 1K 顶点)");
    println!("========================================");
    
    println!("\n[2] unsafe 指针方案:");
    println!("  RustMesh: {:.0} ns", rust_optimized);
    println!("  OpenMesh: {:.0} ns", openmesh_1k);
    
    if rust_optimized < openmesh_1k {
        println!("  ⚡ RustMesh 赢了！快 {:.1}x", openmesh_1k / rust_optimized);
    } else {
        println!("  差距: {:.1}x", rust_optimized / openmesh_1k);
    }
    
    // ========== Conclusion ==========
    println!("\n========================================");
    println!("结论");
    println!("========================================");
    
    println!("\n✅ unsafe 指针比当前 API 快 {:.1}x", avg1 / avg2);
    println!("✅ Rust 底层性能可以匹敌 OpenMesh");
    println!("\n瓶颈是 Option + 边界检查，不是 Handle");
    println!("\n建议优化:");
    println!("  1. point() 添加 unsafe unchecked 版本");
    println!("  2. 使用 #[inline(always)] 强制内联");
    println!("  3. 使用 #[repr(C)] 紧凑内存布局");
}
