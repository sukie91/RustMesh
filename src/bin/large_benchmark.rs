// Large Mesh Performance Benchmark - 修复版
// 使用 black_box 防止编译器优化

use std::time::Instant;
use std::hint::black_box;

fn main() {
    println!("========================================");
    println!("大规模网格性能测试 - 修复版");
    println!("========================================\n");

    // 测试不同规模的网格
    let scales = [
        ("50万", 512, 512),
        ("100万", 724, 724),
        ("200万", 1024, 1024),
    ];

    for (name, rings, segments) in scales {
        println!("[{} 网格 {}×{}]", name, rings, segments);
        println!("----------------------------------------");

        // 生成网格
        println!("生成中...");
        let start = Instant::now();
        let mesh = rustmesh::generate_sphere(100.0, segments, rings);
        let gen_time = start.elapsed();
        
        let n_vertices = mesh.n_vertices();
        let n_faces = mesh.n_faces();
        let n_edges = mesh.n_edges();
        
        println!("  生成时间: {:?}", gen_time);
        println!("  顶点数: {}", n_vertices);
        println!("  面片数: {}", n_faces);
        println!("  边数: {}", n_edges);
        println!();

        // 遍历测试 - 使用 black_box 防止优化
        println!("遍历测试 (使用 black_box 防止优化):");
        
        // 顶点遍历
        let start = Instant::now();
        let mut v_count: usize = 0;
        for v in mesh.vertices() {
            let _p = mesh.point(v);
            black_box(_p);
            v_count += 1;
        }
        let v_time = start.elapsed();
        println!("  顶点遍历: {:?}", v_time);
        println!("  实际遍历: {} 顶点 (期望: {})", v_count, n_vertices);
        
        // 重新生成 mesh 用于面遍历
        let mesh = rustmesh::generate_sphere(100.0, segments, rings);
        
        // 面片遍历
        let start = Instant::now();
        let mut f_count: usize = 0;
        for f in mesh.faces() {
            f_count += 1;
            black_box(f);
        }
        let f_time = start.elapsed();
        println!("  面片遍历: {:?}", f_time);
        println!("  实际遍历: {} 面片 (期望: {})", f_count, n_faces);
        
        // 重新生成 mesh 用于边遍历
        let mesh = rustmesh::generate_sphere(100.0, segments, rings);
        
        // 边遍历
        let start = Instant::now();
        let mut e_count: usize = 0;
        for _e in mesh.edges() {
            e_count += 1;
        }
        let e_time = start.elapsed();
        println!("  边遍历: {:?}", e_time);
        println!("  实际遍历: {} 边 (期望: {})", e_count, n_edges);
        println!();

        // 计算性能
        println!("性能指标:");
        if v_time.as_nanos() > 0 {
            let v_per_sec = n_vertices as f64 / (v_time.as_nanos() as f64 / 1_000_000_000.0);
            println!("  顶点/秒: {:.0}", v_per_sec);
        }
        if f_time.as_nanos() > 0 {
            let f_per_sec = n_faces as f64 / (f_time.as_nanos() as f64 / 1_000_000_000.0);
            println!("  面片/秒: {:.0}", f_per_sec);
        }
        println!();
    }

    // 对比 OpenMesh
    println!("========================================");
    println!("OpenMesh 参考 (来自之前的测试)");
    println!("========================================");
    println!("50万面:");
    println!("  顶点遍历: ~291 ns");
    println!("  面片遍历: ~84 ns");
}
