// Fair Benchmark - Pure Iterator Performance
// 只测试迭代器开销，不测试数据访问

use std::time::Instant;
use std::hint::black_box;

fn main() {
    println!("========================================");
    println!("纯迭代器性能测试");
    println!("========================================\n");

    let scales = [
        ("1K", 32, 32),
        ("4K", 64, 64),
        ("16K", 128, 128),
        ("65K", 256, 256),
        ("262K", 512, 512),
    ];

    for (name, rings, segments) in scales {
        let mesh = rustmesh::generate_sphere(100.0, segments, rings);
        let n_vertices = mesh.n_vertices();
        let n_faces = mesh.n_faces();
        let n_edges = mesh.n_edges();
        
        let runs = 10;
        
        // 纯迭代器测试 - 不访问数据
        let mut v_times = Vec::new();
        for _ in 0..runs {
            let start = Instant::now();
            let mut count = 0;
            for _ in mesh.vertices() {
                count += 1;
                black_box(count);
            }
            v_times.push(start.elapsed().as_nanos() as f64);
        }
        let v_avg = v_times.iter().sum::<f64>() / runs as f64;
        
        let mesh = rustmesh::generate_sphere(100.0, segments, rings);
        let mut f_times = Vec::new();
        for _ in 0..runs {
            let start = Instant::now();
            let mut count = 0;
            for _ in mesh.faces() {
                count += 1;
                black_box(count);
            }
            f_times.push(start.elapsed().as_nanos() as f64);
        }
        let f_avg = f_times.iter().sum::<f64>() / runs as f64;
        
        let mesh = rustmesh::generate_sphere(100.0, segments, rings);
        let mut e_times = Vec::new();
        for _ in 0..runs {
            let start = Instant::now();
            let mut count = 0;
            for _ in mesh.edges() {
                count += 1;
                black_box(count);
            }
            e_times.push(start.elapsed().as_nanos() as f64);
        }
        let e_avg = e_times.iter().sum::<f64>() / runs as f64;
        
        println!("[{} 网格] {} 顶点, {} 面, {} 边", name, n_vertices, n_faces, n_edges);
        println!("  顶点迭代: {:.0} ns ({:.1} ns/顶点)", v_avg, v_avg / n_vertices as f64);
        println!("  面片迭代: {:.0} ns ({:.1} ns/面)", f_avg, f_avg / n_faces as f64);
        println!("  边迭代:   {:.0} ns ({:.1} ns/边)", e_avg, e_avg / n_edges as f64);
        println!();
    }

    // OpenMesh 参考
    println!("========================================");
    println!("OpenMesh 参考 (1K 顶点, 2K 面)");
    println!("========================================");
    println!("  顶点迭代: 291 ns (0.27 ns/顶点)");
    println!("  面片迭代: 84 ns (0.04 ns/面)");
}
