// Performance Benchmark: Handle-based vs Index-based Iteration

use std::time::Instant;
use std::hint::black_box;

fn main() {
    println!("========================================");
    println!("RustMesh 迭代器性能对比");
    println!("Handle-based vs Index-based");
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
        
        let runs = 10;
        
        // === Handle-based iteration (标准 API) ===
        let mut handle_v_times = Vec::new();
        for _ in 0..runs {
            let start = Instant::now();
            let mut sum = 0i32;
            for v in mesh.vertices() {
                if let Some(p) = mesh.point(v) {
                    sum += (p.x + p.y + p.z) as i32;
                }
                black_box(sum);
            }
            handle_v_times.push(start.elapsed().as_nanos() as f64);
        }
        let handle_v_avg = handle_v_times.iter().sum::<f64>() / runs as f64;
        
        // === Index-based iteration (高性能) ===
        let mesh_clone = mesh.clone();
        let mut index_v_times = Vec::new();
        for _ in 0..runs {
            let start = Instant::now();
            let mut sum = 0i32;
            for idx in mesh_clone.vertex_indices() {
                if let Some(v) = mesh_clone.vertex_point(idx) {
                    sum += (v.point.x + v.point.y + v.point.z) as i32;
                }
                black_box(sum);
            }
            index_v_times.push(start.elapsed().as_nanos() as f64);
        }
        let index_v_avg = index_v_times.iter().sum::<f64>() / runs as f64;
        
        // === Face iteration comparison ===
        let mut handle_f_times = Vec::new();
        for _ in 0..runs {
            let start = Instant::now();
            let mut count = 0;
            for _ in mesh.faces() {
                count += 1;
                black_box(count);
            }
            handle_f_times.push(start.elapsed().as_nanos() as f64);
        }
        let handle_f_avg = handle_f_times.iter().sum::<f64>() / runs as f64;
        
        let mesh_clone = mesh.clone();
        let mut index_f_times = Vec::new();
        for _ in 0..runs {
            let start = Instant::now();
            let mut count = 0;
            for _ in mesh_clone.face_indices() {
                count += 1;
                black_box(count);
            }
            index_f_times.push(start.elapsed().as_nanos() as f64);
        }
        let index_f_avg = index_f_times.iter().sum::<f64>() / runs as f64;
        
        println!("[{} 网格] {} 顶点, {} 面", name, n_vertices, n_faces);
        println!("  --- 顶点迭代 ---");
        println!("  Handle-based: {:.0} ns ({:.2} ns/顶点)", handle_v_avg, handle_v_avg / n_vertices as f64);
        println!("  Index-based:  {:.0} ns ({:.2} ns/顶点)", index_v_avg, index_v_avg / n_vertices as f64);
        if index_v_avg < handle_v_avg {
            println!("  ⚡ Index 快了 {:.1f}%", (handle_v_avg - index_v_avg) / handle_v_avg * 100.0);
        }
        println!("  --- 面片迭代 ---");
        println!("  Handle-based: {:.0} ns ({:.2} ns/面)", handle_f_avg, handle_f_avg / n_faces as f64);
        println!("  Index-based:  {:.0} ns ({:.2} ns/面)", index_f_avg, index_f_avg / n_faces as f64);
        if index_f_avg < handle_f_avg {
            println!("  ⚡ Index 快了 {:.1f}%", (handle_f_avg - index_f_avg) / handle_f_avg * 100.0);
        }
        println!();
    }

    // === OpenMesh 对比 ===
    println!("========================================");
    println!("OpenMesh 参考性能");
    println!("========================================");
    println!("  顶点迭代: 291 ns (1K 顶点)");
    println!("  面片迭代: 84 ns (2K 面)");
}
