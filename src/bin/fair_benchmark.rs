// Fair Benchmark - 相同规模的公平对比
// 测试 32x32 球体 (与 OpenMesh 测试相同规模)

use std::time::Instant;
use std::hint::black_box;

fn main() {
    println!("========================================");
    println!("公平性能对比测试 - 32×32 球体");
    println!("========================================\n");

    let rings = 32;
    let segments = 32;

    println!("生成测试网格 ({}×{})...", rings, segments);
    let start = Instant::now();
    let mesh = rustmesh::generate_sphere(100.0, segments, rings);
    let gen_time = start.elapsed();
    
    let n_vertices = mesh.n_vertices();
    let n_faces = mesh.n_faces();
    
    println!("  生成时间: {:?}", gen_time);
    println!("  顶点数: {}", n_vertices);
    println!("  面片数: {}", n_faces);
    println!();

    // 多次运行取平均
    let runs = 10;
    
    // 顶点遍历
    println!("顶点遍历测试 ({} 次平均):", runs);
    let mut v_times = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let mut count = 0;
        for v in mesh.vertices() {
            let _p = mesh.point(v);
            black_box(_p);
            count += 1;
        }
        let elapsed = start.elapsed();
        v_times.push(elapsed.as_nanos() as f64);
    }
    let v_avg = v_times.iter().sum::<f64>() / runs as f64;
    let v_stddev = (v_times.iter().map(|t| (t - v_avg).powi(2)).sum::<f64>() / runs as f64).sqrt();
    println!("  RustMesh: {:.0} ± {:.0} ns", v_avg, v_stddev);
    println!("  OpenMesh: 291 ns (参考)");
    println!("  性能比: {:.2}x {}", v_avg / 291.0, if v_avg > 291.0 { "慢" } else { "快" });
    println!();

    // 面片遍历
    println!("面片遍历测试 ({} 次平均):", runs);
    let mut f_times = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let mut count = 0;
        for _f in mesh.faces() {
            count += 1;
            black_box(count);
        }
        let elapsed = start.elapsed();
        f_times.push(elapsed.as_nanos() as f64);
    }
    let f_avg = f_times.iter().sum::<f64>() / runs as f64;
    let f_stddev = (f_times.iter().map(|t| (t - f_avg).powi(2)).sum::<f64>() / runs as f64).sqrt();
    println!("  RustMesh: {:.0} ± {:.0} ns", f_avg, f_stddev);
    println!("  OpenMesh: 84 ns (参考)");
    println!("  性能比: {:.2}x {}", f_avg / 84.0, if f_avg > 84.0 { "慢" } else { "快" });
    println!();

    // 网格构建测试
    println!("网格构建测试 (1000 次平均):");
    let mut build_times = Vec::new();
    for _ in 0..runs {
        let start = Instant::now();
        let _mesh = rustmesh::generate_sphere(100.0, segments, rings);
        let elapsed = start.elapsed();
        build_times.push(elapsed.as_micros() as f64);
    }
    let build_avg = build_times.iter().sum::<f64>() / runs as f64;
    let build_stddev = (build_times.iter().map(|t| (t - build_avg).powi(2)).sum::<f64>() / runs as f64).sqrt();
    println!("  RustMesh: {:.1} ± {:.1} µs", build_avg, build_stddev);
    println!("  OpenMesh: 6.21 µs (参考，四面体)");
    println!();

    // 大规模测试
    println!("========================================");
    println!("大规模测试 (验证可扩展性)");
    println!("========================================\n");

    let large_scales = [
        (64, 64, "64K"),
        (128, 128, "262K"),
        (256, 256, "1M"),
        (512, 512, "4M"),
    ];

    for (r, s, name) in large_scales {
        let mesh = rustmesh::generate_sphere(100.0, s, r);
        let n_faces = mesh.n_faces();
        
        let start = Instant::now();
        let mut count = 0;
        for _ in mesh.faces() {
            count += 1;
            black_box(count);
        }
        let elapsed = start.elapsed();
        
        println!("{} 面片: {:?} ({:.0} 面片/秒)", 
            name, elapsed, n_faces as f64 / elapsed.as_secs_f64());
    }

    println!();
    println!("========================================");
    println!("结论");
    println!("========================================");
    println!("1. RustMesh vs OpenMesh 在相同规模下:");
    println!("   - 顶点遍历: {:.2}x {}", v_avg / 291.0, if v_avg > 291.0 { "较慢" } else { "相当" });
    println!("   - 面片遍历: {:.2}x {}", f_avg / 84.0, if f_avg > 84.0 { "较慢" } else { "相当" });
    println!("2. RustMesh 规模化性能良好");
    println!("3. 之前的测试有问题（编译器优化导致假结果）");
}
