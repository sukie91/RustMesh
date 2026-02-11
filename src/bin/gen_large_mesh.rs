// Large Mesh Generator - 50万+ 面片
// 用法: cargo run --bin gen_large_mesh

use rustmesh::generate_sphere;

fn main() {
    println!("========================================");
    println!("生成大规模测试网格");
    println!("========================================\n");

    // 生成不同规模的网格
    let scales = [
        ("小型", 64, 64),
        ("中型", 128, 128),
        ("大型", 256, 256),
        ("超大型", 512, 512),
        ("极大型", 1024, 1024),
    ];

    for (name, rings, segments) in scales {
        println!("测试 {} 网格 ({}×{})...", name, rings, segments);
        
        let mesh = generate_sphere(100.0, segments, rings);
        let n_vertices = mesh.n_vertices();
        let n_faces = mesh.n_faces();
        
        println!("  顶点数: {:>10}", n_vertices);
        println!("  面片数: {:>10}", n_faces);
        println!();
    }

    println!("========================================");
    println!("测试完成！");
    println!("========================================");
    println!("\n生成真实网格文件需要额外的面索引逻辑。");
    println!("建议: 使用 Python 或 MeshLab 生成大模型文件。");
}
