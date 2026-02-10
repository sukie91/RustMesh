use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rustmesh::{PolyMesh, Vec3};

fn create_triangle_mesh(n: usize) -> PolyMesh {
    let mut mesh = PolyMesh::new();
    
    // 创建 n x n 网格
    for i in 0..n {
        for j in 0..n {
            let x = i as f32 / n as f32;
            let y = j as f32 / n as f32;
            mesh.add_vertex(Vec3::new(x, y, 0.0));
        }
    }
    
    // 创建面
    for i in 0..(n-1) {
        for j in 0..(n-1) {
            let v0 = i * n + j;
            let v1 = (i + 1) * n + j;
            let v2 = (i + 1) * n + (j + 1);
            let v3 = i * n + (j + 1);
            
            mesh.add_face(&[
                rustmesh::VertexHandle::new(v0 as i32),
                rustmesh::VertexHandle::new(v1 as i32),
                rustmesh::VertexHandle::new(v2 as i32),
            ]);
            
            mesh.add_face(&[
                rustmesh::VertexHandle::new(v0 as i32),
                rustmesh::VertexHandle::new(v2 as i32),
                rustmesh::VertexHandle::new(v3 as i32),
            ]);
        }
    }
    
    mesh
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("create_mesh_10x10", |b| {
        b.iter(|| create_triangle_mesh(black_box(10)))
    });
    
    c.bench_function("create_mesh_50x50", |b| {
        b.iter(|| create_triangle_mesh(black_box(50)))
    });
    
    c.bench_function("vertex_iteration_10x10", |b| {
        let mesh = create_triangle_mesh(10);
        b.iter(|| {
            let mut count = 0;
            for _ in mesh.vertices() {
                count += 1;
            }
            count
        })
    });
    
    c.bench_function("point_access_10x10", |b| {
        let mesh = create_triangle_mesh(10);
        let vertices: Vec<_> = mesh.vertices().collect();
        b.iter(|| {
            let mut sum = Vec3::ZERO;
            for v in &vertices {
                if let Some(p) = mesh.point(*v) {
                    sum += p;
                }
            }
            sum
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
