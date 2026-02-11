//! # Benchmarks
//!
//! Performance benchmarks for RustMesh.

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn generate_cube_benchmark(c: &mut Criterion) {
    use rustmesh::generate_cube;

    c.bench_function("generate_cube", |b| {
        b.iter(|| generate_cube())
    });
}

fn generate_sphere_benchmark(c: &mut Criterion) {
    use rustmesh::generate_sphere;

    c.bench_function("generate_sphere_16x16", |b| {
        b.iter(|| generate_sphere(black_box(1.0), black_box(16), black_box(16)))
    });
}

fn generate_icosahedron_benchmark(c: &mut Criterion) {
    use rustmesh::generate_icosahedron;

    c.bench_function("generate_icosahedron", |b| {
        b.iter(|| generate_icosahedron())
    });
}

fn generate_torus_benchmark(c: &mut Criterion) {
    use rustmesh::generate_torus;

    c.bench_function("generate_torus_24x12", |b| {
        b.iter(|| generate_torus(black_box(2.0), black_box(0.5), black_box(24), black_box(12)))
    });
}

fn mesh_traversal_benchmark(c: &mut Criterion) {
    use rustmesh::generate_sphere;

    let mesh = generate_sphere(1.0, 32, 32);

    c.bench_function("traverse_vertices_32x32", |b| {
        b.iter(|| {
            let mut count = 0;
            for _ in mesh.vertices() {
                count += 1;
            }
            count
        })
    });

    c.bench_function("traverse_edges_32x32", |b| {
        b.iter(|| {
            let mut count = 0;
            for _ in mesh.edges() {
                count += 1;
            }
            count
        })
    });

    c.bench_function("traverse_faces_32x32", |b| {
        b.iter(|| {
            let mut count = 0;
            for _ in mesh.faces() {
                count += 1;
            }
            count
        })
    });
}

fn io_benchmark(c: &mut Criterion) {
    use rustmesh::{generate_cube, write_mesh};
    use std::fs::File;
    use std::io::BufWriter;

    let mesh = generate_cube();
    let file = File::create("/tmp/bench_cube.obj").unwrap();
    let writer = BufWriter::new(file);

    c.bench_function("write_cube_obj", |b| {
        b.iter(|| {
            let file = File::create("/tmp/bench_cube.obj").unwrap();
            let writer = BufWriter::new(file);
            write_mesh(&mesh, writer);
        })
    });
}

fn subdivision_benchmark(c: &mut Criterion) {
    use rustmesh::{generate_cube, Subdivider, SubdivideType};

    let mut mesh = generate_cube();
    let subdivider = Subdivider::new();

    c.bench_function("subdivide_cube_midpoint_1", |b| {
        b.iter(|| {
            let mut m = generate_cube();
            subdivider.subdivide(&mut m, SubdivideType::Midpoint);
        })
    });
}

fn smoothing_benchmark(c: &mut Criterion) {
    use rustmesh::{generate_noisy_sphere, Smoother};

    let mut mesh = generate_noisy_sphere(1.0, 0.1, 16, 16);
    let mut smoother = Smoother::new();

    c.bench_function("smooth_noisy_sphere_5iter", |b| {
        b.iter(|| {
            let mut m = generate_noisy_sphere(1.0, 0.1, 16, 16);
            let mut s = Smoother::new();
            s.smooth(&mut m);
        })
    });
}

criterion_group!(
    benches,
    generate_cube_benchmark,
    generate_sphere_benchmark,
    generate_icosahedron_benchmark,
    generate_torus_benchmark,
    mesh_traversal_benchmark,
    io_benchmark,
    subdivision_benchmark,
    smoothing_benchmark,
);

criterion_main!(benches);
