// High-Performance SIMD Operations for RustMesh

use std::hint::black_box;

/// SIMD-accelerated vertex sum
#[inline]
pub unsafe fn vertex_sum_simd(
    x: &[f32],
    y: &[f32],
    z: &[f32],
) -> (f32, f32, f32) {
    let n = x.len();
    let mut i = 0;
    let n_simd = (n / 4) * 4;

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        let ptr_x = x.as_ptr();
        let ptr_y = y.as_ptr();
        let ptr_z = z.as_ptr();

        let mut acc_x = vdupq_n_f32(0.0);
        let mut acc_y = vdupq_n_f32(0.0);
        let mut acc_z = vdupq_n_f32(0.0);

        while i < n_simd {
            let vx = vld1q_f32(ptr_x.add(i));
            let vy = vld1q_f32(ptr_y.add(i));
            let vz = vld1q_f32(ptr_z.add(i));
            acc_x = vaddq_f32(acc_x, vx);
            acc_y = vaddq_f32(acc_y, vy);
            acc_z = vaddq_f32(acc_z, vz);
            i += 4;
        }

        let mut sum_x = vaddvq_f32(acc_x);
        let mut sum_y = vaddvq_f32(acc_y);
        let mut sum_z = vaddvq_f32(acc_z);

        while i < n {
            sum_x += *ptr_x.add(i);
            sum_y += *ptr_y.add(i);
            sum_z += *ptr_z.add(i);
            i += 1;
        }

        (sum_x, sum_y, sum_z)
    }

    #[cfg(not(target_arch = "aarch64"))]
    {
        let mut sum_x = 0.0f32;
        let mut sum_y = 0.0f32;
        let mut sum_z = 0.0f32;

        while i < n {
            sum_x += x[i];
            sum_y += y[i];
            sum_z += z[i];
            i += 1;
        }

        (sum_x, sum_y, sum_z)
    }
}

/// SIMD-accelerated bounding box computation
#[inline]
pub unsafe fn bounding_box_simd(
    x: &[f32],
    y: &[f32],
    z: &[f32],
) -> (f32, f32, f32, f32, f32, f32) {
    let n = x.len();
    let mut i = 0;
    let n_simd = (n / 4) * 4;

    if n == 0 {
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }

    let ptr_x = x.as_ptr();
    let ptr_y = y.as_ptr();
    let ptr_z = z.as_ptr();

    let mut min_x = x[0];
    let mut max_x = x[0];
    let mut min_y = y[0];
    let mut max_y = y[0];
    let mut min_z = z[0];
    let mut max_z = z[0];

    #[cfg(target_arch = "aarch64")]
    {
        use std::arch::aarch64::*;

        let mut acc_min_x = vdupq_n_f32(min_x);
        let mut acc_max_x = vdupq_n_f32(max_x);
        let mut acc_min_y = vdupq_n_f32(min_y);
        let mut acc_max_y = vdupq_n_f32(max_y);
        let mut acc_min_z = vdupq_n_f32(min_z);
        let mut acc_max_z = vdupq_n_f32(max_z);

        while i < n_simd {
            let vx = vld1q_f32(ptr_x.add(i));
            let vy = vld1q_f32(ptr_y.add(i));
            let vz = vld1q_f32(ptr_z.add(i));

            acc_min_x = vminq_f32(acc_min_x, vx);
            acc_max_x = vmaxq_f32(acc_max_x, vx);
            acc_min_y = vminq_f32(acc_min_y, vy);
            acc_max_y = vmaxq_f32(acc_max_y, vy);
            acc_min_z = vminq_f32(acc_min_z, vz);
            acc_max_z = vmaxq_f32(acc_max_z, vz);

            i += 4;
        }

        min_x = vminvq_f32(acc_min_x);
        max_x = vmaxvq_f32(acc_max_x);
        min_y = vminvq_f32(acc_min_y);
        max_y = vmaxvq_f32(acc_max_y);
        min_z = vminvq_f32(acc_min_z);
        max_z = vmaxvq_f32(acc_max_z);
    }

    while i < n {
        let vx = *ptr_x.add(i);
        let vy = *ptr_y.add(i);
        let vz = *ptr_z.add(i);

        if vx < min_x { min_x = vx; }
        if vx > max_x { max_x = vx; }
        if vy < min_y { min_y = vy; }
        if vy > max_y { max_y = vy; }
        if vz < min_z { min_z = vz; }
        if vz > max_z { max_z = vz; }

        i += 1;
    }

    (min_x, max_x, min_y, max_y, min_z, max_z)
}

/// SIMD-accelerated centroid computation
#[inline]
pub unsafe fn centroid_simd(
    x: &[f32],
    y: &[f32],
    z: &[f32],
) -> (f32, f32, f32) {
    let (sum_x, sum_y, sum_z) = vertex_sum_simd(x, y, z);
    let n = x.len() as f32;
    (sum_x / n, sum_y / n, sum_z / n)
}

/// SIMD-accelerated surface area for triangular mesh
#[inline]
pub unsafe fn surface_area_simd(
    x: &[f32],
    y: &[f32],
    z: &[f32],
    faces: &[u32],
) -> f32 {
    let n_faces = faces.len() / 3;
    let mut area_sum = 0.0f32;
    let mut i = 0;

    let ptr_x = x.as_ptr();
    let ptr_y = y.as_ptr();
    let ptr_z = z.as_ptr();
    let ptr_f = faces.as_ptr();

    while i < n_faces {
        let i0 = *ptr_f.add(i * 3) as usize;
        let i1 = *ptr_f.add(i * 3 + 1) as usize;
        let i2 = *ptr_f.add(i * 3 + 2) as usize;

        let ax = *ptr_x.add(i0);
        let ay = *ptr_y.add(i0);
        let az = *ptr_z.add(i0);
        let bx = *ptr_x.add(i1);
        let by = *ptr_y.add(i1);
        let bz = *ptr_z.add(i1);
        let cx = *ptr_x.add(i2);
        let cy = *ptr_y.add(i2);
        let cz = *ptr_z.add(i2);

        let bax = bx - ax;
        let bay = by - ay;
        let baz = bz - az;
        let cax = cx - ax;
        let cay = cy - ay;
        let caz = cz - az;

        let cx1 = bay * caz - baz * cay;
        let cy1 = baz * cax - bax * caz;
        let cz1 = bax * cay - bay * cax;

        area_sum += 0.5 * (cx1 * cx1 + cy1 * cy1 + cz1 * cz1).sqrt();

        i += 1;
    }

    area_sum
}

/// Benchmark helper for SIMD operations
pub fn benchmark_vertex_sum() {
    let n = 263_169;
    let x: Vec<f32> = vec![0.0; n];
    let y: Vec<f32> = vec![1.0; n];
    let z: Vec<f32> = vec![2.0; n];

    let runs = 10;

    println!("SIMD Vertex Sum Benchmark ({} vertices)", n);

    // Scalar baseline
    let mut scalar_times = Vec::new();
    for _ in 0..runs {
        let start = std::time::Instant::now();
        let mut sum_x = 0.0f32;
        let mut sum_y = 0.0f32;
        let mut sum_z = 0.0f32;
        for i in 0..n {
            sum_x += x[i];
            sum_y += y[i];
            sum_z += z[i];
            black_box((sum_x, sum_y, sum_z));
        }
        scalar_times.push(start.elapsed().as_nanos());
    }

    // SIMD
    let mut simd_times = Vec::new();
    for _ in 0..runs {
        let start = std::time::Instant::now();
        let (s_x, s_y, s_z) = unsafe { vertex_sum_simd(&x, &y, &z) };
        black_box((s_x, s_y, s_z));
        simd_times.push(start.elapsed().as_nanos());
    }

    let scalar_avg = scalar_times.iter().sum::<u128>() as f64 / runs as f64;
    let simd_avg = simd_times.iter().sum::<u128>() as f64 / runs as f64;

    println!("  Scalar: {:.0} ns ({:.2} ns/vertex)", scalar_avg, scalar_avg / n as f64);
    println!("  SIMD:   {:.0} ns ({:.2} ns/vertex)", simd_avg, simd_avg / n as f64);
    println!("  Speedup: {:.1}x", scalar_avg / simd_avg);
}
