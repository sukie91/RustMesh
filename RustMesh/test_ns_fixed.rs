// Test: 验证 ns 级耗时是否真实 (修复版)

fn main() {
    let n = 263_169;
    let mut x: Vec<f32> = vec![0.0; n];
    
    // 填充一些数据
    for i in 0..n {
        x[i] = (i as f32) * 0.001;
    }
    
    // 使用原子操作或 volatile
    use std::sync::atomic::{AtomicU32, Ordering};
    let sum = AtomicU32::new(0);
    
    let runs = 5;
    
    println!("测试: 使用原子操作防止优化");
    for r in 0..runs {
        sum.store(0, Ordering::SeqCst);
        let start = std::time::Instant::now();
        let ptr = x.as_ptr();
        for i in 0..n {
            let val = unsafe { *ptr.add(i) } as u32;
            sum.fetch_add(val, Ordering::SeqCst);
        }
        let elapsed = start.elapsed().as_nanos();
        let result = sum.load(Ordering::SeqCst);
        println!("  Run {}: {} ns (result={})", r + 1, elapsed, result);
    }
    
    // 使用 std::hint::black_box 正确
    println!("\n测试: 使用 black_box(sum) 每次迭代");
    for r in 0..runs {
        let start = std::time::Instant::now();
        let mut sum = 0.0f32;
        let ptr = x.as_ptr();
        for i in 0..n {
            unsafe {
                sum += *ptr.add(i);
            }
            // 每次迭代都 black_box
            std::hint::black_box(&sum);
        }
        let elapsed = start.elapsed().as_nanos();
        println!("  Run {}: {} ns (sum={})", r + 1, elapsed, sum);
    }
    
    // 只在循环后 black_box (可能仍被优化)
    println!("\n测试: 只在循环后 black_box (危险)");
    for r in 0..runs {
        let start = std::time::Instant::now();
        let mut sum = 0.0f32;
        let ptr = x.as_ptr();
        for i in 0..n {
            unsafe {
                sum += *ptr.add(i);
            }
        }
        std::hint::black_box(sum);
        let elapsed = start.elapsed().as_nanos();
        println!("  Run {}: {} ns", r + 1, elapsed);
    }
}
