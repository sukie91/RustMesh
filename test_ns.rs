// Test: 验证 ns 级耗时是否真实

fn main() {
    let n = 263_169;
    let mut x: Vec<f32> = vec![0.0; n];
    
    // 填充一些数据
    for i in 0..n {
        x[i] = (i as f32) * 0.001;
    }
    
    let runs = 10;
    
    // 测试 1: 简单循环
    println!("测试 1: 简单循环 (无 volatile)");
    for r in 0..runs {
        let start = std::time::Instant::now();
        let mut sum = 0.0f32;
        for i in 0..n {
            sum += x[i];
        }
        let elapsed = start.elapsed().as_nanos();
        println!("  Run {}: {} ns (sum={})", r + 1, elapsed, sum);
    }
    
    // 测试 2: 带黑盒
    println!("\n测试 2: 带 black_box");
    for r in 0..runs {
        let start = std::time::Instant::now();
        let mut sum = 0.0f32;
        for i in 0..n {
            sum += x[i];
            std::hint::black_box(sum);
        }
        let elapsed = start.elapsed().as_nanos();
        println!("  Run {}: {} ns", r + 1, elapsed);
    }
    
    // 测试 3: 裸指针
    println!("\n测试 3: 裸指针遍历");
    for r in 0..runs {
        let start = std::time::Instant::now();
        let mut sum = 0.0f32;
        let ptr = x.as_ptr();
        for i in 0..n {
            unsafe {
                sum += *ptr.add(i);
            }
        }
        let elapsed = start.elapsed().as_nanos();
        println!("  Run {}: {} ns", r + 1, elapsed);
    }
}
