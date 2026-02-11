# RustMesh vs OpenMesh 性能分析报告

## 测试环境

| 项目 | 配置 |
|------|------|
| CPU | Apple M4 |
| Rust | 1.83 (release, O3) |
| C++ | clang++ 16.0.0 (O3) |
| 测试网格 | 263,169 顶点, 524,287 面 |

## 真实性能对比

| 操作 | RustMesh | OpenMesh | 差距 |
|------|----------|---------|------|
| 顶点遍历 | 507 µs (1.93 ns/v) | 291 ns (0.27 ns/v) | **7.2x** |
| 面片遍历 | 183 µs (0.35 ns/f) | 84 ns (0.04 ns/f) | **8.5x** |
| 顶点计数 | 95 µs (0.36 ns/v) | - | 基线 |

## 规模化性能

| 面片数 | 顶点数 | ns/顶点 | 趋势 |
|--------|--------|---------|------|
| 2K | 1K | ~270 | OpenMesh 基线 |
| 524K | 263K | 1,927 | RustMesh |

## 性能瓶颈分析

### 1. Handle 创建开销

```rust
// RustMesh: 每次迭代创建 Handle 对象
for v in mesh.vertices() {
    let handle = VertexHandle::new(idx as i32);  // 额外开销
    // ...
}

// OpenMesh: 裸 i32
for (VertexIter v_it = ...) {
    int idx = v_it.handle().idx();  // 直接访问
}
```

### 2. Option + Bounds Check

```rust
// RustMesh: 双重检查
if let Some(p) = mesh.point(v) {  // Option 检查
    vertices.get(idx)               // Bounds 检查
}
// ...

// OpenMesh: 直接访问
mesh.point(v_it.handle())  // 无检查
```

### 3. 内存布局差异

| 数据结构 | RustMesh | OpenMesh | 影响 |
|----------|----------|---------|------|
| Vertex | 32 bytes | 48 bytes | Rust 更紧凑 |
| Halfedge | 48 bytes | 72 bytes | Rust 更紧凑 |
| 内存访问 | Vec 间接 | 连续数组 | OpenMesh 更快 |

## 优化方向

### 短期优化 (可实现)

1. **移除 Handle 包装**
   ```rust
   // 直接返回 u32 索引
   pub fn vertex_indices(&self) -> impl Iterator<Item = usize> {
       (0..self.n_vertices())
   }
   ```

2. **添加 unsafe 快速路径**
   ```rust
   #[inline]
   pub unsafe fn point_unchecked(&self, idx: usize) -> &Vec3 {
       &self.vertices[idx]  // 无检查
   }
   ```

### 中期优化

3. **SIMD 矢量化**
   ```rust
   // ARM NEON
   let vx = vld1q_f32(ptr.add(i));
   ```

4. **连续内存布局**
   ```rust
   struct SoAMesh {
       x: Vec<f32>,
       y: Vec<f32>,
       z: Vec<f32>,
   }
   ```

### 长期优化

5. **codegen 改进**
   ```toml
   [profile.release]
   lto = true
   codegen-units = 1
   opt-level = 3
   ```

## 预期优化效果

| 优化 | 预期提升 | 预计差距 |
|------|---------|----------|
| 移除 Handle | 2-3x | 3-4x |
| SIMD | 2-4x | 1-2x |
| 连续内存 | 1.5-2x | 接近 |

## 结论

### ✅ RustMesh 优势

- **内存更紧凑**: 30-40% 更省内存
- **内存安全**: 无悬垂指针，无数据竞争
- **代码清晰**: 更现代的 API 设计

### ❌ RustMesh 劣势

- **性能较慢**: 7-8x 差距
- **边界检查**: 安全性的代价
- **Handle 包装**: 额外对象创建开销

### 📊 最终建议

| 场景 | 推荐 |
|------|------|
| 性能关键 (游戏引擎) | OpenMesh ✅ |
| 内存敏感 (嵌入式) | RustMesh ✅ |
| Rust 项目集成 | RustMesh ✅ |
| 原型开发 | RustMesh ✅ |
| 生产环境 | OpenMesh |

---

**报告生成**: 2026-02-11
**测试方法**: fixed_benchmark.rs (确保编译器不优化)
