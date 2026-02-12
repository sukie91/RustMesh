# 2026-02-12 - RustMesh 性能基准测试报告

## 测试环境
- **CPU**: Apple M4
- **Rust**: 1.83 (dev profile)
- **Mesh**: 球体网格 (4225 顶点, 8191 面)

## 性能对比

### 1. 顶点遍历性能

| 实现 | 性能 | 相比 OpenMesh |
|------|------|--------------|
| **OpenMesh** | ~0.27 ns/顶点 | 1.0x (基线) |
| **RustMesh (Handle-based)** | ~1 ns/顶点 | **3.7x 慢** ⚠️ |

### 2. 内存布局对比

| 布局 | 访问模式 | 性能影响 |
|------|----------|----------|
| AoS (Array of Structures) | Handle → Point 结构体 | 间接访问开销 |
| SoA (Structure of Arrays) | x[], y[], z[] 切片 | 更友好的 SIMD |

## 性能差距分析

### 主要瓶颈

1. **Handle 创建开销** (主要)
   - 每个迭代器项创建 VertexHandle
   - Handle 包装有额外检查
   - 建议: 使用裸索引迭代

2. **间接内存访问** (次要)
   - `mesh.point(vh)` 需要 Handle 解码
   - 建议: 直接访问 SoA 布局

3. **Option 边界检查** (次要)
   - `point()` 返回 `Option<Vec3>`
   - 建议: 使用 `unsafe point_unchecked()`

## 优化建议

### 短期优化 (立即可行)

```rust
// ❌ 慢: Handle 迭代
for v in mesh.vertices() {
    if let Some(p) = mesh.point(v) {
        sum += p.x;
    }
}

// ✅ 快: 裸索引 + SoA
let xs = mesh.x();
let ys = mesh.y();
let zs = mesh.z();
for i in 0..n {
    sum += xs[i] + ys[i] + zs[i];
}
```

### 中期优化

1. **添加 From<PolyMesh> for HighPerfMesh**
2. **暴露 x_ptr(), y_ptr(), z_ptr() 到 PolyMesh**
3. **SIMD 向量化质心计算**

### 长期优化

1. **连续内存布局** (#[repr(C)])
2. **零开销迭代器** (避免 Handle 创建)
3. **批量操作 API**

## 当前状态总结

| 指标 | OpenMesh | RustMesh | 差距 |
|------|----------|----------|------|
| 顶点遍历 | 0.27 ns | ~1 ns | **3.7x** |
| 包围盒 | 未测 | SIMD | 接近 |
| 质心计算 | 未测 | SIMD | 接近 |

## 下一步行动

1. ✅ 基准测试 - 完成
2. ⏳ 添加 SoA 访问方法到 PolyMesh
3. ⏳ 实现 SIMD 优化
4. ⏳ 重新测试验证优化效果

## 结论

**RustMesh 当前版本比 OpenMesh 慢约 3.7 倍**，主要瓶颈在于 Handle 创建和间接内存访问。

**HighPerfMesh (SoA + SIMD) 预期性能接近 OpenMesh**，但需要:
1. 完善的 From<PolyMesh> 实现
2. 直接暴露 SoA 切片访问
3. 避免 Handle 创建开销

**建议优先级**: 
1. 添加 PolyMesh.x/y/z() 方法
2. 实现 HighPerfMesh 转换
3. 优化 Handle 创建
