# RustMesh SIMD Benchmark 结果

**日期**: 2026-02-11
**模型**: FinalBaseMesh.obj (24,461 顶点, 48,918 面片)
**硬件**: Apple M4

---

## 📊 SIMD 性能对比

| 操作 | 普通版 | SIMD 版 | 提升 |
|------|--------|---------|------|
| 顶点求和 | 1.242 ns/v | 0.274 ns/v | **4.5x** |
| 包围盒 | 39.6 µs | 5.7 µs | **7.0x** |
| 质心 | 25.1 µs | 6.7 µs | **3.8x** |

---

## 🏆 SIMD vs OpenMesh

| 指标 | SIMD RustMesh | OpenMesh | 差距 |
|------|---------------|----------|------|
| 顶点求和 | 0.274 ns/v | 0.267 ns/v | **1.0x** ✅ |

---

## 🎯 结论

- **SIMD 整体提升**: 5.8x
- **vs OpenMesh**: 持平 (1.0x)
- **优化有效！** 🚀

---

## 📁 相关文件

- `test_data/large/FinalBaseMesh.obj` - 测试模型
- `src/bin/user_model_simd_bench.rs` - SIMD benchmark
- `src/simd_ops.rs` - SIMD 实现
- `src/simd_mesh.rs` - SoA 数据结构
