# RustScan Epic Issues - 未完成任务列表

> 生成时间: 2026-02-18
> 项目: RustScan Real-Time SLAM

---

## P1 - HIGH PRIORITY (质量关键)

### Issue #1: P1.1 完成 3DGS 可微渲染（EWA 投影 + 正确混合）

- **Epic**: P1
- **Description**: 实现正确的 3D→2D 协方差投影（EWA）、透视投影 Jacobian、深度排序与透射率/alpha 混合，并修复当前简化实现带来的画质问题。
- **Related Files**: 
  - `RustSLAM/src/fusion/tiled_renderer.rs`
  - `RustSLAM/src/fusion/diff_splat.rs`
  - `RustSLAM/src/fusion/diff_renderer.rs`
- **Estimate**: 5–7 天

---

### Issue #2: P1.2 完成训练管线反向传播与优化器

- **Epic**: P1
- **Description**: 使用 Candle 正确的 backward 流程，提取参数梯度（位置/尺度/旋转/透明度/颜色），实现 Adam 更新、密化与剪枝逻辑，替换占位实现。
- **Related Files**:
  - `RustSLAM/src/fusion/complete_trainer.rs`
  - `RustSLAM/src/fusion/autodiff_trainer.rs`
  - `RustSLAM/src/fusion/trainer.rs`
- **Estimate**: 6–8 天

---

### Issue #3: P1.3 Metal GPU 加速端到端落地

- **Epic**: P1
- **Description**: 保证张量全程位于 Metal 设备，补齐投影/栅格化/混合/梯度的 GPU kernel，减少 CPU-GPU 同步，完成性能 profiling。
- **Related Files**:
  - `RustSLAM/src/fusion/gpu_trainer.rs`
  - `RustSLAM/Cargo.toml`
- **Estimate**: 4–5 天

---

### Issue #4: P1.4 TSDF 体素优化（稀疏哈希 + 并行）

- **Epic**: P1
- **Description**: 引入稀疏体素哈希、SIMD 更新、rayon 并行集成，优化当前简化的光线-体素求交以提升速度。
- **Related Files**:
  - `RustSLAM/src/fusion/tsdf_volume.rs`
- **Estimate**: 3–4 天

---

## P2 - MEDIUM PRIORITY (体验提升)

### Issue #5: P2.1 实时可视化 GUI（RustGUI）

- **Epic**: P2
- **Description**: 新建 GUI crate，包含 3D 视口、控制面板与状态面板；与 SLAM 通道连接显示轨迹/高斯/网格。
- **Related Files**:
  - `RustGUI/`（新 crate）
- **Estimate**: 2–3 周

---

### Issue #6: P2.2 关键帧选择策略（运动驱动）

- **Epic**: P2
- **Description**: 基于平移/旋转阈值与跟踪特征比例的关键帧插入逻辑，替换当前的固定间隔策略并完成集成。
- **Related Files**:
  - `RustSLAM/src/core/keyframe_selector.rs`
- **Estimate**: 2–3 天

---

### Issue #7: P2.3 RGB 视频深度估计（LiDAR/单目/融合）

- **Epic**: P2
- **Description**: 支持 iPhone LiDAR 深度流提取；加入单目深度（MiDaS/DPT/ONNX）作为回退；实现多源深度融合。
- **Related Files**:
  - `RustSLAM/src/depth/stereo.rs`
  - `RustSLAM/src/depth/fusion.rs`
  - `RustSLAM/src/io/lidar_extractor.rs`
- **Estimate**: 1–2 周

---

## 总计

| 类型 | 数量 | 总预估时间 |
|------|------|-----------|
| P1 | 4 | 18-24 天 |
| P2 | 3 | 2-3 周 |
| **总计** | **7** | **~ 5-6 周** |
