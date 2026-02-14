# RustScan

<p align="center">
  <img src="https://img.shields.io/badge/Rust-1.75+-dea584?style=for-the-badge&logo=rust" alt="Rust">
  <img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" alt="License">
</p>

用 Rust 语言实现的 3D Scanner 全套算法库。

## 项目目标

打造一个纯 Rust 实现的 3D 扫描与重建技术栈，涵盖从数据获取到网格处理的完整流程。

```
Pipeline: 相机输入 → RustSLAM → 3DGS 融合 → 网格抽取 → RustMesh 后处理 → 导出
```

---

## 核心模块

### 🟩 RustMesh (网格处理)

**核心网格表示与几何处理算法库**

- 网格数据结构 (Half-edge, SoA 布局)
- IO 格式支持 (OBJ, OFF, PLY, STL, OM)
- 网格算法
  - 细分 (Loop, Catmull-Clark, Sqrt3)
  - 简化 (Decimation + Quadric 误差)
  - 光滑 (Laplace, Tangential)
  - 孔洞填充
  - 网格修复
  - 对偶变换
  - 渐进网格 (VDPM)
- Smart Handle 导航系统
- 属性系统

**进度: ~50-60%** | [详细](./rustmesh/ROADMAP.md)

---

### 🟩 RustSLAM (视觉 SLAM)

**纯 Rust 实现的视觉 SLAM 库**

- 特征提取 (ORB, AKAZE, SuperPoint)
- 视觉里程计 (VO + PnP)
- 局部建图 (三角化 + BA)
- 回环检测 (BoW)
- **3D Gaussian Splatting** - 实时/离线稠密重建
- SLAM + 3DGS 融合

**技术栈**:
- opencv-rust: 图像处理
- glam: SIMD 数学库
- candle: PyTorch 绑定 + Metal GPU
- apex-solver: 图优化
- g2o-rs: 图优化

**进度: ~80%** | [详细](./rustslam/README.md)

---

### ⬜ RustGUI (GUI + 3D 渲染)

**待开发 - 计划使用 egui + wgpu**

- 实时 3D 可视化
- GUI 界面
- 相机控制

**进度: 0%**

---

## 完整流水线

```
┌─────────────────────────────────────────────────────────────────┐
│                    3D Scanning Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [数据获取] → [SLAM] → [3DGS] → [Mesh抽取] → [后处理] → [导出] │
│                      ↓                                          │
│                 实时渲染                                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 模块设计

```
RustScan/
├── RustMesh/           # 核心网格库 (~50%)
│   ├── Core/           # 基础数据结构
│   │   ├── handles.rs      # Handle 系统
│   │   ├── connectivity.rs  # 连接关系
│   │   ├── soa_kernel.rs  # SoA 存储
│   │   ├── smart_handles.rs # Smart Handle
│   │   └── om_format.rs    # OM 格式
│   ├── Tools/          # 网格算法
│   │   ├── decimation.rs   # 简化
│   │   ├── subdivision.rs  # 细分
│   │   ├── smoother.rs    # 平滑
│   │   ├── hole_filling.rs # 孔洞填充
│   │   └── ...
│   └── Utils/          # 工具
│
├── RustSLAM/           # SLAM + 3DGS (~80%)
│   ├── core/           # 核心结构
│   │   ├── frame.rs       # 帧
│   │   ├── keyframe.rs    # 关键帧
│   │   ├── map_point.rs   # 地图点
│   │   └── camera.rs      # 相机模型
│   ├── features/        # 特征提取
│   │   ├── orb.rs         # ORB
│   │   └── pure_rust.rs   # Harris/FAST
│   ├── tracker/         # 视觉里程计
│   ├── optimizer/       # BA 优化
│   ├── loop_closing/    # 回环检测
│   └── fusion/          # 3DGS 融合
│       ├── gaussian.rs    # 高斯数据结构
│       ├── renderer.rs    # 渲染器
│       └── trainer.rs      # 训练
│
└── RustGUI/            # GUI (待开发)
```

---

## 技术栈

- **语言**: Rust 2021
- **数学库**: glam (SIMD 加速)
- **GPU**: wgpu, candle-metal
- **多线程**: rayon
- **对标**: OpenMesh, Open3D, ORB-SLAM3

---

## 快速开始

### RustMesh

```bash
cd RustMesh
cargo build
cargo test
cargo run --example smart_handles_demo
```

### RustSLAM

```bash
cd RustSLAM
cargo build --release
cargo run --example run_vo
cargo test
```

---

## 进度总览

| 模块 | 完成度 | 优先级 | 说明 |
|------|--------|--------|------|
| **RustSLAM** | ~80% | P0 | 核心 SLAM + 3DGS 完备 |
| **RustMesh** | ~50-60% | P1 | 基础扎实，需完善集成 |
| **RustGUI** | 0% | P2 | 待启动 |

### RustSLAM 完成清单

- [x] SE3 Pose
- [x] ORB Feature Extraction
- [x] Feature Matching
- [x] Visual Odometry
- [x] Bundle Adjustment
- [x] Loop Closing
- [x] Relocalization
- [x] 3D Gaussian data structures
- [x] Gaussian Renderer
- [x] Tiled Rasterization
- [x] Depth Sorting
- [x] Alpha Blending
- [x] Gaussian Tracking
- [x] Densification
- [x] Pruning
- [x] Differentiable Renderer
- [x] Training Pipeline
- [x] SLAM Integration

### RustMesh 完成清单

- [x] Handle 系统
- [x] Half-edge 数据结构
- [x] SoA 内存布局
- [x] OFF/OBJ/PLY/STL IO
- [x] MTL 材质支持
- [x] OM 格式 (基础)
- [x] Smart Handle 系统
- [x] EdgeFace 循环器
- [x] Quadric Decimation
- [x] Loop/Catmull-Clark/√3 细分
- [x] Laplace/Tangential 平滑
- [x] Hole Filling
- [x] Mesh Repair
- [x] VDPM 基础

---

## 优先级

| 优先级 | 模块 | 说明 |
|--------|------|------|
| P0 | SLAM | 核心，同时定位与建图 |
| P1 | 网格后处理 | 3DGS → Mesh 抽取 |
| P2 | 表面重建 | Poisson、Ball-Pivoting |
| P3 | RustGUI | 可视化界面 |
| P4 | 纹理映射 | UV 展开 + 贴图 |

---

## 参考

- [OpenMesh](https://www.openmesh.org/) - C++ 网格处理库
- [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) - 视觉 SLAM
- [Open3D](http://www.open3d.org/) - 3D 重建库
- [SplaTAM](https://github.com/spla-tam/SplaTAM) - 3DGS SLAM (CVPR 2024)
- [RTG-SLAM](https://github.com/MisEty/RTG-SLAM) - Real-time 3DGS
- [PensieveRust](https://github.com/sukie91/PensieveRust) - 3D Gaussian Splatting

---

## License

MIT License - see LICENSE file for details.

---

<p align="center">
Built with ❤️ in Rust
</p>
