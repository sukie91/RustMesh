# RustScanner

用 Rust 语言实现的 3D Scanner 全套算法库。

## 项目目标

打造一个纯 Rust 实现的 3D 扫描与重建技术栈，涵盖从点云获取到网格处理的完整流程。

## 核心模块

### RustMesh

**核心网格表示与几何处理算法库**

- 网格数据结构 (Half-edge, SoA 布局)
- IO 格式支持 (OBJ, OFF, PLY, STL)
- 网格算法
  - 细分 (Loop, Catmull-Clark, Sqrt3)
  - 简化 (Decimation + Quadric 误差)
  - 光滑 (Laplace, Tangential)
  - 孔洞填充
  - 网格修复
  - 对偶变换
  - 渐进网格 (VDPM)

## 技术栈

- **语言**: Rust
- **数学库**: glam (SIMD 加速)
- **对标**: OpenMesh

## Roadmap

- [ ] 点云获取与预处理
- [ ] 表面重建
- [ ] 网格优化
- [ ] 纹理映射
- [ ] 多视角融合

## 参考

- [OpenMesh](https://www.openmesh.org/) - C++ 网格处理库
- [PensieveRust](https://github.com/sukie91/PensieveRust) - 3D Gaussian Splatting
