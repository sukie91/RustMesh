# RustMesh 项目结构

## 项目概述
RustMesh 是 OpenMesh 11.0.0 的 Rust 语言重写版本，提供通用的几何数据结构用于表示和操作多边形网格。

## 目录结构

```
~/Projects/RustMesh/
├── OpenMesh-11.0.0/          # 原始 OpenMesh C++ 源码
└── rustmesh/                  # Rust 重写项目
    ├── Cargo.toml             # 项目配置
    ├── src/
    │   ├── lib.rs             # 主模块入口
    │   ├── handles.rs         # 句柄类型定义
    │   ├── items.rs           # 网格元素定义
    │   ├── kernel.rs          # 核心存储 (ArrayKernel)
    │   ├── connectivity.rs    # 连接关系 (PolyConnectivity)
    │   └── geometry.rs        # 几何运算
    └── benches/
        └── benchmarks.rs      # 基准测试
```

## 核心组件

### 1. Handles (src/handles.rs)
- `VertexHandle` - 顶点句柄
- `HalfedgeHandle` - 半边句柄  
- `EdgeHandle` - 边句柄
- `FaceHandle` - 面句柄

### 2. Items (src/items.rs)
- `Vertex` - 顶点数据（位置 + 半边引用）
- `Halfedge` - 半边数据（顶点、面、邻接关系）
- `Edge` - 边数据（两个半边）
- `Face` - 面数据（半边引用）

### 3. Kernel (src/kernel.rs)
- `ArrayKernel` - 使用 Vec 存储网格元素
- `PropertyContainer` - 属性容器
- `StatusInfo` - 状态标志

### 4. Connectivity (src/connectivity.rs)
- `PolyMesh` - 多边形网格
- 迭代器 (VertexIter, EdgeIter, FaceIter, HalfedgeIter)
- 循环器 (VertexVertexCirculator, VertexFaceCirculator)

### 5. Geometry (src/geometry.rs)
- `triangle_centroid` - 三角形重心
- `triangle_area` - 三角形面积
- `triangle_normal` - 三角形法线
- `bounding_box` - 边界框

## 当前状态

### ✅ 已完成
- 项目基础结构
- Handles 类型定义
- Mesh Items 数据结构
- ArrayKernel 核心存储
- PolyConnectivity 连接关系
- 几何运算工具
- 单元测试 (12/12 通过)

### ⏳ 待实现
- TriConnectivity (三角形网格特化)
- AttribKernel (属性管理)
- IO 模块 (OBJ, OFF, STL, PLY 读写)
- 完整的半边结构 (prev/next linkage)
- 垃圾回收机制
- 性能优化

## 依赖
- `glam` - 线性代数库 (Vec3 类型)
- `nalgebra` - 备选数学库
- `serde` - 序列化支持
- `itertools` - 迭代器工具

## 使用示例

```rust
use rustmesh::{PolyMesh, Vec3};

let mut mesh = PolyMesh::new();

// 添加顶点
let v0 = mesh.add_vertex(Vec3::new(0.0, 0.0, 0.0));
let v1 = mesh.add_vertex(Vec3::new(1.0, 0.0, 0.0));
let v2 = mesh.add_vertex(Vec3::new(0.0, 1.0, 0.0));

// 添加面
mesh.add_face(&[v0, v1, v2]);

// 遍历
for v in mesh.vertices() {
    println!("{:?}", mesh.point(v));
}
```
