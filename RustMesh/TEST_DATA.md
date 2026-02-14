# RustMesh 测试数据获取指南

## 一、内置测试数据生成器

RustMesh 提供了内置的测试数据生成器，无需下载外部数据！

```rust
use rustmesh::{
    generate_cube,
    generate_tetrahedron,
    generate_pyramid,
    generate_icosahedron,
    generate_sphere,
    generate_torus,
    generate_grid,
};
```

### 简单测试模型

| 函数 | 顶点数 | 面数 | 用途 |
|------|--------|------|------|
| `generate_cube()` | 8 | 6 | 基础测试 |
| `generate_tetrahedron()` | 4 | 4 | 最小四面体 |
| `generate_pyramid()` | 5 | 5 | 金字塔测试 |
| `generate_icosahedron()` | 12 | 20 | 二十面体 |

### 中等规模模型

| 函数 | 规模 | 顶点数 | 面数 | 用途 |
|------|------|--------|------|------|
| `generate_sphere(r, 16, 16)` | 16×16 | ~272 | ~544 | 平滑曲面 |
| `generate_sphere(r, 32, 32)` | 32×32 | ~1056 | ~2112 | 性能测试 |
| `generate_torus(2.0, 0.5, 24, 12)` | 24×12 | 325 | 288 | 复杂拓扑 |
| `generate_grid(32, 32)` | 32×32 | 1024 | 961 | 网格测试 |

### 基准测试命令

```bash
# 运行所有基准测试
cd ~/Projects/RustMesh/rustmesh
cargo bench

# 运行特定基准
cargo bench generate_cube
cargo bench generate_sphere
cargo bench traverse_vertices
```

## 二、获取外部测试数据

如果需要真实扫描数据，可以使用以下命令：

```bash
# 创建测试数据目录
mkdir -p ~/Projects/RustMesh/test_data
cd ~/Projects/RustMesh/test_data

# Stanford Bunny (需要手动下载)
# 地址: https://graphics.stanford.edu/data/3Dscanrep/
# 文件: bun000.zip

# McGuire 数据集 (Git clone)
git clone https://github.com/McGuireComputerGraphics/Data-Sets

# glTF 示例模型
git clone https://github.com/KhronosGroup/glTF-Sample-Models
```

## 三、基准测试输出示例

```
generate_cube      time:   [1.23 µs 1.25 µs 1.28 µs]
generate_sphere_16x16  time:   [45.2 µs 46.1 µs 47.0 µs]
traverse_vertices_32x32 time:   [12.1 µs 12.3 µs 12.5 µs]
traverse_faces_32x32   time:   [8.4 µs 8.6 µs 8.8 µs]
subdivide_cube_midpoint_1 time: [156 µs 159 µs 162 µs]
```

## 四、测试数据统计

| 模型 | 顶点数 | 面数 | 边数 |
|------|--------|------|------|
| Cube | 8 | 6 | 12 |
| Tetrahedron | 4 | 4 | 6 |
| Pyramid | 5 | 5 | 8 |
| Icosahedron | 12 | 20 | 30 |
| Sphere 16×16 | 272 | 544 | 816 |
| Sphere 32×32 | 1056 | 2112 | 3168 |
| Torus 24×12 | 325 | 288 | 613 |
| Grid 32×32 | 1024 | 961 | 1985 |

## 五、使用示例

```rust
use rustmesh::{PolyMesh, generate_sphere, Smoother};

// 生成测试数据
let mesh = generate_sphere(1.0, 32, 32);
println!("顶点: {}, 面: {}", mesh.n_vertices(), mesh.n_faces());

// 运行平滑测试
let mut noisy_mesh = generate_noisy_sphere(1.0, 0.1, 16, 16);
let mut smoother = Smoother::new();
smoother.smooth(&mut noisy_mesh);

// 保存到文件
use rustmesh::write_mesh;
write_mesh(&mesh, "test_sphere.obj").unwrap();
```

## 六、测试数据来源

| 数据源 | 特点 | 访问方式 |
|--------|------|----------|
| 内置生成器 | 无需下载，即开即用 | Rust 代码 |
| Stanford 扫描 | 经典测试数据 | 手动下载 |
| Thingi10K | 10K 3D 打印模型 | 网站下载 |
| ModelNet | CAD 分类数据 | Princeton 网站 |
| glTF-Samples | 现代格式示例 | Git clone |

---

**建议**: 先使用内置生成器进行开发和测试，需要真实数据时再下载外部数据集。
