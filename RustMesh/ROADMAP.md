# RustMesh 开发计划

## 阶段 1: 核心功能完善 (当前阶段)

### 1.1 修复半边结构
- [ ] 在 `Edge` 中存储实际的 `Halfedge` 对象，而不是句柄
- [ ] 实现完整的 prev/next 半边链接
- [ ] 添加 `halfedge()` 方法返回可变引用

### 1.2 完善连接关系
- [ ] 实现 `next_halfedge_handle()` / `set_next_halfedge_handle()`
- [ ] 实现 `prev_halfedge_handle()` 
- [ ] 实现 `VertexEdgeCirculator` (顶点相邻边循环器)
- [ ] 实现 `FaceVertexCirculator` (面顶点循环器)
- [ ] 实现 `FaceEdgeCirculator` (面边循环器)

## 阶段 2: 三角形网格特化

### 2.1 TriConnectivity
- [x] 创建 `TriMesh` 类型
- [ ] 实现三角形特有的操作 (三角形面积、法线计算)
- [ ] 添加三角形网格验证

## 阶段 3: 属性系统

### 3.1 通用属性
- [x] 属性系统合并进 `ArrayKernel`
- [x] 添加法线属性 (`Normal`)
- [x] 添加颜色属性 (`Color`)
- [x] 添加纹理坐标属性 (`TexCoord`)
- [ ] 实现属性迭代器

### 3.2 Trait 系统
- [ ] 定义 `MeshGeometry` trait
- [ ] 定义 `MeshConnectivity` trait
- [ ] 支持多态网格操作

## 阶段 4: 文件 IO

### 4.1 基础 IO
- [ ] 实现 OBJ 文件读写
- [ ] 实现 OFF 文件读写
- [ ] 实现 STL 文件读写
- [ ] 实现 PLY 文件读写

### 4.2 高级特性
- [ ] 支持压缩格式
- [ ] 二进制格式支持
- [ ] 增量加载

## 阶段 5: 性能优化

### 5.1 内存优化
- [ ] 使用 `Vec` 的 capacity 管理
- [ ] 实现内存池 (可选)
- [ ] 延迟分配

### 5.2 算法优化
- [ ] 并行遍历 (Rayon 集成)
- [ ] 缓存友好的数据结构
- [ ] SIMD 优化 (通过 glam)

## 阶段 6: 测试与验证

### 6.1 单元测试
- [ ] 网格操作测试
- [ ] 连接关系测试
- [ ] 属性测试

### 6.2 基准测试
- [ ] 顶点/边/面创建
- [ ] 遍历性能
- [ ] 与 OpenMesh C++ 对比

### 6.3 集成测试
- [ ] 真实模型加载
- [ ] 算法验证

## 任务优先级

高优先级:
1. 完善半边数据结构 (阻塞其他连接操作)
2. 实现完整的 circulator 循环器
3. OBJ/IO 读写 (实用功能)

中优先级:
4. 属性系统
5. TriConnectivity

低优先级:
6. 高级优化
7. 压缩格式

## 贡献指南

### 代码风格
- 遵循 Rust 标准风格 (`rustfmt`)
- 添加单元测试
- 文档注释 (doc strings)

### 提交信息
- 使用 conventional commits
- 关联 GitHub issues
