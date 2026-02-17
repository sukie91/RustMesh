---
stepsCompleted: [1, 2, 3, 4, 5]
inputDocuments:
  - _bmad-output/planning-artifacts/prd.md
  - _bmad-output/planning-artifacts/product-brief-RustScan-2026-02-16.md
  - _bmad-output/project-context.md
  - docs/ARCHITECTURE.md
  - docs/RustSLAM-DESIGN.md
  - docs/ROADMAP.md
workflowType: 'architecture'
project_name: 'RustScan'
user_name: ' 飞哥'
date: '2026-02-17'
---

# Architecture Decision Document

_This document builds collaboratively through step-by-step discovery. Sections are appended as we work through each architectural decision together._

## 项目上下文分析

### 需求概览

**功能需求分类**：

RustScan 的功能需求涵盖五大核心模块：

1. **视频输入处理**（FR1-FR2）
   - 支持 iPhone 视频格式（MP4/MOV/HEVC）
   - 视频格式验证和错误报告

2. **SLAM 处理**（FR3-FR7）
   - 特征提取（ORB/Harris/FAST）
   - 帧间特征匹配
   - 相机位姿估计
   - Bundle Adjustment 优化
   - 回环检测与闭合

3. **3DGS 训练**（FR8-FR10）
   - 深度约束的 3DGS 训练
   - GPU 加速（Metal/MPS）
   - 训练场景文件输出

4. **网格生成**（FR11-FR13）
   - TSDF 体积融合
   - Marching Cubes 网格提取
   - 可导出网格文件（OBJ/PLY）

5. **CLI 接口与诊断**（FR14-FR21）
   - 命令行完整管道执行
   - 非交互式模式
   - 结构化数据输出（JSON）
   - 配置文件支持（YAML/TOML）
   - 可配置日志级别
   - 清晰的错误消息和恢复建议

**非功能需求（架构驱动因素）**：

1. **性能要求**
   - 处理时间：≤ 30 分钟（2-3 分钟视频）
   - 3DGS 渲染质量：PSNR > 28 dB
   - SLAM 跟踪成功率：> 95%
   - 网格质量：< 1% 孤立三角形

2. **GPU 加速**
   - Metal/MPS 后端（Apple Silicon）
   - 最大化 GPU 利用率
   - 优化内存管理

3. **集成与兼容性**
   - 输出格式：OBJ、PLY 网格文件
   - 兼容性：Blender 和 Unity 可导入

4. **可脚本化**
   - 非交互式执行
   - 无提示自动化
   - 结构化输出（JSON）

**规模与复杂度**：

- **主要领域**：科学计算/计算机视觉
- **复杂度级别**：高
- **估计架构组件**：
  - RustSLAM：~48 个源文件（核心、特征、跟踪、优化、回环、融合、管道、IO）
  - RustMesh：~50 个源文件（核心、工具、实用程序）
  - 总计：~98 个文件，~27K 行代码，245+ 测试用例

- **项目类型**：CLI 工具（当前）→ 桌面应用（未来）
- **项目上下文**：棕地项目（~85% 完成，Phase 1 核心管道已打通）

### 技术约束与依赖

**语言与平台约束**：
- Rust Edition 2021（纯 Rust 实现）
- Apple Silicon 平台（Metal/MPS GPU 加速）
- macOS 目标平台

**核心依赖**：
- `glam 0.25` - SIMD 加速数学库（Vec3、Mat4、Quat）- **唯一的数学库**
- `candle-core 0.9.2` + `candle-metal 0.27.1` - GPU 加速（Apple MPS）
- `apex-solver 1.0` - Bundle Adjustment 优化器（可选，用于高级 BA）
- `rayon 1.8` - 数据并行处理
- `kiddo 5.2.1` - KD-Tree（KNN 匹配）
- `crossbeam-channel 0.5` - 线程间通信

**可选特性**：
- `opencv 0.98`（特性："opencv"）
- `tch 0.5`（特性："deep-learning"）
- `image 0.25`（特性："image"）

**现有架构约束**：
- 双库架构：RustMesh（网格处理）+ RustSLAM（SLAM + 3DGS）
- SoA（Structure of Arrays）内存布局（RustMesh）
- Half-edge 数据结构（RustMesh）
- 多线程管道：跟踪、建图、回环、融合线程（RustSLAM）

### 已识别的横切关注点

1. **GPU 加速与内存管理**
   - 影响：3DGS 训练、渲染、反向传播、密集化/修剪
   - 约束：最小化 CPU↔GPU 数据传输，批处理操作，GPU 内存限制

2. **并行处理**
   - 影响：多线程 SLAM 管道、并行特征提取、并行网格操作
   - 技术：rayon（数据并行）、crossbeam-channel（线程通信）

3. **性能优化**
   - 影响：所有热路径代码
   - 技术：SIMD（glam）、内联标注、LTO、零成本抽象、缓存友好的 SoA 布局

4. **错误处理**
   - 影响：所有公共 API
   - 约束：库代码返回 Result/Option，不 panic（除非不可恢复的程序员错误）
   - 技术：thiserror 自定义错误类型

5. **测试策略**
   - 影响：所有模块
   - 覆盖：245+ 测试用例（单元测试 + 集成测试）
   - 工具：criterion（基准测试）、tempfile（临时文件测试）

6. **类型安全与不变量**
   - 影响：Handle 系统、Half-edge 连接性
   - 约束：使用 u32 作为 handle 索引、u32::MAX 作为无效标记、newtype 模式强制不变量

7. **文档与 AI 代理一致性**
   - 影响：所有公共 API
   - 约束：85 条 AI 代理实施规则（project-context.md）
   - 要求：模块级文档（//!）、公共 API 文档（///）、坐标系统文档

## 现有架构确认（棕地项目）

### 已实施的架构决策

#### 1. 项目结构

**决策**：Cargo 工作空间 + 双库架构
- **RustMesh**：网格处理库（~50 个文件，~12K 行代码）
- **RustSLAM**：Visual SLAM + 3DGS 库（~48 个文件，~15K 行代码）

**理由**：
- 关注点分离：网格处理与 SLAM 逻辑解耦
- 可独立测试和基准测试
- 可作为独立库发布和重用

#### 2. 数据结构

**RustMesh - Half-edge + SoA 布局**：
- Half-edge 数据结构：高效的网格拓扑遍历
- SoA 内存布局：缓存友好、SIMD 优化、更好的内存局部性
- 与 OpenMesh 兼容的 API 设计

**RustSLAM - 帧管理**：
- Frame/KeyFrame/MapPoint 分离
- 清晰的数据生命周期管理
- 支持滑动窗口优化

#### 3. 并行处理

**多线程管道架构**：
- 跟踪线程（Tracking）
- 局部建图线程（Local Mapping）
- 回环检测线程（Loop Closing）
- 高斯融合线程（Gaussian Fusion）

**技术选择**：
- rayon - 数据并行
- crossbeam-channel - 线程间通信（比 std::sync::mpsc 更快）

#### 4. GPU 加速

**Apple Metal/MPS 通过 candle-metal**：
- 适用于：3DGS 训练、渲染、反向传播
- 约束：最小化 CPU↔GPU 数据传输，批处理操作

#### 5. 错误处理

**Result/Option + thiserror**：
- 库代码返回 Result/Option
- 不 panic（除非不可恢复的程序员错误）
- 使用 thiserror 定义自定义错误类型

#### 6. 3DGS → 网格提取

**TSDF Volume + Marching Cubes**：
- TSDF Volume：纯 Rust 实现，体积融合
- Marching Cubes：256 种情况查找表
- 后处理：聚类过滤、法线平滑

## 关键架构决策记录（ADR）

### ADR-001: CLI 接口框架

**状态**：已决策
**日期**：2026-02-17

**决策**：使用 clap（派生宏方式）

**上下文**：需要专业的 CLI 接口，支持配置文件、参数验证和帮助文档生成。

**选项考虑**：
1. **clap (derive)** - 类型安全、声明式、社区支持
2. **手动解析** - 零依赖但维护负担重

**权衡矩阵**：

| 标准 | clap (derive) | 手动解析 |
|------|---------------|----------|
| 类型安全 | ✅ 编译时 | ⚠️ 运行时 |
| 维护性 | ✅ 声明式 | ❌ 命令式 |
| 帮助文档 | ✅ 自动生成 | ❌ 手动维护 |
| 编译时间 | ⚠️ +宏展开 | ✅ 最快 |
| 配置文件集成 | ✅ clap-serde | ❌ 需自建 |

**决策理由**：
1. 类型安全与项目理念一致（85 条规则："使用类型系统强制不变量"）
2. 声明式 API 易于理解和修改
3. 自动生成专业帮助文档
4. clap-serde 支持 YAML/TOML 配置

**实施指导**：
```rust
use clap::Parser;

#[derive(Parser)]
#[command(name = "rustscan")]
#[command(about = "3D reconstruction from iPhone videos")]
struct Cli {
    /// Input video file (MP4/MOV/HEVC)
    #[arg(short, long)]
    input: PathBuf,

    /// Output directory
    #[arg(short, long, default_value = "./output")]
    output: PathBuf,

    /// Configuration file (YAML/TOML)
    #[arg(short, long)]
    config: Option<PathBuf>,

    /// Log level
    #[arg(long, default_value = "info")]
    log_level: String,
}
```

**后果**：
- ✅ 编译时类型安全
- ✅ 长期维护性提升
- ⚠️ 编译时间略增（可接受）

---

### ADR-002: 视频解码策略

**状态**：已决策
**日期**：2026-02-17

**决策**：ffmpeg-next + 按需解码 + LRU 缓存

**上下文**：需要高效解码 iPhone 视频（H.264/HEVC），满足 30 分钟处理时间要求。

**选项考虑**：
1. **ffmpeg-next** - 硬件加速、格式全面、元数据丰富
2. **opencv** - API 简单但功能有限
3. **gstreamer** - 功能强大但过于复杂

**权衡矩阵**：

| 标准 | ffmpeg-next | opencv | gstreamer |
|------|-------------|--------|-----------|
| 格式支持 | ✅ 全面 | ⚠️ 基本 | ✅ 全面 |
| 硬件加速 | ✅ VideoToolbox | ⚠️ 有限 | ✅ 支持 |
| API 复杂度 | ⚠️ 中等 | ✅ 简单 | ❌ 复杂 |
| 元数据提取 | ✅ 丰富 | ⚠️ 基本 | ✅ 丰富 |
| 性能 | ✅ 优秀 | ⚠️ 一般 | ✅ 优秀 |

**决策理由**：
1. VideoToolbox 硬件加速关键（5-10x 性能提升）
2. 支持所有 iPhone 视频格式（H.264/HEVC）
3. 丰富的元数据提取能力（相机内参、时间戳）
4. 按需解码减少内存占用

**实施指导**：
```rust
struct VideoDecoder {
    decoder: ffmpeg::decoder::Video,
    frame_cache: LruCache<usize, Frame>,
}

impl VideoDecoder {
    fn decode_frame(&mut self, index: usize) -> Result<&Frame> {
        if let Some(frame) = self.frame_cache.get(&index) {
            return Ok(frame);
        }

        let frame = self.decode_at_index(index)?;
        self.frame_cache.put(index, frame);
        Ok(self.frame_cache.get(&index).unwrap())
    }
}
```

**后果**：
- ✅ 性能目标可达成
- ⚠️ 部署需要系统安装 FFmpeg
- ⚠️ API 相对底层，需要仔细处理生命周期

---

### ADR-003: 管道集成架构

**状态**：已决策
**日期**：2026-02-17

**决策**：顺序执行 + 检查点机制

**上下文**：集成 SLAM → 3DGS → Mesh 的端到端管道，需要可靠性和可调试性。

**选项考虑**：
1. **顺序执行 + 检查点** - 简单、可调试、支持恢复
2. **流式管道** - 内存效率高但复杂度高

**决策理由**：
1. MVP 优先简单性
2. SLAM 全局优化需要完整数据（Bundle Adjustment、Loop Closing）
3. 检查点支持失败恢复
4. 中间结果可检查验证

**检查点设计**：
```rust
struct Pipeline {
    config: PipelineConfig,
    checkpoints: CheckpointManager,
}

impl Pipeline {
    fn run(&mut self, video_path: &Path) -> Result<Output> {
        // Stage 1: SLAM
        let slam_result = if let Some(cp) = self.checkpoints.load("slam")? {
            cp
        } else {
            let result = self.run_slam(video_path)?;
            self.checkpoints.save("slam", &result)?;
            result
        };

        // Stage 2: 3DGS Training
        let gaussian_result = if let Some(cp) = self.checkpoints.load("gaussian")? {
            cp
        } else {
            let result = self.run_gaussian_training(&slam_result)?;
            self.checkpoints.save("gaussian", &result)?;
            result
        };

        // Stage 3: Mesh Extraction
        let mesh = self.extract_mesh(&gaussian_result)?;

        Ok(Output { slam_result, gaussian_result, mesh })
    }
}
```

**后果**：
- ✅ 调试和维护简单
- ✅ 用户体验好（支持恢复）
- ⚠️ 内存占用可能较大（可通过检查点缓解）
- ⚠️ 无法并行处理各阶段

---

### ADR-004: 数学库选择

**状态**：已决策
**日期**：2026-02-17

**决策**：只使用 glam，移除 nalgebra

**上下文**：
- 当前 nalgebra 导入是死代码（在 ba.rs 中导入但未使用）
- 所有功能都用 glam 或手动实现
- 项目需要 SIMD 优化的 3D 数学运算

**代码分析发现**：
```rust
// ba.rs 第 7 行
use nalgebra::{Matrix3, Vector3};  // ❌ 未使用的导入

// 实际使用原始数组
pub struct BALandmark {
    pub position: [f64; 3],  // ✅ 手动实现
}
```

**glam 功能评估**：

✅ **glam 提供**：
- Vec2, Vec3, Vec4 - 向量运算
- Mat2, Mat3, Mat4 - 矩阵运算（含逆矩阵）
- Quat - 四元数旋转
- SIMD 加速（SSE2, AVX2）
- 零成本抽象

❌ **glam 不提供**：
- 任意大小矩阵
- 矩阵分解（SVD, QR, Cholesky）
- 线性方程组求解器
- 特征值/特征向量

**项目需求分析**：

| 功能 | 需要什么 | glam 是否足够 |
|------|----------|---------------|
| SE3 pose | Mat4 + Quat | ✅ 是 |
| 相机变换 | Mat4 乘法 | ✅ 是 |
| 3D 点变换 | Vec3 运算 | ✅ 是 |
| 简化 BA | 梯度下降 | ✅ 是 |
| 高级 BA | 矩阵分解 | ⚠️ 使用 apex-solver |
| 3DGS 训练 | GPU 张量 | ✅ candle 处理 |
| 网格处理 | 几何运算 | ✅ 是 |

**选项考虑**：
1. **glam only** - SIMD 优化、零成本抽象、统一 API
2. **glam + nalgebra** - 支持高级线性代数，但增加复杂度
3. **glam + apex-solver** - 专业 BA 求解器，不需要 nalgebra

**决策理由**：
1. nalgebra 当前未被使用（死代码）
2. glam 满足所有 3D 图形需求
3. 如需高级 BA，使用 apex-solver 而非 nalgebra
4. 减少依赖，加快编译
5. 统一数学库，避免类型转换

**实施步骤**：
1. 从 `RustSLAM/Cargo.toml` 移除 nalgebra 依赖
2. 从 `optimizer/ba.rs` 移除未使用的 nalgebra 导入
3. 如需改进 BA，集成 apex-solver（已在依赖中）

**后果**：
- ✅ 编译时间减少
- ✅ 依赖简化
- ✅ API 统一（避免 glam ↔ nalgebra 转换）
- ✅ SIMD 优化一致性
- ⚠️ 如需矩阵分解，需要 apex-solver 或其他库

**更新的核心依赖列表**：
- `glam 0.25` - **唯一的数学库**（Vec3、Mat4、Quat）
- `candle-core 0.9.2` + `candle-metal 0.27.1` - GPU 加速（Apple MPS）
- `apex-solver 1.0` - Bundle Adjustment 优化器（如需高级 BA）
- `rayon 1.8` - 数据并行处理
- `kiddo 5.2.1` - KD-Tree（KNN 匹配）
- `crossbeam-channel 0.5` - 线程间通信

---

### ADR-005: 输出格式与导出

**状态**：已决策
**日期**：2026-02-17

**决策**：多格式输出 + 结构化元数据

**上下文**：需要支持多种输出格式以满足不同用例（Blender、Unity、分析）。

**输出格式决策**：

1. **3DGS 场景文件**：
   - 格式：自定义二进制格式（`.gs` 扩展名）
   - 元数据：JSON 格式（`scene_meta.json`）
   - 包含：高斯参数、训练配置、质量指标

2. **网格导出**：
   - 同时支持 OBJ 和 PLY 格式
   - OBJ：广泛兼容（Blender、Unity）
   - PLY：支持顶点颜色和法线

3. **元数据保存**：
   - 相机轨迹：JSON 格式（`camera_trajectory.json`）
   - 质量指标：JSON 格式（`quality_metrics.json`）
   - 处理参数：包含在各元数据文件中

**输出目录结构**：
```
output/
├── scene.gs                    # 3DGS 场景（二进制）
├── scene_meta.json             # 场景元数据
├── mesh.obj                    # 网格（OBJ 格式）
├── mesh.ply                    # 网格（PLY 格式）
├── camera_trajectory.json      # 相机轨迹
└── quality_metrics.json        # 质量指标（PSNR、跟踪率等）
```

**决策理由**：
1. 灵活性：支持多种用例
2. 兼容性：OBJ/PLY 是行业标准
3. 可追溯性：完整的元数据记录
4. 可分析性：JSON 格式易于解析

**实施指导**：
```rust
pub struct OutputManager {
    output_dir: PathBuf,
}

impl OutputManager {
    pub fn save_all(&self, results: &PipelineResults) -> Result<()> {
        // 保存 3DGS 场景
        self.save_gaussian_scene(&results.gaussian)?;
        
        // 保存网格（两种格式）
        self.save_mesh_obj(&results.mesh)?;
        self.save_mesh_ply(&results.mesh)?;
        
        // 保存元数据
        self.save_camera_trajectory(&results.slam.trajectory)?;
        self.save_quality_metrics(&results.metrics)?;
        
        Ok(())
    }
}
```

**后果**：
- ✅ 用户可选择最适合的格式
- ✅ 完整的处理记录
- ⚠️ 磁盘空间占用增加（可接受）

---

### ADR-006: 日志与诊断

**状态**：已决策
**日期**：2026-02-17

**决策**：log + env_logger + 结构化输出选项

**上下文**：需要清晰的日志输出和诊断信息，支持调试和生产使用。

**日志库选择**：
- **主库**：`log` crate（Rust 标准日志门面）
- **实现**：`env_logger`（已在项目中使用）
- **理由**：简单、标准、轻量级

**日志级别策略**：
```rust
// 默认级别：info
// 可通过 CLI 参数覆盖
#[arg(long, default_value = "info")]
log_level: String,  // error/warn/info/debug/trace

// 或通过环境变量
// RUST_LOG=debug rustscan --input video.mp4
```

**性能指标收集**：
- 内置基本性能计时（每个阶段的耗时）
- 输出到日志（info 级别）
- 保存到 `quality_metrics.json`

**错误报告格式**：
- **默认**：文本格式（人类可读）
- **可选**：JSON 格式（`--format json`）
- 包含：错误类型、上下文、恢复建议

**实施指导**：
```rust
use log::{info, warn, error, debug};

pub fn run_pipeline(config: &Config) -> Result<()> {
    info!("Starting RustScan pipeline");
    
    let start = Instant::now();
    
    // Stage 1: SLAM
    info!("Stage 1/3: Running SLAM...");
    let slam_result = run_slam(config)?;
    info!("SLAM completed in {:.2}s", start.elapsed().as_secs_f64());
    
    // Stage 2: 3DGS Training
    info!("Stage 2/3: Training 3D Gaussians...");
    let gaussian_result = train_gaussians(&slam_result)?;
    info!("3DGS training completed in {:.2}s", start.elapsed().as_secs_f64());
    
    // Stage 3: Mesh Extraction
    info!("Stage 3/3: Extracting mesh...");
    let mesh = extract_mesh(&gaussian_result)?;
    info!("Mesh extraction completed in {:.2}s", start.elapsed().as_secs_f64());
    
    info!("Pipeline completed successfully in {:.2}s", start.elapsed().as_secs_f64());
    
    Ok(())
}
```

**后果**：
- ✅ 清晰的进度反馈
- ✅ 易于调试
- ✅ 支持自动化（JSON 输出）

---

### ADR-007: 配置管理

**状态**：已决策
**日期**：2026-02-17

**决策**：TOML 配置文件 + serde 验证

**上下文**：需要灵活的配置系统，支持默认值和用户自定义。

**配置文件格式**：TOML（Rust 生态标准）

**配置文件结构**：
```toml
# config.toml

[slam]
feature_type = "ORB"        # ORB, Harris, FAST
max_features = 2000
min_parallax = 0.01
enable_loop_closing = true

[gaussian]
iterations = 30000
learning_rate = 0.01
densify_interval = 100
prune_interval = 100
opacity_threshold = 0.005

[mesh]
voxel_size = 0.01           # TSDF voxel size (meters)
truncation_distance = 0.05  # TSDF truncation distance
min_cluster_size = 100      # Minimum triangle cluster size
smooth_normals = true

[output]
save_obj = true
save_ply = true
save_trajectory = true
save_metrics = true
```

**配置验证**：
```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct Config {
    #[serde(default)]
    pub slam: SlamConfig,
    
    #[serde(default)]
    pub gaussian: GaussianConfig,
    
    #[serde(default)]
    pub mesh: MeshConfig,
    
    #[serde(default)]
    pub output: OutputConfig,
}

impl Config {
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }
    
    pub fn validate(&self) -> Result<()> {
        // 自定义验证逻辑
        if self.gaussian.iterations == 0 {
            return Err("iterations must be > 0".into());
        }
        Ok(())
    }
}
```

**默认值管理**：
- 在代码中定义（使用 `#[serde(default)]`）
- 用户配置文件只需覆盖需要修改的值
- CLI 参数优先级最高

**决策理由**：
1. TOML 是 Rust 生态的标准（Cargo.toml）
2. 易读易写，支持注释
3. serde 提供强大的序列化/反序列化
4. 类型安全的配置验证

**后果**：
- ✅ 灵活的配置系统
- ✅ 类型安全
- ✅ 易于维护

---

### ADR-008: 性能优化策略

**状态**：延后到 Phase 2
**日期**：2026-02-17

**决策**：MVP 阶段使用默认配置，性能优化延后

**上下文**：过早优化是万恶之源，先确保功能正确。

**MVP 阶段默认配置**：

1. **内存分配器**：
   - 使用系统默认分配器
   - 不引入 jemalloc 或其他自定义分配器
   - 理由：简化部署，避免过早优化

2. **SIMD 优化**：
   - 由 glam 自动处理
   - 编译时启用 target-cpu=native（可选）
   - 理由：glam 已经提供了优秀的 SIMD 支持

3. **GPU 批处理大小**：
   - 根据可用 GPU 内存动态调整
   - 使用 candle 的默认配置
   - 理由：避免硬编码，适应不同硬件

4. **并行度配置**：
   - 使用 rayon 的默认配置（CPU 核心数）
   - 不手动设置线程池大小
   - 理由：rayon 的默认策略已经很好

**Phase 2 优化计划**（延后）：
- 性能分析（profiling）
- 识别瓶颈
- 针对性优化
- 基准测试验证

**决策理由**：
1. 功能正确性优先
2. 避免过早优化
3. 保持代码简单
4. 基于实际数据优化

**后果**：
- ✅ 代码简单易维护
- ✅ 避免过早优化陷阱
- ⚠️ 可能需要后续优化（可接受）

---

## 决策优先级分析

### 关键决策（阻塞实施）

以下决策必须在实施前完成，已全部决策：

1. ✅ **CLI 接口框架**（ADR-001）- clap
2. ✅ **视频解码策略**（ADR-002）- ffmpeg-next
3. ✅ **管道集成架构**（ADR-003）- 顺序 + 检查点
4. ✅ **数学库选择**（ADR-004）- glam only

### 重要决策（塑造架构）

以下决策显著影响架构，已全部决策：

5. ✅ **输出格式与导出**（ADR-005）- 多格式 + 元数据
6. ✅ **日志与诊断**（ADR-006）- log + env_logger
7. ✅ **配置管理**（ADR-007）- TOML

### 延后决策（Post-MVP）

以下决策可以延后到 Phase 2：

8. ⏳ **性能优化策略**（ADR-008）- 基于实际数据优化
9. ⏳ **桌面应用架构** - GUI 框架选择（egui vs iced vs tauri）
10. ⏳ **多平台支持** - Windows/Linux 移植

### 实施顺序建议

基于依赖关系，建议的实施顺序：

1. **基础设施**（Week 1）
   - CLI 接口（ADR-001）
   - 配置管理（ADR-007）
   - 日志系统（ADR-006）

2. **核心管道**（Week 2-3）
   - 视频解码（ADR-002）
   - SLAM 集成
   - 3DGS 训练集成
   - 网格提取集成
   - 管道编排（ADR-003）

3. **输出与诊断**（Week 4）
   - 输出格式实现（ADR-005）
   - 质量指标收集
   - 错误处理完善

4. **测试与优化**（Week 5+）
   - 端到端测试
   - 性能分析
   - 文档完善

### 跨组件依赖

**依赖关系图**：
```
CLI (ADR-001)
  ├─> 配置管理 (ADR-007)
  ├─> 日志系统 (ADR-006)
  └─> 管道编排 (ADR-003)
        ├─> 视频解码 (ADR-002)
        ├─> SLAM (使用 glam, ADR-004)
        ├─> 3DGS (使用 glam, ADR-004)
        └─> 网格提取 (使用 glam, ADR-004)
              └─> 输出管理 (ADR-005)
```

**关键路径**：
CLI → 配置 → 管道 → 视频解码 → SLAM → 3DGS → 网格 → 输出

所有关键决策已完成，可以开始实施！


---

## 实施模式与一致性规则

### 决策：使用现有的 project-context.md

**状态**：已决策（复用现有规则）
**日期**：2026-02-17

**决策**：不创建新的实施模式文档，使用现有的 `project-context.md`（85 条 AI 代理实施规则）

**理由**：

项目已经有非常完善的 AI 代理实施规则文档（`_bmad-output/project-context.md`），包含：

1. **语言特定规则（Rust）**：
   - Handle 系统与类型安全
   - 性能标注（#[inline]）
   - 文档标准（//! 和 ///）
   - 错误处理（thiserror）
   - SIMD 与数学库使用
   - 可选特性管理
   - 并行处理模式

2. **架构与库特定规则**：
   - RustMesh: Half-edge 数据结构不变量
   - SoA 内存布局模式
   - RustSLAM: Visual Odometry 管道
   - 3D Gaussian Splatting 模式
   - GPU 加速（Candle + Metal）
   - Bundle Adjustment 模式

3. **测试规则**：
   - 测试组织（#[cfg(test)]、examples/）
   - 测试命名约定
   - 断言与验证模式
   - 性能测试（criterion）
   - 测试数据与 fixtures
   - 覆盖率期望

4. **代码质量与风格规则**：
   - 文件与目录组织
   - 命名约定（类型、函数、常量、模块）
   - 代码格式化（rustfmt）
   - 文档要求
   - 导入组织
   - 代码复杂度控制
   - 安全性与正确性

5. **开发工作流规则**：
   - Git 与仓库管理
   - 构建与开发命令
   - Pull Request 要求
   - 发布流程
   - 依赖管理

6. **关键的"不要错过"规则**：
   - 反模式（❌ 标记）
   - 边界情况处理（⚠️ 标记）
   - 性能陷阱（🐌 标记）
   - 安全考虑（🔒 标记）
   - 坐标系统约定（📐 标记）

**规则总数**：85 条

**规则优化**：
- 为 LLM 优化（optimized_for_llm: true）
- 专注于非显而易见的细节
- 包含具体的代码示例
- 明确标注关键约束

**实施指导**：

所有 AI 代理在实施代码时必须：
1. 完整阅读 `_bmad-output/project-context.md`
2. 严格遵守所有 85 条规则
3. 遇到不确定的情况时，参考规则中的示例
4. 优先遵守标记为 ❌（禁止）的反模式规则

**后果**：
- ✅ 避免重复文档
- ✅ 规则已经过验证和优化
- ✅ 涵盖了 Rust 项目的所有关键模式
- ✅ 保持单一真实来源（Single Source of Truth）

**参考**：
- 完整规则文档：`_bmad-output/project-context.md`
- 规则数量：85 条
- 最后更新：2026-02-16

