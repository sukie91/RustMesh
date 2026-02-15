# RustSLAM 开发任务清单

> 最后更新: 2026-02-15
> 目标: 打通从图像输入到相机位姿准确估计、实时稠密点云生成、3DGS 训练，并完成最终网格提取的完整管道

---

## 一、当前状态分析

RustSLAM 项目已完成核心管道的大部分组件（~85%）。

### 已完成模块

| 模块 | 状态 | 文件位置 |
|------|------|----------|
| SE3 位姿表示 | ✅ | `src/core/pose.rs` |
| ORB/Harris/FAST 特征提取 | ✅ | `src/features/orb.rs`, `pure_rust.rs` |
| 特征匹配 | ✅ | `src/features/matcher.rs`, `knn_matcher.rs` |
| 视觉里程计 (VO) | ✅ | `src/tracker/vo.rs` |
| Bundle Adjustment | ✅ | `src/optimizer/ba.rs` |
| 回环检测 | ✅ | `src/loop_closing/detector.rs` |
| 3DGS 数据结构 | ✅ | `src/fusion/gaussian.rs` |
| 3DGS 渲染（Tiled Rasterization） | ✅ | `src/fusion/tiled_renderer.rs` |
| 3DGS 训练（多种训练器） | ✅ | `src/fusion/complete_trainer.rs`, `autodiff_trainer.rs` |
| 网格提取（TSDF + Marching Cubes） | ✅ | `src/fusion/tsdf_volume.rs`, `marching_cubes.rs` |
| SLAM 集成（稀疏 + 稠密） | ✅ | `src/fusion/slam_integrator.rs` |
| TUM RGB-D 数据集加载 | ✅ | `src/io/dataset.rs` |

### 缺失部分

- ❌ 端到端示例程序（各模块未串联运行）
- ❌ 实时多线程处理管道
- ❌ 单目深度估计
- ❌ KITTI/EuRoC 数据集支持（仅有占位符）
- ❌ IMU 集成
- ❌ 离线全局 3DGS 优化
- ❌ 纹理映射
- ❌ 实时可视化 GUI
- ❌ 配置文件系统
- ❌ 性能基准测试

---

## 二、待开发任务清单

---

### P0 - 核心管道打通（必须完成）

---

#### 任务 1: 端到端示例程序

**目标**: 实现从图像输入到网格导出的完整流程示例，验证管道可行性

**文件位置**: `examples/e2e_slam_to_mesh.rs`

**技术方案**:

```rust
// examples/e2e_slam_to_mesh.rs
use rustslam::io::{DatasetConfig, TumRgbdDataset, Dataset};
use rustslam::tracker::VisualOdometry;
use rustslam::fusion::{GaussianMapper, CompleteTrainer, MeshExtractor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. 加载 TUM RGB-D 数据集
    let config = DatasetConfig {
        root_path: PathBuf::from("data/rgbd_dataset_freiburg1_xyz"),
        load_depth: true,
        load_ground_truth: true,
        ..Default::default()
    };
    let dataset = TumRgbdDataset::load(config)?;

    // 2. 初始化 VO 跟踪器
    let camera = dataset.camera();
    let mut vo = VisualOdometry::new(camera.clone());

    // 3. 初始化 3DGS 映射器
    let mut mapper = GaussianMapper::new();

    // 4. 逐帧处理：位姿估计 + 关键帧选择 + 3DGS 映射
    for frame_result in dataset.frames() {
        let frame = frame_result?;
        let pose = vo.track_frame(&frame)?;

        if frame.index % 5 == 0 {
            mapper.add_keyframe(frame, pose);
        }
    }

    // 5. 3DGS 训练优化
    let mut trainer = CompleteTrainer::new(mapper.gaussians());
    trainer.train(5000)?;

    // 6. 网格提取
    let mesh = MeshExtractor::from_gaussians(trainer.gaussians())?;

    // 7. 导出
    rustmesh::io::write_obj(&mesh, "output.obj")?;
    Ok(())
}
```

**依赖**: 需要整合现有 VO、GaussianMapper、CompleteTrainer、MeshExtractor 模块的接口

---

#### 任务 2: 实时多线程处理管道

**目标**: 实现三线程并行架构，保证跟踪线程实时性

**文件位置**: `src/pipeline/realtime.rs`

**技术方案**:

```
架构设计：
  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │ Tracking     │ ──> │ Mapping      │ ──> │ Optimization │
  │ (30-60 FPS)  │     │ (5-10 FPS)   │     │ (1-2 FPS)    │
  └──────────────┘     └──────────────┘     └──────────────┘
     高优先级              中优先级              低优先级
```

```rust
// src/pipeline/realtime.rs
use crossbeam_channel::{bounded, Sender, Receiver};
use std::thread;

pub struct RealtimePipeline {
    tracking_thread: thread::JoinHandle<()>,
    mapping_thread: thread::JoinHandle<()>,
    optimization_thread: thread::JoinHandle<()>,
}

impl RealtimePipeline {
    pub fn start(dataset: impl Dataset) -> Self {
        // 线程间通信通道
        let (track_tx, map_rx) = bounded::<(Frame, SE3)>(16);
        let (map_tx, opt_rx) = bounded::<KeyFrame>(8);

        // 跟踪线程：最高优先级，逐帧处理
        let tracking_thread = thread::spawn(move || {
            let mut vo = VisualOdometry::new(camera);
            for frame in dataset.frames() {
                let pose = vo.track_frame(&frame).unwrap();
                track_tx.send((frame, pose)).ok();
            }
        });

        // 映射线程：处理关键帧，初始化 Gaussians
        let mapping_thread = thread::spawn(move || {
            let mut mapper = GaussianMapper::new();
            while let Ok((frame, pose)) = map_rx.recv() {
                if mapper.should_add_keyframe(&frame, &pose) {
                    let kf = mapper.add_keyframe(frame, pose);
                    map_tx.send(kf).ok();
                }
            }
        });

        // 优化线程：后台运行 BA + 3DGS 训练
        let optimization_thread = thread::spawn(move || {
            let mut trainer = CompleteTrainer::new();
            while let Ok(kf) = opt_rx.recv() {
                trainer.add_keyframe(kf);
                trainer.train_step(100); // 每次 100 iterations
            }
        });

        Self { tracking_thread, mapping_thread, optimization_thread }
    }
}
```

**关键技术点**:
- 使用 `crossbeam-channel` 的 bounded channel 实现背压控制
- 跟踪线程不会被阻塞，保证实时性
- 映射线程异步处理关键帧
- 优化线程后台运行全局 BA 和 3DGS 训练
- 需要在 `Cargo.toml` 添加 `crossbeam-channel` 依赖

---

#### 任务 3: 深度图生成与融合

**目标**: 为单目情况提供深度估计，或融合多视角深度

**文件位置**: `src/depth/estimator.rs`

**技术方案**:

**方案 A - 单目深度估计（需要深度学习模型）**:

```rust
// src/depth/estimator.rs
use candle_core::{Device, Tensor, Module};

pub struct MonocularDepthEstimator {
    model: Box<dyn Module>,
    device: Device,
}

impl MonocularDepthEstimator {
    /// 加载预训练 MiDaS / Depth-Anything 模型
    pub fn load(model_path: &Path) -> Result<Self> {
        let device = Device::new_metal(0)?;
        let model = load_onnx_model(model_path, &device)?;
        Ok(Self { model, device })
    }

    /// 从 RGB 图像估计深度
    pub fn estimate(&self, rgb: &[u8], width: u32, height: u32) -> Result<Vec<f32>> {
        // 1. 预处理：resize + normalize
        let input = preprocess_image(rgb, width, height, 384, 384)?;
        // 2. 推理
        let output = self.model.forward(&input)?;
        // 3. 后处理：resize 回原始分辨率
        let depth = postprocess_depth(&output, width, height)?;
        Ok(depth)
    }
}
```

**方案 B - 立体匹配（双目/已有深度数据）**:

```rust
// src/depth/stereo.rs
pub struct StereoMatcher {
    block_size: usize,
    num_disparities: usize,
}

impl StereoMatcher {
    /// Semi-Global Matching (SGM) 立体匹配
    pub fn match_stereo(&self, left: &[u8], right: &[u8],
                        width: u32, height: u32) -> Vec<f32> {
        // 1. Census transform
        // 2. Cost volume 计算
        // 3. SGM 路径聚合（8 方向）
        // 4. 视差 -> 深度转换: depth = baseline * fx / disparity
    }
}
```

**建议**: RGB-D 数据集（TUM）已有深度，优先方案 B 用于 KITTI 双目；单目场景再考虑方案 A。

---

### P1 - 性能与鲁棒性（重要）

---

#### 任务 4: GPU 加速 3DGS 训练优化

**目标**: 优化现有 `autodiff_trainer.rs`，减少 CPU-GPU 数据传输

**文件位置**: 优化现有 `src/fusion/autodiff_trainer.rs`

**技术方案**:

```rust
// 优化方向：
// 1. Gaussian 参数常驻 GPU（避免反复拷贝）
pub struct GpuGaussianBuffer {
    positions: Tensor,    // [N, 3] on Metal
    rotations: Tensor,    // [N, 4]
    scales: Tensor,       // [N, 3]
    opacities: Tensor,    // [N, 1]
    sh_coeffs: Tensor,    // [N, C]
}

// 2. 前向渲染完全在 GPU 执行
// 3. Loss 计算在 GPU（L1 + SSIM）
// 4. Backward 在 GPU（链式法则）
// 5. Adam 优化器状态保持在 GPU

// 6. 仅在 densify/prune 时做 CPU-GPU 同步
impl GpuTrainer {
    pub fn train_step(&mut self) {
        // 全部 GPU 操作，无 CPU 回传
        let rendered = self.forward_gpu();
        let loss = self.compute_loss_gpu(&rendered, &self.target);
        let grads = self.backward_gpu(&loss);
        self.adam_step_gpu(&grads);

        self.step_count += 1;
        if self.step_count % 1000 == 0 {
            // 仅此时同步 CPU，执行 densify + prune
            self.densify_and_prune();
        }
    }
}
```

**性能目标**:
- 训练速度 > 10 iterations/sec（1024x768 分辨率）
- 内存占用 < 4GB（100K Gaussians）

---

#### 任务 5: 回环检测优化

**目标**: 提升回环检测速度和准确率

**文件位置**: 优化现有 `src/loop_closing/detector.rs`

**技术方案**:

```rust
// 1. 倒排索引加速词袋检索
pub struct InvertedIndex {
    index: HashMap<WordId, Vec<(FrameId, f32)>>,
}

impl InvertedIndex {
    /// 查询相似帧，O(W) 复杂度（W 为查询帧词数）
    pub fn query(&self, bow: &BowVector, top_k: usize) -> Vec<(FrameId, f32)> {
        let mut scores: HashMap<FrameId, f32> = HashMap::new();
        for (word_id, weight) in bow.iter() {
            if let Some(entries) = self.index.get(word_id) {
                for (frame_id, entry_weight) in entries {
                    *scores.entry(*frame_id).or_default() += weight * entry_weight;
                }
            }
        }
        // 返回 top-k
        let mut ranked: Vec<_> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        ranked.truncate(top_k);
        ranked
    }
}

// 2. 几何验证：RANSAC + PnP
pub struct GeometricVerifier {
    pub fn verify(&self, matches: &[(Vec2, Vec2)]) -> Option<SE3> {
        // RANSAC 5-point algorithm
        // 内点数 > threshold 则确认回环
    }
}

// 3. 使用 SIMD 加速描述子距离计算
// glam 已有 SIMD 支持，可用于批量 Hamming 距离
```

---

#### 任务 6: 关键帧选择策略

**目标**: 智能选择关键帧，减少冗余计算

**文件位置**: `src/core/keyframe_selector.rs`

**技术方案**:

```rust
pub struct KeyframeSelector {
    last_kf_pose: SE3,
    last_kf_feature_count: usize,
    config: KeyframeSelectorConfig,
}

pub struct KeyframeSelectorConfig {
    pub translation_threshold: f32,   // 平移距离阈值 (0.1m)
    pub rotation_threshold: f32,      // 旋转角度阈值 (5 degrees)
    pub covisibility_threshold: f32,  // 共视率阈值 (0.7)
    pub min_tracked_ratio: f32,       // 最小跟踪比例 (0.5)
}

impl KeyframeSelector {
    pub fn should_insert(&self, current_pose: &SE3,
                         tracked_features: usize,
                         total_features: usize) -> bool {
        // 条件 1: 平移距离
        let delta = self.last_kf_pose.inverse() * current_pose;
        let trans_dist = delta.translation().length();
        if trans_dist > self.config.translation_threshold { return true; }

        // 条件 2: 旋转角度
        let angle = delta.rotation().angle();
        if angle > self.config.rotation_threshold.to_radians() { return true; }

        // 条件 3: 特征跟踪比例下降
        let tracked_ratio = tracked_features as f32 / total_features as f32;
        if tracked_ratio < self.config.min_tracked_ratio { return true; }

        false
    }
}
```

---

### P2 - 功能增强（可选）

---

#### 任务 7: 多数据集支持（KITTI / EuRoC）

**目标**: 完成 KITTI 和 EuRoC 数据集加载器（当前仅有占位符）

**文件位置**: `src/io/dataset.rs`（扩展现有文件）

**技术方案**:

**KITTI Odometry**:
```rust
pub struct KittiDataset {
    image_paths: Vec<PathBuf>,     // image_0/*.png
    calibration: KittiCalibration, // P0, P1, P2, P3
    poses: Option<Vec<SE3>>,       // poses.txt
}

impl KittiDataset {
    pub fn load(config: DatasetConfig) -> Result<Self> {
        // 1. 解析 calib.txt -> 4 个 3x4 投影矩阵
        // 2. 加载 image_0/ 下所有 PNG（左目）
        // 3. 可选加载 poses.txt（仅训练集有）
        // 4. 从 P0 提取内参: fx=P0[0,0], fy=P0[1,1], cx=P0[0,2], cy=P0[1,2]
    }
}
```

**EuRoC MAV**:
```rust
pub struct EurocDataset {
    cam0_data: Vec<(f64, PathBuf)>, // 左目
    cam1_data: Vec<(f64, PathBuf)>, // 右目
    imu_data: Vec<ImuMeasurement>, // IMU
    ground_truth: Vec<(f64, SE3)>,
}

impl EurocDataset {
    pub fn load(config: DatasetConfig) -> Result<Self> {
        // 1. 解析 mav0/cam0/data.csv -> 时间戳 + 文件名
        // 2. 解析 mav0/cam0/sensor.yaml -> 内参 + 畸变
        // 3. 可选加载 IMU: mav0/imu0/data.csv
        // 4. 可选加载 GT: mav0/state_groundtruth_estimate0/data.csv
    }
}
```

---

#### 任务 8: IMU 集成

**目标**: 融合 IMU 数据提升位姿估计精度（Visual-Inertial Odometry）

**文件位置**: `src/imu/preintegration.rs`, `src/imu/vio.rs`

**技术方案**:

```rust
// IMU 预积分（参考 ORB-SLAM3）
pub struct ImuPreintegrator {
    delta_p: Vec3,       // 位置增量
    delta_v: Vec3,       // 速度增量
    delta_q: Quat,       // 旋转增量
    covariance: Mat9,    // 协方差矩阵
    jacobian_ba: Mat9x3, // 对加速度偏置的雅可比
    jacobian_bg: Mat9x3, // 对陀螺仪偏置的雅可比
    bias_acc: Vec3,      // 加速度偏置
    bias_gyro: Vec3,     // 陀螺仪偏置
    dt: f64,             // 累计时间
}

impl ImuPreintegrator {
    /// 积分一个 IMU 测量
    pub fn integrate(&mut self, acc: Vec3, gyro: Vec3, dt: f64) {
        // 中值积分法
        let un_acc = self.delta_q * (acc - self.bias_acc);
        let un_gyro = gyro - self.bias_gyro;
        self.delta_p += self.delta_v * dt as f32 + 0.5 * un_acc * (dt * dt) as f32;
        self.delta_v += un_acc * dt as f32;
        self.delta_q *= Quat::from_scaled_axis(un_gyro * dt as f32);
        self.dt += dt;
        // 更新雅可比和协方差 ...
    }
}

// IMU-Visual 紧耦合优化
pub struct VisualInertialOptimizer {
    pub fn optimize(
        &mut self,
        visual_factors: &[VisualFactor],       // 重投影误差
        imu_factors: &[ImuPreintegrator],       // IMU 预积分误差
    ) -> Result<Vec<SE3>> {
        // 使用 apex-solver 联合优化
        // 状态向量: [pose, velocity, bias_acc, bias_gyro] per keyframe
    }
}
```

**参考**: ORB-SLAM3 的 IMU 初始化和优化流程

---

#### 任务 9: 离线全局 3DGS 优化

**目标**: SLAM 完成后对整个 3DGS 场景进行全局优化，提升重建质量

**文件位置**: `src/fusion/global_optimizer.rs`

**技术方案**:

```rust
pub struct GlobalOptimizer {
    gaussians: Vec<Gaussian>,
    keyframes: Vec<KeyFrame>,
    config: GlobalOptConfig,
}

pub struct GlobalOptConfig {
    pub iterations: usize,        // 10K-30K
    pub densify_interval: usize,  // 每 1000 步
    pub lr_position: f32,         // 1e-4
    pub lr_sh: f32,               // 1e-3
    pub ssim_weight: f32,         // 0.2
}

impl GlobalOptimizer {
    pub fn optimize(&mut self) -> Result<()> {
        let mut adam = AdamOptimizer::new(self.config.lr_position);

        for iter in 0..self.config.iterations {
            // 随机选择一个关键帧视角
            let kf = &self.keyframes[iter % self.keyframes.len()];

            // 前向渲染
            let rendered = render_gaussians(&self.gaussians, &kf.pose, &kf.camera);

            // 计算 Loss = (1-w)*L1 + w*SSIM
            let loss = (1.0 - self.config.ssim_weight) * l1_loss(&rendered, &kf.color)
                     + self.config.ssim_weight * (1.0 - ssim(&rendered, &kf.color));

            // 反向传播 + 参数更新
            let grads = backward(&loss);
            adam.step(&mut self.gaussians, &grads);

            // 定期 densify + prune
            if iter % self.config.densify_interval == 0 && iter > 0 {
                densify(&mut self.gaussians, &grads, 0.0002);
                prune(&mut self.gaussians, 0.005);
            }
        }
        Ok(())
    }
}
```

---

#### 任务 10: 纹理映射

**目标**: 为提取的网格生成纹理图集

**文件位置**: `src/fusion/texture_mapper.rs`

**技术方案**:

```rust
pub struct TextureMapper {
    keyframes: Vec<KeyFrame>,
}

impl TextureMapper {
    /// 为网格生成纹理
    pub fn generate_texture(
        &self,
        mesh: &TriangleMesh,
        atlas_size: u32,  // e.g. 4096
    ) -> Result<TexturedMesh> {
        // 1. 为每个三角形选择最佳视角
        //    评分 = dot(face_normal, view_direction) * resolution_factor
        let face_views = self.assign_best_views(mesh);

        // 2. 将三角形打包到纹理图集（rect packing）
        let (uv_coords, packing) = pack_triangles_to_atlas(mesh, atlas_size);

        // 3. 从关键帧采样颜色，写入纹理图
        let texture = render_texture_atlas(mesh, &face_views, &packing,
                                           &self.keyframes, atlas_size);

        // 4. 返回带 UV 和纹理的网格
        Ok(TexturedMesh {
            vertices: mesh.vertices.clone(),
            triangles: mesh.triangles.clone(),
            uv_coords,
            texture,  // RGBA atlas
        })
    }
}

// 导出带纹理的 OBJ（.obj + .mtl + .png）
pub fn write_textured_obj(mesh: &TexturedMesh, path: &Path) -> Result<()> {
    // 写 .obj 文件（含 vt 纹理坐标）
    // 写 .mtl 材质文件
    // 写 .png 纹理图集
}
```

---

### P3 - 用户体验（长期）

---

#### 任务 11: 实时可视化 GUI

**目标**: 创建 RustGUI 项目，提供实时 3D 可视化和控制

**文件位置**: 新建 `RustGUI/` 项目

**技术方案**:

```
技术栈:
- egui: UI 框架（跨平台）
- wgpu: GPU 渲染后端
- winit: 窗口管理
```

```rust
// RustGUI/src/main.rs
pub struct SlamViewer {
    // 3D 渲染
    renderer: WgpuRenderer,
    camera_controller: OrbitCamera,

    // UI 面板
    control_panel: ControlPanel,   // 开始/暂停/保存
    stats_panel: StatsPanel,       // FPS, 内存, Gaussian 数量

    // 实时数据（通过 channel 接收）
    trajectory: Vec<SE3>,          // 相机轨迹
    gaussians: Vec<Gaussian>,      // 3DGS 点云
    mesh: Option<TriangleMesh>,    // 提取的网格
}

// 功能:
// - 实时显示相机轨迹（彩色线条）
// - 渲染 3DGS 点云（splatting 或 point cloud）
// - 切换显示提取的网格
// - 控制面板：开始 / 暂停 / 导出
// - 性能统计：FPS, 帧数, Gaussian 数量, 内存占用
// - 支持鼠标旋转 / 缩放 / 平移
```

---

#### 任务 12: 配置文件系统

**目标**: 使用 YAML 配置所有管道参数，避免硬编码

**文件位置**: `src/config.rs`, `config/default.yaml`

**技术方案**:

```yaml
# config/default.yaml
tracking:
  feature_type: "ORB"
  num_features: 2000
  match_ratio: 0.7
  min_inliers: 30

mapping:
  keyframe_translation: 0.1   # meters
  keyframe_rotation: 5.0      # degrees
  max_keyframes: 200

gaussian:
  max_gaussians: 100000
  densify_threshold: 0.0002
  prune_opacity: 0.005
  sh_degree: 3

optimization:
  ba_iterations: 10
  training_iterations: 5000
  learning_rate: 0.001
  ssim_weight: 0.2

mesh:
  voxel_size: 0.01
  truncation_distance: 0.05
  cluster_min_triangles: 100

output:
  format: "obj"              # obj / ply
  export_texture: true
  atlas_size: 4096
```

```rust
// src/config.rs
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub struct PipelineConfig {
    pub tracking: TrackingConfig,
    pub mapping: MappingConfig,
    pub gaussian: GaussianConfig,
    pub optimization: OptimizationConfig,
    pub mesh: MeshConfig,
    pub output: OutputConfig,
}

impl PipelineConfig {
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        Ok(serde_yaml::from_str(&content)?)
    }

    pub fn default() -> Self { /* 内置默认值 */ }
}
```

**依赖**: 需要添加 `serde_yaml` 到 `Cargo.toml`

---

#### 任务 13: 性能基准测试与评估

**目标**: 在标准数据集上评估位姿精度和重建质量

**文件位置**: `benches/slam_benchmark.rs`, `src/evaluation/`

**技术方案**:

```rust
// src/evaluation/trajectory.rs

/// 绝对轨迹误差 (ATE)
pub fn compute_ate(estimated: &[SE3], ground_truth: &[SE3]) -> f32 {
    // 1. Umeyama 对齐（Sim3）
    let aligned = umeyama_alignment(estimated, ground_truth);
    // 2. 计算 RMSE
    let sum_sq: f32 = aligned.iter().zip(ground_truth.iter())
        .map(|(e, g)| (e.translation() - g.translation()).length_squared())
        .sum();
    (sum_sq / aligned.len() as f32).sqrt()
}

/// 相对位姿误差 (RPE)
pub fn compute_rpe(estimated: &[SE3], ground_truth: &[SE3], delta: usize) -> (f32, f32) {
    // 返回 (translation_error, rotation_error)
}

// benches/slam_benchmark.rs
use criterion::{criterion_group, criterion_main, Criterion};

fn bench_vo_track_frame(c: &mut Criterion) { /* 单帧跟踪耗时 */ }
fn bench_3dgs_train_iteration(c: &mut Criterion) { /* 单次训练迭代 */ }
fn bench_mesh_extraction(c: &mut Criterion) { /* 网格提取耗时 */ }

criterion_group!(benches, bench_vo_track_frame, bench_3dgs_train_iteration, bench_mesh_extraction);
criterion_main!(benches);
```

**评估指标**:
- ATE RMSE (m) — 目标 < 0.05m（TUM fr1/xyz）
- RPE trans (m/s) — 相对平移精度
- RPE rot (deg/s) — 相对旋转精度
- 3DGS PSNR (dB) — 目标 > 25dB
- 3DGS SSIM — 目标 > 0.85
- 网格提取耗时 — 目标 < 5s（100K Gaussians）

---

## 三、优先级与路线图

### 第一阶段：管道打通

- [ ] 任务 1: 端到端示例程序
- [ ] 任务 2: 实时多线程处理管道
- [ ] 任务 3: 深度图生成与融合

### 第二阶段：性能优化

- [ ] 任务 4: GPU 加速 3DGS 训练优化
- [ ] 任务 6: 关键帧选择策略
- [ ] 任务 12: 配置文件系统

### 第三阶段：功能扩展

- [ ] 任务 7: 多数据集支持（KITTI / EuRoC）
- [ ] 任务 9: 离线全局 3DGS 优化
- [ ] 任务 5: 回环检测优化
- [ ] 任务 11: 实时可视化 GUI

### 第四阶段：高级功能

- [ ] 任务 8: IMU 集成
- [ ] 任务 10: 纹理映射
- [ ] 任务 13: 性能基准测试与评估

---

## 四、技术难点预警

| 难点 | 说明 | 应对策略 |
|------|------|----------|
| 实时性能 | 3DGS 训练计算量大 | 充分利用 Metal GPU，减少 CPU-GPU 同步 |
| 内存管理 | 大规模场景 Gaussian 数量膨胀 | 分块管理，主动 prune 低贡献 Gaussian |
| 跟踪鲁棒性 | 快速运动、纹理缺失导致跟踪丢失 | Relocalization 恢复 + IMU 辅助 |
| 深度估计 | 单目深度需要 DL 模型，增加依赖 | 优先支持 RGB-D / 双目，单目作为可选 |
| 纹理接缝 | 多视角纹理映射产生接缝 | Poisson blending + seam optimization |
| 尺度漂移 | 单目 SLAM 无绝对尺度 | 回环约束 + IMU 提供尺度 |
