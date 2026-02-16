//! Parameter structures for SLAM components

use serde::{Deserialize, Serialize};

/// Tracker/VO parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackerParams {
    /// Maximum number of features to detect
    pub max_features: usize,
    /// Minimum number of features to maintain
    pub min_features: usize,
    /// Number of pyramid levels
    pub pyramid_levels: u32,
    /// Patch size for ORB
    pub patch_size: u32,
    /// Scale factor between pyramid levels
    pub scale_factor: f32,
    /// FAST threshold
    pub fast_threshold: u32,
    /// Matching ratio threshold (Lowe's ratio)
    pub match_ratio: f32,
    /// Minimum matches to proceed
    pub min_matches: usize,
    /// Minimum inliers after PnP
    pub min_inliers: usize,
    /// Maximum iterations for PnP
    pub pnp_max_iterations: usize,
}

impl Default for TrackerParams {
    fn default() -> Self {
        Self {
            max_features: 2000,
            min_features: 500,
            pyramid_levels: 8,
            patch_size: 31,
            scale_factor: 1.2,
            fast_threshold: 20,
            match_ratio: 0.75,
            min_matches: 20,
            min_inliers: 10,
            pnp_max_iterations: 20,
        }
    }
}

/// Mapper parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MapperParams {
    /// Maximum number of keyframes
    pub max_keyframes: usize,
    /// Minimum number of keyframes
    pub min_keyframes: usize,
    /// Keyframe interval (frames)
    pub keyframe_interval: usize,
    /// Maximum MapPoints per keyframe
    pub max_points_per_keyframe: usize,
    /// Maximum distance for new MapPoints
    pub max_point_distance: f32,
    /// Minimum distance for new MapPoints
    pub min_point_distance: f32,
    /// Maximum reprojection error
    pub max_reproj_error: f32,
    /// Minimum triangulation angle (degrees)
    pub min_triangulation_angle: f32,
    /// Whether to use local mapping
    pub use_local_mapping: bool,
    /// Local mapping window size
    pub local_mapping_window: usize,
}

impl Default for MapperParams {
    fn default() -> Self {
        Self {
            max_keyframes: 100,
            min_keyframes: 5,
            keyframe_interval: 5,
            max_points_per_keyframe: 500,
            max_point_distance: 50.0,
            min_point_distance: 0.1,
            max_reproj_error: 4.0,
            min_triangulation_angle: 3.0,
            use_local_mapping: true,
            local_mapping_window: 10,
        }
    }
}

/// Optimizer parameters (Bundle Adjustment)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerParams {
    /// Number of iterations for local BA
    pub local_ba_iterations: usize,
    /// Number of iterations for full BA
    pub full_ba_iterations: usize,
    /// Number of iterations for pose optimization
    pub pose_iterations: usize,
    /// Maximum reprojection error (pixels)
    pub max_reproj_error: f32,
    /// Robust kernel threshold
    pub robust_kernel_threshold: f32,
    /// Use parallel BA
    pub use_parallel: bool,
    /// Number of threads for parallel BA
    pub num_threads: usize,
    /// Convergence threshold
    pub convergence_threshold: f32,
}

impl Default for OptimizerParams {
    fn default() -> Self {
        Self {
            local_ba_iterations: 20,
            full_ba_iterations: 100,
            pose_iterations: 10,
            max_reproj_error: 4.0,
            robust_kernel_threshold: 5.0,
            use_parallel: true,
            num_threads: 4,
            convergence_threshold: 1e-6,
        }
    }
}

/// Loop closing parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopClosingParams {
    /// Minimum score for loop detection
    pub min_loop_score: f32,
    /// Minimum matches for geometric verification
    pub min_matches: usize,
    /// Minimum inliers after geometric verification
    pub min_inliers: usize,
    /// Minimum distance (keyframes) between current and loop candidate
    pub min_distance: usize,
    /// RANSAC iterations for geometric verification
    pub ransac_iterations: usize,
    /// Inlier threshold (pixels)
    pub inlier_threshold: f32,
    /// Use SIMD for descriptor matching
    pub use_simd: bool,
    /// Covisibility threshold for keyframe selection
    pub covisibility_threshold: f32,
    /// Enable similarity transform (Sim3) optimization
    pub use_sim3: bool,
    /// Number of iterations for essential matrix optimization
    pub essential_iterations: usize,
}

impl Default for LoopClosingParams {
    fn default() -> Self {
        Self {
            min_loop_score: 0.05,
            min_matches: 20,
            min_inliers: 15,
            min_distance: 30,
            ransac_iterations: 200,
            inlier_threshold: 3.0,
            use_simd: true,
            covisibility_threshold: 0.6,
            use_sim3: true,
            essential_iterations: 100,
        }
    }
}

/// Dataset parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetParams {
    /// Dataset type (tum, kitti, euroc, realsense)
    pub dataset_type: String,
    /// Dataset root path
    pub root_path: String,
    /// Load depth images
    pub load_depth: bool,
    /// Load ground truth
    pub load_ground_truth: bool,
    /// Maximum frames to process
    pub max_frames: usize,
    /// Frame stride (process every N frames)
    pub stride: usize,
    /// Depth scale factor
    pub depth_scale: f32,
    /// Depth truncation threshold
    pub depth_trunc: f32,
}

impl Default for DatasetParams {
    fn default() -> Self {
        Self {
            dataset_type: "tum".to_string(),
            root_path: "".to_string(),
            load_depth: true,
            load_ground_truth: true,
            max_frames: 0,
            stride: 1,
            depth_scale: 1000.0,
            depth_trunc: 10.0,
        }
    }
}

/// Viewer parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewerParams {
    /// Enable viewer
    pub enabled: bool,
    /// Viewer width
    pub width: u32,
    /// Viewer height
    pub height: u32,
    /// Point size
    pub point_size: f32,
    /// Line width
    pub line_width: f32,
    /// Camera view width
    pub camera_view_width: f32,
    /// Camera view height
    pub camera_view_height: f32,
    /// Background color (R, G, B)
    pub background_color: [f32; 3],
    /// Show keyframes
    pub show_keyframes: bool,
    /// Show map points
    pub show_points: bool,
    /// Show current frame
    pub show_current_frame: bool,
    /// Show trajectory
    pub show_trajectory: bool,
    /// Update period (ms)
    pub update_period_ms: u32,
}

impl Default for ViewerParams {
    fn default() -> Self {
        Self {
            enabled: true,
            width: 1024,
            height: 768,
            point_size: 2.0,
            line_width: 1.0,
            camera_view_width: 0.1,
            camera_view_height: 0.1,
            background_color: [0.8, 0.8, 0.8],
            show_keyframes: true,
            show_points: true,
            show_current_frame: true,
            show_trajectory: true,
            update_period_ms: 50,
        }
    }
}

/// 3DGS (Gaussian Splatting) parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaussianSplattingParams {
    /// Maximum number of Gaussians
    pub max_gaussians: usize,
    /// Initial number of Gaussians
    pub init_gaussians: usize,
    /// Densify interval (iterations)
    pub densify_interval: usize,
    /// Densify threshold
    pub densify_threshold: f32,
    /// Prune interval (iterations)
    pub prune_interval: usize,
    /// Prune opacity threshold
    pub prune_opacity: f32,
    /// Learning rate for position
    pub lr_position: f32,
    /// Learning rate for rotation
    pub lr_rotation: f32,
    /// Learning rate for scaling
    pub lr_scale: f32,
    /// Learning rate for opacity
    pub lr_opacity: f32,
    /// Learning rate for color
    pub lr_color: f32,
    /// Batch size for training
    pub batch_size: usize,
    /// Number of training iterations
    pub num_iterations: usize,
    /// Use GPU acceleration
    pub use_gpu: bool,
}

impl Default for GaussianSplattingParams {
    fn default() -> Self {
        Self {
            max_gaussians: 100_000,
            init_gaussians: 1000,
            densify_interval: 100,
            densify_threshold: 0.0002,
            prune_interval: 100,
            prune_opacity: 0.005,
            lr_position: 0.00016,
            lr_rotation: 0.002,
            lr_scale: 0.005,
            lr_opacity: 0.05,
            lr_color: 0.0025,
            batch_size: 4096,
            num_iterations: 30_000,
            use_gpu: true,
        }
    }
}

/// TSDF volume parameters for mesh extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TsdfParams {
    /// Voxel size (meters)
    pub voxel_size: f32,
    /// Truncation distance
    pub trunc_dist: f32,
    /// Volume size (meters)
    pub volume_size: f32,
    /// Marching cubes isolevel
    pub isolevel: f32,
    /// Maximum weight per voxel
    pub max_weight: u32,
}

impl Default for TsdfParams {
    fn default() -> Self {
        Self {
            voxel_size: 0.01,
            trunc_dist: 0.03,
            volume_size: 2.0,
            isolevel: 0.0,
            max_weight: 100,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracker_params_default() {
        let params = TrackerParams::default();
        assert_eq!(params.max_features, 2000);
    }

    #[test]
    fn test_mapper_params_default() {
        let params = MapperParams::default();
        assert_eq!(params.max_keyframes, 100);
    }

    #[test]
    fn test_optimizer_params_default() {
        let params = OptimizerParams::default();
        assert_eq!(params.local_ba_iterations, 20);
    }

    #[test]
    fn test_loop_closing_params_default() {
        let params = LoopClosingParams::default();
        assert_eq!(params.min_loop_score, 0.05);
    }

    #[test]
    fn test_dataset_params_default() {
        let params = DatasetParams::default();
        assert_eq!(params.dataset_type, "tum");
    }

    #[test]
    fn test_viewer_params_default() {
        let params = ViewerParams::default();
        assert!(params.enabled);
    }

    #[test]
    fn test_gaussian_splatting_params_default() {
        let params = GaussianSplattingParams::default();
        assert_eq!(params.max_gaussians, 100_000);
    }

    #[test]
    fn test_tsdf_params_default() {
        let params = TsdfParams::default();
        assert_eq!(params.voxel_size, 0.01);
    }
}
