//! Main configuration structures for RustSLAM

use serde::{Deserialize, Serialize};
use std::path::Path;
use thiserror::Error;

use super::params::*;

/// Configuration loading errors
#[derive(Error, Debug)]
pub enum ConfigError {
    #[error("Failed to read config file: {0}")]
    IoError(#[from] std::io::Error),
    #[error("Failed to parse YAML: {0}")]
    YamlError(#[from] serde_yaml::Error),
    #[error("Failed to parse TOML: {0}")]
    TomlError(#[from] toml::de::Error),
    #[error("Failed to serialize TOML: {0}")]
    TomlSerializeError(#[from] toml::ser::Error),
    #[error("Unsupported config format: {0}")]
    UnsupportedFormat(String),
}

/// Main SLAM configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlamConfig {
    /// Camera configuration
    pub camera: CameraConfig,
    /// Tracker parameters
    pub tracker: TrackerParams,
    /// Mapper parameters
    pub mapper: MapperParams,
    /// Optimizer parameters
    pub optimizer: OptimizerParams,
    /// Loop closing parameters
    pub loop_closing: LoopClosingParams,
    /// Dataset parameters
    pub dataset: DatasetParams,
    /// Viewer parameters
    pub viewer: ViewerParams,
}

impl Default for SlamConfig {
    fn default() -> Self {
        Self {
            camera: CameraConfig::default(),
            tracker: TrackerParams::default(),
            mapper: MapperParams::default(),
            optimizer: OptimizerParams::default(),
            loop_closing: LoopClosingParams::default(),
            dataset: DatasetParams::default(),
            viewer: ViewerParams::default(),
        }
    }
}

/// Camera configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraConfig {
    /// Camera width
    pub width: u32,
    /// Camera height
    pub height: u32,
    /// Focal length x
    pub fx: f32,
    /// Focal length y
    pub fy: f32,
    /// Principal point x
    pub cx: f32,
    /// Principal point y
    pub cy: f32,
    /// Camera model
    pub model: String,
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            width: 640,
            height: 480,
            fx: 525.0,
            fy: 525.0,
            cx: 319.5,
            cy: 239.5,
            model: "pinhole".to_string(),
        }
    }
}

impl CameraConfig {
    /// Convert to core Camera struct
    pub fn to_camera(&self) -> crate::core::Camera {
        crate::core::Camera::new(self.fx, self.fy, self.cx, self.cy, self.width, self.height)
    }
}

/// Configuration loader supporting YAML and TOML
pub struct ConfigLoader;

impl ConfigLoader {
    /// Load configuration from file
    ///
    /// # Arguments
    /// * `path` - Path to configuration file
    ///
    /// # Returns
    /// SlamConfig if successful
    pub fn load<P: AsRef<Path>>(path: P) -> Result<SlamConfig, ConfigError> {
        let path = path.as_ref();
        let extension = path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        match extension.to_lowercase().as_str() {
            "yaml" | "yml" => Self::load_yaml(path),
            "toml" => Self::load_toml(path),
            _ => Err(ConfigError::UnsupportedFormat(extension.to_string())),
        }
    }

    /// Load configuration from YAML file
    pub fn load_yaml<P: AsRef<Path>>(path: P) -> Result<SlamConfig, ConfigError> {
        let content = std::fs::read_to_string(path)?;
        let config: SlamConfig = serde_yaml::from_str(&content)?;
        Ok(config)
    }

    /// Load configuration from TOML file
    pub fn load_toml<P: AsRef<Path>>(path: P) -> Result<SlamConfig, ConfigError> {
        let content = std::fs::read_to_string(path)?;
        let config: SlamConfig = toml::from_str(&content)?;
        Ok(config)
    }

    /// Save configuration to YAML file
    pub fn save_yaml<P: AsRef<Path>>(config: &SlamConfig, path: P) -> Result<(), ConfigError> {
        let content = serde_yaml::to_string(config)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Save configuration to TOML file
    pub fn save_toml<P: AsRef<Path>>(config: &SlamConfig, path: P) -> Result<(), ConfigError> {
        let content = toml::to_string(config)?;
        std::fs::write(path, content)?;
        Ok(())
    }
}

impl SlamConfig {
    /// Create default configuration for TUM dataset
    pub fn tum_rgbd() -> Self {
        Self {
            camera: CameraConfig {
                width: 640,
                height: 480,
                fx: 525.0,
                fy: 525.0,
                cx: 319.5,
                cy: 239.5,
                model: "pinhole".to_string(),
            },
            tracker: TrackerParams {
                max_features: 2000,
                min_features: 500,
                pyramid_levels: 8,
                patch_size: 31,
                ..Default::default()
            },
            mapper: MapperParams::default(),
            optimizer: OptimizerParams::default(),
            loop_closing: LoopClosingParams::default(),
            dataset: DatasetParams {
                dataset_type: "tum".to_string(),
                ..Default::default()
            },
            viewer: ViewerParams::default(),
        }
    }

    /// Create default configuration for KITTI dataset
    pub fn kitti() -> Self {
        Self {
            camera: CameraConfig {
                width: 1241,
                height: 376,
                fx: 718.856,
                fy: 718.856,
                cx: 607.1928,
                cy: 185.2157,
                model: "pinhole".to_string(),
            },
            tracker: TrackerParams {
                max_features: 2000,
                min_features: 500,
                pyramid_levels: 4,
                patch_size: 31,
                ..Default::default()
            },
            mapper: MapperParams::default(),
            optimizer: OptimizerParams::default(),
            loop_closing: LoopClosingParams::default(),
            dataset: DatasetParams {
                dataset_type: "kitti".to_string(),
                ..Default::default()
            },
            viewer: ViewerParams::default(),
        }
    }

    /// Create default configuration for EuRoC dataset
    pub fn euroc() -> Self {
        Self {
            camera: CameraConfig {
                width: 752,
                height: 480,
                fx: 458.654,
                fy: 457.296,
                cx: 367.215,
                cy: 248.375,
                model: "pinhole".to_string(),
            },
            tracker: TrackerParams {
                max_features: 1200,
                min_features: 400,
                pyramid_levels: 8,
                patch_size: 31,
                ..Default::default()
            },
            mapper: MapperParams::default(),
            optimizer: OptimizerParams::default(),
            loop_closing: LoopClosingParams::default(),
            dataset: DatasetParams {
                dataset_type: "euroc".to_string(),
                ..Default::default()
            },
            viewer: ViewerParams::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_config_default() {
        let config = SlamConfig::default();
        assert_eq!(config.camera.width, 640);
    }

    #[test]
    fn test_config_tum() {
        let config = SlamConfig::tum_rgbd();
        assert_eq!(config.dataset.dataset_type, "tum");
    }

    #[test]
    fn test_config_kitti() {
        let config = SlamConfig::kitti();
        assert_eq!(config.dataset.dataset_type, "kitti");
    }

    #[test]
    fn test_config_euroc() {
        let config = SlamConfig::euroc();
        assert_eq!(config.dataset.dataset_type, "euroc");
    }

    #[test]
    fn test_load_yaml() {
        // Test by loading a full config from default
        let config = SlamConfig::default();
        let yaml = serde_yaml::to_string(&config).unwrap();
        let loaded: SlamConfig = serde_yaml::from_str(&yaml).unwrap();
        assert_eq!(loaded.camera.width, 640);
    }

    #[test]
    fn test_save_load_yaml() {
        let config = SlamConfig::tum_rgbd();
        let temp_file = NamedTempFile::new().unwrap();

        ConfigLoader::save_yaml(&config, temp_file.path()).unwrap();
        let loaded = ConfigLoader::load_yaml(temp_file.path()).unwrap();

        assert_eq!(loaded.camera.width, config.camera.width);
    }

    #[test]
    fn test_save_load_toml() {
        let config = SlamConfig::kitti();
        let temp_file = NamedTempFile::new().unwrap();

        ConfigLoader::save_toml(&config, temp_file.path()).unwrap();
        let loaded = ConfigLoader::load_toml(temp_file.path()).unwrap();

        assert_eq!(loaded.camera.fx, config.camera.fx);
    }
}
