//! Configuration module for RustSLAM
//!
//! This module provides configuration management for all SLAM components.

pub mod config;
pub mod params;

pub use config::{SlamConfig, ConfigLoader};
pub use params::{
    TrackerParams, MapperParams, OptimizerParams, LoopClosingParams,
    DatasetParams, ViewerParams,
};