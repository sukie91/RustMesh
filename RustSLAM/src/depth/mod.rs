//! Depth estimation module
//!
//! This module provides depth estimation capabilities:
//! - Stereo matching (for stereo cameras like KITTI)
//! - Depth fusion (combining multiple depth sources)

pub mod stereo;
pub mod fusion;

#[cfg(test)]
mod additional_tests;

pub use stereo::{StereoMatcher, StereoConfig, BlockMatcher};
pub use fusion::{DepthFusion, DepthFusionConfig, DepthObservation, TemporalDepthFusion};
