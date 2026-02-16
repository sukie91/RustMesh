//! Core data structures for RustSLAM

pub mod pose;
pub mod frame;
pub mod keyframe;
pub mod map_point;
pub mod map;
pub mod camera;
pub mod keyframe_selector;

#[cfg(test)]
mod additional_tests;

pub use pose::SE3;
pub use frame::{Frame, FrameFeatures};
pub use keyframe::KeyFrame;
pub use map_point::MapPoint;
pub use map::Map;
pub use camera::Camera;
pub use keyframe_selector::{KeyframeSelector, KeyframeDecision, KeyframeCulling, KeyframeSelectorConfig};
