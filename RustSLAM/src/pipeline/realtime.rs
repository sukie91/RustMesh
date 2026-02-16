//! Real-time Multi-threaded SLAM Pipeline
//!
//! Implements a three-thread architecture for real-time processing:
//! - Tracking Thread: High priority, processes frames at 30-60 FPS
//! - Mapping Thread: Medium priority, processes keyframes at 5-10 FPS
//! - Optimization Thread: Low priority, runs BA and 3DGS training at 1-2 FPS
//!
//! Architecture:
//! ```
//! ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
//! │ Tracking     │ ──> │ Mapping      │ ──> │ Optimization │
//! │ (30-60 FPS) │     │ (5-10 FPS)   │     │ (1-2 FPS)   │
//! └──────────────┘     └──────────────┘     └──────────────┘
//!    高优先级              中优先级              低优先级
//! ```

use crossbeam_channel::{bounded, Sender, Receiver};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread;
use std::time::Duration;

use crate::core::{Camera, SE3};
use crate::fusion::GaussianMapper;
use crate::io::{Dataset, Frame};

/// Message sent from Tracking thread to Mapping thread
#[derive(Debug, Clone)]
pub struct TrackingMessage {
    /// Frame data
    pub frame: Frame,
    /// Estimated pose
    pub pose: SE3,
    /// Number of inliers
    pub num_inliers: usize,
    /// Whether tracking is successful
    pub success: bool,
}

/// Message sent from Mapping thread to Optimization thread
#[derive(Debug, Clone)]
pub struct MappingMessage {
    /// Keyframe index
    pub keyframe_index: usize,
    /// Keyframe pose
    pub pose: SE3,
    /// Camera intrinsics
    pub camera: Camera,
}

/// Pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Maximum frames in flight for tracking -> mapping channel
    pub track_map_channel_size: usize,
    /// Maximum keyframes in mapping -> optimization channel
    pub map_opt_channel_size: usize,
    /// Keyframe insertion interval (frames)
    pub keyframe_interval: usize,
    /// Whether to enable optimization thread
    pub enable_optimization: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            track_map_channel_size: 16,
            map_opt_channel_size: 8,
            keyframe_interval: 5,
            enable_optimization: true,
        }
    }
}

/// Pipeline state
#[derive(Debug, Clone, PartialEq)]
pub enum PipelineState {
    /// Pipeline is not started
    Stopped,
    /// Pipeline is initializing
    Initializing,
    /// Pipeline is running
    Running,
    /// Pipeline encountered an error
    Error(String),
}

/// Real-time SLAM Pipeline
pub struct RealtimePipeline {
    /// Handle for tracking thread
    tracking_thread: Option<thread::JoinHandle<()>>,
    /// Handle for mapping thread
    mapping_thread: Option<thread::JoinHandle<()>>,
    /// Handle for optimization thread
    optimization_thread: Option<thread::JoinHandle<()>>,
    /// Stop flag shared across threads
    stop_flag: Arc<AtomicBool>,
    /// Current pipeline state
    state: PipelineState,
    /// Configuration
    config: PipelineConfig,
}

impl RealtimePipeline {
    /// Create a new pipeline with default configuration
    pub fn new() -> Self {
        Self::with_config(PipelineConfig::default())
    }

    /// Create a new pipeline with custom configuration
    pub fn with_config(config: PipelineConfig) -> Self {
        Self {
            tracking_thread: None,
            mapping_thread: None,
            optimization_thread: None,
            stop_flag: Arc::new(AtomicBool::new(false)),
            state: PipelineState::Stopped,
            config,
        }
    }

    /// Start the pipeline with a dataset
    pub fn start<D: Dataset + Send + 'static>(&mut self, dataset: D)
    where
        D: Dataset,
    {
        self.stop_flag.store(false, Ordering::SeqCst);

        // Create channels
        let (track_tx, track_rx) = bounded::<TrackingMessage>(self.config.track_map_channel_size);
        let (map_tx, map_rx) = bounded::<MappingMessage>(self.config.map_opt_channel_size);

        let keyframe_interval = self.config.keyframe_interval;
        let enable_optimization = self.config.enable_optimization;
        let camera = dataset.camera();

        // Clone stop flag for each thread
        let stop_tracking = Arc::clone(&self.stop_flag);
        let stop_mapping = Arc::clone(&self.stop_flag);
        let stop_optimization = Arc::clone(&self.stop_flag);

        // Spawn tracking thread
        let tracking_thread = thread::spawn(move || {
            tracking_thread_main(dataset, track_tx, stop_tracking);
        });

        // Spawn mapping thread
        let mapping_thread = thread::spawn(move || {
            mapping_thread_main(camera.clone(), track_rx, map_tx, stop_mapping, keyframe_interval);
        });

        // Spawn optimization thread (optional)
        let optimization_thread = if enable_optimization {
            Some(thread::spawn(move || {
                optimization_thread_main(map_rx, stop_optimization);
            }))
        } else {
            None
        };

        self.tracking_thread = Some(tracking_thread);
        self.mapping_thread = Some(mapping_thread);
        self.optimization_thread = optimization_thread;
        self.state = PipelineState::Running;
    }

    /// Stop the pipeline
    pub fn stop(&mut self) {
        self.stop_flag.store(true, Ordering::SeqCst);

        if let Some(handle) = self.tracking_thread.take() {
            let _ = handle.join();
        }
        if let Some(handle) = self.mapping_thread.take() {
            let _ = handle.join();
        }
        if let Some(handle) = self.optimization_thread.take() {
            let _ = handle.join();
        }

        self.state = PipelineState::Stopped;
    }

    /// Get current pipeline state
    pub fn state(&self) -> &PipelineState {
        &self.state
    }
}

impl Default for RealtimePipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for RealtimePipeline {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Tracking thread main function
fn tracking_thread_main<D: Dataset>(
    mut dataset: D,
    track_tx: Sender<TrackingMessage>,
    stop_flag: Arc<AtomicBool>,
) {
    let camera = dataset.camera();
    let mut vo = crate::tracker::VisualOdometry::new(camera);

    for frame_result in dataset.frames() {
        if stop_flag.load(Ordering::Relaxed) {
            break;
        }

        let frame = match frame_result {
            Ok(f) => f,
            Err(_) => continue,
        };

        let gray = rgb_to_grayscale(&frame.color, frame.width as usize, frame.height as usize);
        let vo_result = vo.process_frame(&gray, frame.width, frame.height);

        let msg = TrackingMessage {
            frame,
            pose: vo_result.pose,
            num_inliers: vo_result.num_inliers,
            success: vo_result.success,
        };

        // Best-effort send to keep tracking thread real-time
        let _ = track_tx.try_send(msg);
    }
}

/// Mapping thread main function
fn mapping_thread_main(
    camera: Camera,
    track_rx: Receiver<TrackingMessage>,
    map_tx: Sender<MappingMessage>,
    stop_flag: Arc<AtomicBool>,
    keyframe_interval: usize,
) {
    let mut mapper = GaussianMapper::new(
        camera.width as usize,
        camera.height as usize,
    );

    loop {
        // Check for stop signal
        if stop_flag.load(Ordering::Relaxed) {
            break;
        }

        // Try to receive tracking message
        match track_rx.recv_timeout(Duration::from_millis(100)) {
            Ok(msg) => {
                if msg.success && msg.frame.index % keyframe_interval == 0 {
                    if let Some(depth) = &msg.frame.depth {
                        let color: Vec<[u8; 3]> = msg.frame.color
                            .chunks(3)
                            .map(|c| [c[0], c[1], c[2]])
                            .collect();
                        let rot = msg.pose.rotation();
                        let t = msg.pose.translation();

                        let _ = mapper.update(
                            depth,
                            &color,
                            msg.frame.width as usize,
                            msg.frame.height as usize,
                            camera.focal.x,
                            camera.focal.y,
                            camera.principal.x,
                            camera.principal.y,
                            &rot,
                            &t,
                        );
                    }

                    let _ = map_tx.send(MappingMessage {
                        keyframe_index: msg.frame.index,
                        pose: msg.pose,
                        camera: camera.clone(),
                    });
                }
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                // Continue waiting
            }
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                break;
            }
        }
    }
}

/// Optimization thread main function
fn optimization_thread_main(
    map_rx: Receiver<MappingMessage>,
    stop_flag: Arc<AtomicBool>,
) {
    loop {
        // Check for stop signal
        if stop_flag.load(Ordering::Relaxed) {
            break;
        }

        // Try to receive mapping message
        match map_rx.recv_timeout(Duration::from_secs(1)) {
            Ok(_msg) => {
                // In real implementation:
                // 1. Run BA optimization
                // 2. Run 3DGS training step
                // 3. Densify/prune if needed
            }
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                // Continue waiting
            }
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => {
                break;
            }
        }
    }
}

fn rgb_to_grayscale(rgb: &[u8], width: usize, height: usize) -> Vec<u8> {
    let expected = width.saturating_mul(height).saturating_mul(3);
    if rgb.len() != expected {
        return Vec::new();
    }

    let mut gray = Vec::with_capacity(width * height);
    for c in rgb.chunks(3) {
        let r = c[0] as u16;
        let g = c[1] as u16;
        let b = c[2] as u16;
        let y = (30 * r + 59 * g + 11 * b) / 100;
        gray.push(y as u8);
    }
    gray
}

/// Builder for creating RealtimePipeline with fluent API
pub struct RealtimePipelineBuilder {
    config: PipelineConfig,
}

impl RealtimePipelineBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: PipelineConfig::default(),
        }
    }

    /// Set track-map channel size
    pub fn track_map_channel_size(mut self, size: usize) -> Self {
        self.config.track_map_channel_size = size;
        self
    }

    /// Set map-opt channel size
    pub fn map_opt_channel_size(mut self, size: usize) -> Self {
        self.config.map_opt_channel_size = size;
        self
    }

    /// Set keyframe interval
    pub fn keyframe_interval(mut self, interval: usize) -> Self {
        self.config.keyframe_interval = interval;
        self
    }

    /// Enable or disable optimization thread
    pub fn enable_optimization(mut self, enable: bool) -> Self {
        self.config.enable_optimization = enable;
        self
    }

    /// Build the pipeline
    pub fn build(self) -> RealtimePipeline {
        RealtimePipeline::with_config(self.config)
    }
}

impl Default for RealtimePipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_config_default() {
        let config = PipelineConfig::default();
        assert_eq!(config.track_map_channel_size, 16);
        assert_eq!(config.map_opt_channel_size, 8);
        assert_eq!(config.keyframe_interval, 5);
        assert!(config.enable_optimization);
    }

    #[test]
    fn test_pipeline_creation() {
        let pipeline = RealtimePipeline::new();
        assert_eq!(*pipeline.state(), PipelineState::Stopped);
    }

    #[test]
    fn test_pipeline_builder() {
        let pipeline = RealtimePipelineBuilder::new()
            .track_map_channel_size(32)
            .map_opt_channel_size(16)
            .keyframe_interval(10)
            .enable_optimization(false)
            .build();

        assert_eq!(*pipeline.state(), PipelineState::Stopped);
    }

    #[test]
    fn test_pipeline_state() {
        let mut pipeline = RealtimePipeline::new();
        assert_eq!(*pipeline.state(), PipelineState::Stopped);

        // Pipeline should be stoppable even if not started
        pipeline.stop();
        assert_eq!(*pipeline.state(), PipelineState::Stopped);
    }

    #[test]
    fn test_message_sending() {
        let (tx, rx) = bounded::<TrackingMessage>(1);

        let camera = Camera::new(100.0, 100.0, 1.0, 1.0, 2, 2);
        let frame = Frame::new(
            0,
            0.0,
            vec![0u8; 2 * 2 * 3],
            None,
            camera,
            None,
        );
        let msg = TrackingMessage {
            frame,
            pose: SE3::identity(),
            num_inliers: 100,
            success: true,
        };

        tx.send(msg).unwrap();

        let received = rx.recv().unwrap();
        assert!(received.success);
        assert_eq!(received.num_inliers, 100);
    }
}
