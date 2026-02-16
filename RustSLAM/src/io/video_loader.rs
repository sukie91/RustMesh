//! Video file loader for SLAM pipeline
//!
//! Uses OpenCV VideoCapture to decode common video formats (MP4/MOV/HEVC).

use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::core::Camera;
use super::{Dataset, DatasetMetadata, Frame};

#[cfg(feature = "opencv")]
use opencv::{
    core::Mat,
    imgproc,
    prelude::*,
    videoio::{VideoCapture, CAP_ANY, CAP_PROP_FPS, CAP_PROP_FRAME_COUNT, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH, CAP_PROP_POS_FRAMES},
};

/// Errors that can occur while loading video files
#[derive(Debug, Error)]
pub enum VideoError {
    #[error("invalid video path")]
    InvalidPath,
    #[error("failed to open video: {0}")]
    OpenFailed(String),
    #[error("OpenCV error: {0}")]
    OpenCv(String),
    #[error("frame index out of bounds: {0}")]
    FrameIndex(usize),
    #[error("failed to read frame: {0}")]
    ReadFrame(usize),
}

pub type Result<T> = std::result::Result<T, VideoError>;

/// Video loader backed by OpenCV
pub struct VideoLoader {
    path: PathBuf,
    fps: f64,
    width: i32,
    height: i32,
    frame_count: i32,
    camera: Camera,
    metadata: DatasetMetadata,
}

impl VideoLoader {
    /// Open a video file
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let path_str = path.to_str().ok_or(VideoError::InvalidPath)?;

        let mut capture = VideoCapture::from_file(path_str, CAP_ANY)
            .map_err(|e| VideoError::OpenCv(e.to_string()))?;

        let opened = capture
            .is_opened()
            .map_err(|e| VideoError::OpenCv(e.to_string()))?;
        if !opened {
            return Err(VideoError::OpenFailed(path_str.to_string()));
        }

        let fps = capture
            .get(CAP_PROP_FPS)
            .map_err(|e| VideoError::OpenCv(e.to_string()))?;
        let width = capture
            .get(CAP_PROP_FRAME_WIDTH)
            .map_err(|e| VideoError::OpenCv(e.to_string()))? as i32;
        let height = capture
            .get(CAP_PROP_FRAME_HEIGHT)
            .map_err(|e| VideoError::OpenCv(e.to_string()))? as i32;
        let frame_count = capture
            .get(CAP_PROP_FRAME_COUNT)
            .map_err(|e| VideoError::OpenCv(e.to_string()))? as i32;

        let fps = if fps > 0.0 { fps } else { 30.0 };
        let camera = Self::estimate_camera(width as u32, height as u32);

        let metadata = DatasetMetadata {
            name: "Video".to_string(),
            sequence: path
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_else(|| "video".to_string()),
            total_frames: frame_count.max(0) as usize,
            has_depth: false,
            has_ground_truth: false,
            frame_rate: Some(fps as f32),
            avg_speed: None,
            trajectory_length: None,
            notes: "Video file input".to_string(),
        };

        Ok(Self {
            path,
            fps,
            width,
            height,
            frame_count,
            camera,
            metadata,
        })
    }

    /// Frames per second reported by the container
    pub fn fps(&self) -> f64 {
        self.fps
    }

    /// Total number of frames (best effort)
    pub fn total_frames(&self) -> usize {
        self.frame_count.max(0) as usize
    }

    /// Estimated camera intrinsics from resolution
    pub fn estimate_camera(width: u32, height: u32) -> Camera {
        // Rough iPhone FOV estimate (~60-70 degrees)
        let fx = width as f32 * 1.2;
        let fy = height as f32 * 1.2;
        let cx = width as f32 / 2.0;
        let cy = height as f32 / 2.0;
        Camera::new(fx, fy, cx, cy, width, height)
    }

    #[cfg(feature = "opencv")]
    fn read_frame_at(&self, index: usize) -> Result<Vec<u8>> {
        let path_str = self.path.to_str().ok_or(VideoError::InvalidPath)?;
        let mut capture = VideoCapture::from_file(path_str, CAP_ANY)
            .map_err(|e| VideoError::OpenCv(e.to_string()))?;

        capture
            .set(CAP_PROP_POS_FRAMES, index as f64)
            .map_err(|e| VideoError::OpenCv(e.to_string()))?;

        let mut mat = Mat::default();
        let ok = capture
            .read(&mut mat)
            .map_err(|e| VideoError::OpenCv(e.to_string()))?;
        if !ok || mat.empty().map_err(|e| VideoError::OpenCv(e.to_string()))? {
            return Err(VideoError::ReadFrame(index));
        }

        let mut rgb = Mat::default();
        imgproc::cvt_color(&mat, &mut rgb, imgproc::COLOR_BGR2RGB, 0)
            .map_err(|e| VideoError::OpenCv(e.to_string()))?;

        let data = rgb
            .data_bytes()
            .map_err(|e| VideoError::OpenCv(e.to_string()))?;

        Ok(data.to_vec())
    }
}

impl Dataset for VideoLoader {
    fn len(&self) -> usize {
        self.total_frames()
    }

    fn get_frame(&self, index: usize) -> std::result::Result<Frame, super::DatasetError> {
        if index >= self.total_frames() {
            return Err(super::DatasetError::FrameIndex(index));
        }

        #[cfg(feature = "opencv")]
        {
            let color = self.read_frame_at(index)
                .map_err(|e| super::DatasetError::Image(e.to_string()))?;
            let timestamp = index as f64 / self.fps;

            Ok(Frame::new(
                index,
                timestamp,
                color,
                None,
                self.camera.clone(),
                None,
            ))
        }

        #[cfg(not(feature = "opencv"))]
        {
            let _ = index;
            Err(super::DatasetError::Image(
                "Video loading requires 'opencv' feature".to_string(),
            ))
        }
    }

    fn camera(&self) -> Camera {
        self.camera.clone()
    }

    fn metadata(&self) -> &DatasetMetadata {
        &self.metadata
    }
}
