//! Gaussian initialization from SLAM sparse map points.
//!
//! This module provides utilities to create initial Gaussians for 3DGS
//! training directly from the SLAM `Map` points.

use candle_core::Device;
use glam::Vec3;
use kiddo::{KdTree, SquaredEuclidean};

use crate::core::Map;
use crate::fusion::diff_splat::TrainableGaussians;
use crate::fusion::tiled_renderer::Gaussian;

/// Configuration for Gaussian initialization from SLAM points.
#[derive(Debug, Clone)]
pub struct GaussianInitConfig {
    /// Minimum scale (meters).
    pub min_scale: f32,
    /// Maximum scale (meters).
    pub max_scale: f32,
    /// Scale factor applied to nearest-neighbor distance.
    pub scale_factor: f32,
    /// Default color when map point color is unavailable (RGB, 0-1).
    pub default_color: [f32; 3],
    /// Default opacity for initialized Gaussians.
    pub opacity: f32,
}

impl Default for GaussianInitConfig {
    fn default() -> Self {
        Self {
            min_scale: 0.005,
            max_scale: 0.2,
            scale_factor: 0.5,
            default_color: [0.5, 0.5, 0.5],
            opacity: 0.5,
        }
    }
}

/// Initialize Gaussians from SLAM map points.
pub fn initialize_gaussians_from_map(
    map: &Map,
    config: &GaussianInitConfig,
) -> Vec<Gaussian> {
    let mut positions: Vec<Vec3> = Vec::new();
    let mut colors: Vec<Option<[f32; 3]>> = Vec::new();

    for mp in map.valid_points() {
        positions.push(mp.position);
        colors.push(mp.color);
    }

    if positions.is_empty() {
        return Vec::new();
    }

    let scales = compute_scales(&positions, config);

    positions
        .iter()
        .zip(scales.iter())
        .zip(colors.iter())
        .map(|((pos, scale), color)| {
            let rgb = color.unwrap_or(config.default_color);
            Gaussian::new(
                [pos.x, pos.y, pos.z],
                [*scale, *scale, *scale],
                [1.0, 0.0, 0.0, 0.0],
                config.opacity,
                rgb,
            )
        })
        .collect()
}

/// Initialize trainable Gaussians (for differentiable training) from map points.
pub fn initialize_trainable_gaussians_from_map(
    map: &Map,
    config: &GaussianInitConfig,
    device: &Device,
) -> candle_core::Result<TrainableGaussians> {
    let gaussians = initialize_gaussians_from_map(map, config);
    let n = gaussians.len();

    let mut positions = Vec::with_capacity(n * 3);
    let mut scales = Vec::with_capacity(n * 3);
    let mut rotations = Vec::with_capacity(n * 4);
    let mut opacities = Vec::with_capacity(n);
    let mut colors = Vec::with_capacity(n * 3);

    for g in gaussians {
        positions.extend_from_slice(&g.position);
        scales.extend_from_slice(&[
            g.scale[0].ln(),
            g.scale[1].ln(),
            g.scale[2].ln(),
        ]);
        rotations.extend_from_slice(&g.rotation);
        opacities.push(opacity_to_logit(g.opacity));
        colors.extend_from_slice(&g.color);
    }

    TrainableGaussians::new(
        &positions,
        &scales,
        &rotations,
        &opacities,
        &colors,
        device,
    )
}

fn compute_scales(points: &[Vec3], config: &GaussianInitConfig) -> Vec<f32> {
    if points.len() == 1 {
        return vec![config.min_scale];
    }

    let mut tree: KdTree<f32, 3> = KdTree::new();
    for (idx, pos) in points.iter().enumerate() {
        tree.add(&[pos.x, pos.y, pos.z], idx as u64);
    }

    let mut scales = Vec::with_capacity(points.len());
    for (idx, pos) in points.iter().enumerate() {
        let query = [pos.x, pos.y, pos.z];
        let neighbors = tree.nearest_n::<SquaredEuclidean>(&query, 2);

        let mut nearest = None;
        for n in neighbors {
            if n.item as usize != idx {
                nearest = Some(n.distance);
                break;
            }
        }

        let dist = nearest.map(|d| d.sqrt()).unwrap_or(config.min_scale);
        let scale = (dist * config.scale_factor)
            .clamp(config.min_scale, config.max_scale);
        scales.push(scale);
    }

    scales
}

fn opacity_to_logit(opacity: f32) -> f32 {
    let clamped = opacity.clamp(1e-6, 1.0 - 1e-6);
    (clamped / (1.0 - clamped)).ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    use crate::core::{Map, MapPoint};

    #[test]
    fn test_initialize_gaussians_from_map_scale_and_color() {
        let mut map = Map::new();

        let mut mp1 = MapPoint::new(0, Vec3::ZERO, 0);
        mp1.set_color([0.2, 0.3, 0.4]);
        map.add_point(mp1);

        let mut mp2 = MapPoint::new(1, Vec3::new(1.0, 0.0, 0.0), 0);
        mp2.set_color([0.9, 0.1, 0.2]);
        map.add_point(mp2);

        let config = GaussianInitConfig {
            min_scale: 0.1,
            max_scale: 1.0,
            scale_factor: 0.5,
            default_color: [0.5, 0.5, 0.5],
            opacity: 0.5,
        };

        let gaussians = initialize_gaussians_from_map(&map, &config);
        assert_eq!(gaussians.len(), 2);

        for g in &gaussians {
            assert!((g.scale[0] - 0.5).abs() < 1e-6);
            assert_eq!(g.scale[0], g.scale[1]);
            assert_eq!(g.scale[1], g.scale[2]);
            assert_eq!(g.opacity, 0.5);
        }

        assert!(gaussians.iter().any(|g| g.color == [0.2, 0.3, 0.4]));
        assert!(gaussians.iter().any(|g| g.color == [0.9, 0.1, 0.2]));
    }

    #[test]
    fn test_initialize_trainable_gaussians_from_map_count() {
        let mut map = Map::new();
        let mp = MapPoint::new(0, Vec3::new(0.0, 0.0, 1.0), 0);
        map.add_point(mp);

        let config = GaussianInitConfig::default();
        let device = Device::Cpu;

        let gaussians = initialize_trainable_gaussians_from_map(&map, &config, &device)
            .expect("trainable gaussians");
        assert_eq!(gaussians.len(), 1);
    }

    #[test]
    fn test_initialize_gaussians_defaults() {
        let mut map = Map::new();
        let mp = MapPoint::new(0, Vec3::new(0.0, 0.0, 1.0), 0);
        map.add_point(mp);

        let config = GaussianInitConfig::default();
        let gaussians = initialize_gaussians_from_map(&map, &config);
        assert_eq!(gaussians.len(), 1);

        let g = &gaussians[0];
        assert_eq!(g.rotation, [1.0, 0.0, 0.0, 0.0]);
        assert_eq!(g.opacity, config.opacity);
        assert_eq!(g.color, config.default_color);
        assert_eq!(g.scale[0], config.min_scale);
        assert_eq!(g.scale[1], config.min_scale);
        assert_eq!(g.scale[2], config.min_scale);
    }
}
