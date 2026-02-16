//! Tests for core module (inline)

#[cfg(test)]
mod tests {
    use crate::core::{Camera, SE3};

    #[test]
    fn test_camera_creation() {
        // Camera::new(fx, fy, cx, cy, width, height)
        let camera = Camera::new(525.0, 525.0, 319.5, 239.5, 640, 480);
        assert_eq!(camera.width, 640);
    }

    #[test]
    fn test_se3_compose() {
        let pose1 = SE3::identity();
        let pose2 = SE3::new(&[0.0, 0.0, 0.0, 1.0], &[1.0, 0.0, 0.0]);
        let composed = pose1.compose(&pose2);
        let t = composed.translation();
        assert!((t[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_se3_inverse() {
        let pose = SE3::new(&[0.0, 0.0, 0.0, 1.0], &[1.0, 2.0, 3.0]);
        let inverse = pose.inverse();
        let composed = pose.compose(&inverse);
        let t = composed.translation();
        assert!(t[0].abs() < 1e-5);
    }

    #[test]
    fn test_se3_transform_point() {
        let pose = SE3::new(&[0.0, 0.0, 0.0, 1.0], &[1.0, 0.0, 0.0]);
        let point = [1.0, 0.0, 0.0];
        let transformed = pose.transform_point(&point);
        assert!((transformed[0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_se3_quaternion() {
        let pose = SE3::new(&[0.0, 0.0, 0.0, 1.0], &[0.0, 0.0, 0.0]);
        let q = pose.quaternion();
        assert!((q[3] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_se3_rotation_matrix() {
        let pose = SE3::new(&[0.0, 0.0, 0.0, 1.0], &[0.0, 0.0, 0.0]);
        let r = pose.rotation_matrix();
        assert!((r[0][0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_se3_exp_log() {
        let tangent = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0];
        let pose = SE3::exp(&tangent);
        let recovered = pose.log();
        assert!((recovered[0] - 0.1).abs() < 1e-3);
    }

    #[test]
    fn test_se3_identity() {
        let pose = SE3::identity();
        let t = pose.translation();
        assert!(t[0].abs() < 1e-5);
        assert!(t[1].abs() < 1e-5);
        assert!(t[2].abs() < 1e-5);
    }
}
