//! # Geometry
//! 
//! Geometric utilities for mesh operations.

pub type Point = glam::Vec3;
pub type Vector = glam::Vec3;
pub type Normal = glam::Vec3;

/// Calculate the centroid of a triangle
#[inline]
pub fn triangle_centroid(p0: Point, p1: Point, p2: Point) -> Point {
    (p0 + p1 + p2) / 3.0
}

/// Calculate the area of a triangle using cross product
#[inline]
pub fn triangle_area(p0: Point, p1: Point, p2: Point) -> f32 {
    (p1 - p0).cross(p2 - p0).length() * 0.5
}

/// Calculate the normal of a triangle
#[inline]
pub fn triangle_normal(p0: Point, p1: Point, p2: Point) -> Vector {
    (p1 - p0).cross(p2 - p0).normalize()
}

/// Calculate the bounding box of a point set
#[inline]
pub fn bounding_box(points: &[Point]) -> (Point, Point) {
    if points.is_empty() {
        return (Point::ZERO, Point::ZERO);
    }
    
    let mut min = points[0];
    let mut max = points[0];
    
    for &p in points {
        min = min.min(p);
        max = max.max(p);
    }
    
    (min, max)
}

/// Calculate the center of mass of a point set
#[inline]
pub fn center_of_mass(points: &[Point]) -> Point {
    if points.is_empty() {
        return Point::ZERO;
    }
    
    let sum: Point = points.iter().fold(Point::ZERO, |acc, &p| acc + p);
    sum / points.len() as f32
}

/// Calculate the squared distance between two points
#[inline]
pub fn squared_distance(p0: Point, p1: Point) -> f32 {
    (p0 - p1).length_squared()
}

/// Calculate the distance between two points
#[inline]
pub fn distance(p0: Point, p1: Point) -> f32 {
    (p0 - p1).length()
}

/// Normalize a vector
#[inline]
pub fn normalize(v: Vector) -> Vector {
    v.normalize()
}

/// Calculate the dot product of two vectors
#[inline]
pub fn dot(v0: Vector, v1: Vector) -> f32 {
    v0.dot(v1)
}

/// Calculate the cross product of two vectors
#[inline]
pub fn cross(v0: Vector, v1: Vector) -> Vector {
    v0.cross(v1)
}

/// Calculate the length of a vector
#[inline]
pub fn length(v: Vector) -> f32 {
    v.length()
}

/// Calculate the squared length of a vector
#[inline]
pub fn squared_length(v: Vector) -> f32 {
    v.length_squared()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangle_area() {
        let p0 = glam::vec3(0.0, 0.0, 0.0);
        let p1 = glam::vec3(1.0, 0.0, 0.0);
        let p2 = glam::vec3(0.0, 1.0, 0.0);
        
        // Right triangle with legs of length 1
        assert!((triangle_area(p0, p1, p2) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_triangle_normal() {
        let p0 = glam::vec3(0.0, 0.0, 0.0);
        let p1 = glam::vec3(1.0, 0.0, 0.0);
        let p2 = glam::vec3(0.0, 1.0, 0.0);
        
        let normal = triangle_normal(p0, p1, p2);
        assert!((normal.length() - 1.0).abs() < 1e-6);
        // Should point in +Z direction
        assert!(normal.z > 0.0);
    }

    #[test]
    fn test_bounding_box() {
        let points = [
            glam::vec3(0.0, 0.0, 0.0),
            glam::vec3(1.0, 2.0, 3.0),
            glam::vec3(-1.0, 5.0, -2.0),
        ];
        
        let (min, max) = bounding_box(&points);
        
        assert_eq!(min, glam::vec3(-1.0, 0.0, -2.0));
        assert_eq!(max, glam::vec3(1.0, 5.0, 3.0));
    }
}
