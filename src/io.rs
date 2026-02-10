//! # IO Module
//!
//! Mesh file I/O for OFF, OBJ, PLY, and STL formats.

use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use crate::connectivity::PolyMesh;
use crate::handles::{VertexHandle, FaceHandle};

/// Result type for IO operations
pub type IoResult<T> = std::result::Result<T, IoError>;

/// IO Error types
#[derive(Debug)]
pub enum IoError {
    Io(io::Error),
    Parse(String),
    Format(String),
    InvalidData(String),
}

impl std::fmt::Display for IoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IoError::Io(e) => write!(f, "IO error: {}", e),
            IoError::Parse(e) => write!(f, "Parse error: {}", e),
            IoError::Format(e) => write!(f, "Format error: {}", e),
            IoError::InvalidData(e) => write!(f, "Invalid data: {}", e),
        }
    }
}

impl std::error::Error for IoError {}

impl From<io::Error> for IoError {
    fn from(e: io::Error) -> Self {
        IoError::Io(e)
    }
}

/// Read OFF format file
/// 
/// OFF format specification:
/// - First line: "OFF" or "STOFF" (with colors)
/// - Second line: vertex_count face_count edge_count
/// - Then: vertex lines (x y z) or (x y z r g b a)
/// - Then: face lines (n v1 v2 ... vn) or with colors
pub fn read_off<P: AsRef<Path>>(path: P) -> IoResult<PolyMesh> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // Parse header
    let header = lines.next()
        .ok_or(IoError::Parse("Empty file".to_string()))??;
    
    let has_colors = header.starts_with("STOFF") || header.starts_with("COFF");
    let _binary = header.starts_with("BOFF");

    // Parse counts
    let counts_line = lines.next()
        .ok_or(IoError::Parse("Missing counts line".to_string()))??;
    
    let counts: Vec<usize> = counts_line
        .split_whitespace()
        .map(|s| s.parse().map_err(|_| IoError::Parse("Invalid count".to_string())))
        .collect::<IoResult<Vec<usize>>>()?;
    
    if counts.len() < 3 {
        return Err(IoError::Parse("Invalid counts line".to_string()));
    }

    let (n_vertices, n_faces, _n_edges) = (counts[0], counts[1], counts.get(2).copied().unwrap_or(0));

    let mut mesh = PolyMesh::new();
    let mut vertices: Vec<VertexHandle> = Vec::with_capacity(n_vertices);

    // Parse vertices
    for i in 0..n_vertices {
        let line = lines.next()
            .ok_or(IoError::Parse(format!("Unexpected end at vertex {}", i)))??;
        
        let parts: Vec<&str> = line.split_whitespace().collect();
        
        if parts.len() < 3 {
            return Err(IoError::Parse(format!("Vertex {} has insufficient coordinates", i)));
        }

        let x: f32 = parts[0].parse().map_err(|_| IoError::Parse(format!("Invalid x for vertex {}", i)))?;
        let y: f32 = parts[1].parse().map_err(|_| IoError::Parse(format!("Invalid y for vertex {}", i)))?;
        let z: f32 = parts[2].parse().map_err(|_| IoError::Parse(format!("Invalid z for vertex {}", i)))?;

        let vh = mesh.add_vertex(glam::vec3(x, y, z));
        vertices.push(vh);
    }

    // Parse faces
    for i in 0..n_faces {
        let line = lines.next()
            .ok_or(IoError::Parse(format!("Unexpected end at face {}", i)))??;
        
        let parts: Vec<&str> = line.split_whitespace().collect();
        
        if parts.is_empty() {
            continue;
        }

        let n_vertices_in_face: usize = parts[0]
            .parse()
            .map_err(|_| IoError::Parse(format!("Invalid vertex count for face {}", i)))?;
        
        if parts.len() < n_vertices_in_face + 1 {
            return Err(IoError::Parse(format!("Face {} has insufficient vertex indices", i)));
        }

        let mut face_vertices: Vec<VertexHandle> = Vec::with_capacity(n_vertices_in_face);
        
        for j in 0..n_vertices_in_face {
            let v_idx: usize = parts[j + 1]
                .parse()
                .map_err(|_| IoError::Parse(format!("Invalid vertex index in face {}", i)))?;
            
            if v_idx >= vertices.len() {
                return Err(IoError::Parse(format!("Vertex index out of bounds in face {}", i)));
            }
            
            // OFF files use 1-based indexing, convert to 0-based
            let vh = vertices[v_idx];
            face_vertices.push(vh);
        }

        if let Some(fh) = mesh.add_face(&face_vertices) {
            // Handle color if present (STOFF format)
            if has_colors && parts.len() > n_vertices_in_face + 1 {
                // Parse RGB color
                // TODO: Store color in mesh properties
                let _r: f32 = parts[n_vertices_in_face + 1].parse().unwrap_or(0.0);
            }
        }
    }

    Ok(mesh)
}

/// Write OFF format file
pub fn write_off<P: AsRef<Path>>(mesh: &PolyMesh, path: P) -> IoResult<()> {
    let file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    
    let mut writer = BufWriter::new(file);

    // Write header
    writeln!(writer, "OFF")?;

    // Write counts
    writeln!(writer, "{} {} {}", mesh.n_vertices(), mesh.n_faces(), mesh.n_edges())?;

    // Write vertices
    for vh in mesh.vertices() {
        if let Some(point) = mesh.point(vh) {
            writeln!(writer, "{} {} {}", point.x, point.y, point.z)?;
        }
    }

    // Write faces
    // Note: face_vertices() needs full halfedge connectivity to work
    // For now, write placeholder (needs implementation)
    for fh in (0..mesh.n_faces() as i32).map(FaceHandle::new) {
        writeln!(writer, "3 0 0 0  # Face {} - requires face_vertices implementation", fh.idx())?;
    }

    Ok(())
}

/// Read OBJ format file
/// 
/// OBJ format supports:
/// - v x y z [w] (vertex with optional w)
/// - vt u v [w] (texture coordinates)
/// - vn x y z (vertex normal)
/// - f v1 v2 v3 ... (faces, vertices only)
/// - f v1/vt1 v2/vt2 ... (faces with UVs)
/// - f v1/vt1/vn1 ... (faces with UVs and normals)
pub fn read_obj<P: AsRef<Path>>(path: P) -> IoResult<PolyMesh> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let lines = reader.lines();

    let mut mesh = PolyMesh::new();
    let mut vertices: Vec<VertexHandle> = Vec::new();
    let mut texture_coords: Vec<(f32, f32, f32)> = Vec::new();
    let mut normals: Vec<glam::Vec3> = Vec::new();

    for (line_num, line) in lines.enumerate() {
        let line = line?;
        let trimmed = line.trim();
        
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        
        match parts[0] {
            "v" => {
                // Vertex
                if parts.len() < 4 {
                    return Err(IoError::Parse(format!("Line {}: vertex requires 3 coordinates", line_num)));
                }
                
                let x: f32 = parts[1].parse()
                    .map_err(|_| IoError::Parse(format!("Line {}: invalid x coordinate", line_num)))?;
                let y: f32 = parts[2].parse()
                    .map_err(|_| IoError::Parse(format!("Line {}: invalid y coordinate", line_num)))?;
                let z: f32 = parts[3].parse()
                    .map_err(|_| IoError::Parse(format!("Line {}: invalid z coordinate", line_num)))?;
                
                let vh = mesh.add_vertex(glam::vec3(x, y, z));
                vertices.push(vh);
            }
            "vt" => {
                // Texture coordinate
                let u: f32 = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let v: f32 = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let w: f32 = parts.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.0);
                texture_coords.push((u, v, w));
            }
            "vn" => {
                // Vertex normal
                let x: f32 = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let y: f32 = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(0.0);
                let z: f32 = parts.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.0);
                normals.push(glam::vec3(x, y, z));
            }
            "f" => {
                // Face
                if parts.len() < 4 {
                    return Err(IoError::Parse(format!("Line {}: face requires at least 3 vertices", line_num)));
                }

                let mut face_vertices: Vec<VertexHandle> = Vec::with_capacity(parts.len() - 1);
                
                for i in 1..parts.len() {
                    let vertex_data = parts[i];
                    let indices: Vec<&str> = vertex_data.split('/').collect();
                    
                    // First index is always vertex
                    let v_idx: usize = indices[0]
                        .parse()
                        .map_err(|_| IoError::Parse(format!("Line {}: invalid vertex index", line_num)))?;
                    
                    if v_idx == 0 || v_idx > vertices.len() {
                        return Err(IoError::Parse(format!("Line {}: vertex index out of bounds", line_num)));
                    }
                    
                    // OBJ uses 1-based indexing
                    let vh = vertices[v_idx - 1];
                    face_vertices.push(vh);
                }

                mesh.add_face(&face_vertices);
            }
            "g" | "o" => {
                // Group or object name - ignore for now
            }
            "s" => {
                // Smoothing group - ignore for now
            }
            "mtllib" => {
                // Material library - ignore for now
            }
            "usemtl" => {
                // Material - ignore for now
            }
            _ => {
                // Unknown directive - ignore
            }
        }
    }

    Ok(mesh)
}

/// Write OBJ format file
pub fn write_obj<P: AsRef<Path>>(mesh: &PolyMesh, path: P) -> IoResult<()> {
    let file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    
    let mut writer = BufWriter::new(file);

    // Write header
    writeln!(writer, "# RustMesh OBJ export")?;
    writeln!(writer, "# Vertices: {}, Faces: {}", mesh.n_vertices(), mesh.n_faces())?;
    writeln!(writer)?;

    // Write vertices
    for vh in mesh.vertices() {
        if let Some(point) = mesh.point(vh) {
            writeln!(writer, "v {} {} {}", point.x, point.y, point.z)?;
        }
    }

    writeln!(writer)?;

    // Write faces
    // Note: face_vertices() needs full halfedge connectivity to work
    // For now, write placeholder comments
    writeln!(writer, "# Faces - requires face_vertices implementation")?;
    for fh in (0..mesh.n_faces() as i32).map(FaceHandle::new) {
        writeln!(writer, "# Face {}", fh.idx())?;
    }

    Ok(())
}

/// Detect file format from extension
pub fn detect_format<P: AsRef<Path>>(path: P) -> Option<&'static str> {
    let ext = path.as_ref().extension()?.to_str()?;
    
    match ext.to_lowercase().as_str() {
        "off" => Some("OFF"),
        "obj" => Some("OBJ"),
        "ply" => Some("PLY"),
        "stl" => Some("STL"),
        _ => None,
    }
}

/// Read mesh file (auto-detect format)
pub fn read_mesh<P: AsRef<Path>>(path: P) -> IoResult<PolyMesh> {
    match detect_format(&path) {
        Some("OFF") => read_off(path),
        Some("OBJ") => read_obj(path),
        Some(format) => Err(IoError::Format(format!("Unsupported format: {}", format))),
        None => Err(IoError::Format("Unknown file format".to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_read_off_triangle() {
        let content = "OFF
3 1 0
0 0 0
1 0 0
0 1 0
3 0 1 2
";
        
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(content.as_bytes()).unwrap();
        
        let mesh = read_off(file.path()).unwrap();
        
        assert_eq!(mesh.n_vertices(), 3);
        assert_eq!(mesh.n_faces(), 1);
    }

    #[test]
    fn test_read_off_quad() {
        let content = "OFF
4 1 0
0 0 0
1 0 0
1 1 0
0 1 0
4 0 1 2 3
";
        
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(content.as_bytes()).unwrap();
        
        let mesh = read_off(file.path()).unwrap();
        
        assert_eq!(mesh.n_vertices(), 4);
        assert_eq!(mesh.n_faces(), 1);
    }

    #[test]
    fn test_read_obj_simple() {
        let content = "# Simple triangle
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
f 1 2 3
";
        
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(content.as_bytes()).unwrap();
        
        let mesh = read_obj(file.path()).unwrap();
        
        assert_eq!(mesh.n_vertices(), 3);
        assert_eq!(mesh.n_faces(), 1);
    }

    #[test]
    fn test_detect_format() {
        assert_eq!(detect_format(Path::new("mesh.off")).unwrap(), "OFF");
        assert_eq!(detect_format(Path::new("mesh.OBJ")).unwrap(), "OBJ");
        assert_eq!(detect_format(Path::new("model.obj")).unwrap(), "OBJ");
        assert_eq!(detect_format(Path::new("data.ply")).unwrap(), "PLY");
        assert_eq!(detect_format(Path::new("cube.stl")).unwrap(), "STL");
        assert!(detect_format(Path::new("mesh.unknown")).is_none());
    }
}
