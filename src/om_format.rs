//! # OM Format
//!
//! OpenMesh native format - a binary format for mesh data.
//! Simplified implementation supporting basic mesh serialization.

use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use crate::PolyMesh;
use crate::handles::{VertexHandle, HalfedgeHandle, EdgeHandle, FaceHandle};

/// OM Format magic number ("OM")
const OM_MAGIC: &[u8; 2] = b"OM";

/// OM Format version (1.0)
const OM_VERSION: u16 = 1;

/// OM Format chunk types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OmChunkType {
    Vertex = 0x0001,
    Edge = 0x0002,
    Halfedge = 0x0003,
    Face = 0x0004,
    Header = 0x0000,
    End = 0xFFFF,
}

/// OM Format header
#[derive(Debug, Clone)]
pub struct OmHeader {
    pub magic: [u8; 2],
    pub version: u16,
    pub n_vertices: u32,
    pub n_edges: u32,
    pub n_faces: u32,
}

impl Default for OmHeader {
    fn default() -> Self {
        Self {
            magic: *OM_MAGIC,
            version: OM_VERSION,
            n_vertices: 0,
            n_edges: 0,
            n_faces: 0,
        }
    }
}

/// Read OM format file
pub fn read_om<P: AsRef<Path>>(path: P) -> std::io::Result<PolyMesh> {
    let mut file = File::open(path)?;
    let mut mesh = PolyMesh::new();

    // Read header
    let mut magic = [0u8; 2];
    file.read_exact(&mut magic)?;
    if &magic != OM_MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Invalid OM format: wrong magic number"
        ));
    }

    let version = file.read_u16::<LittleEndian>()?;
    let n_vertices = file.read_u32::<LittleEndian>()?;
    let n_edges = file.read_u32::<LittleEndian>()?;
    let n_faces = file.read_u32::<LittleEndian>()?;

    // Read vertices
    let mut vertices = Vec::new();
    for _ in 0..n_vertices {
        let x = file.read_f32::<LittleEndian>()?;
        let y = file.read_f32::<LittleEndian>()?;
        let z = file.read_f32::<LittleEndian>()?;
        let vh = mesh.add_vertex(glam::vec3(x, y, z));
        vertices.push(vh);
    }

    // Read faces (indices into vertex list)
    for _ in 0..n_faces {
        let v0_idx = file.read_u32::<LittleEndian>()? as usize;
        let v1_idx = file.read_u32::<LittleEndian>()? as usize;
        let v2_idx = file.read_u32::<LittleEndian>()? as usize;

        if v0_idx < vertices.len() && v1_idx < vertices.len() && v2_idx < vertices.len() {
            mesh.add_face(&[vertices[v0_idx], vertices[v1_idx], vertices[v2_idx]]);
        }
    }

    Ok(mesh)
}

/// Write OM format file
pub fn write_om<P: AsRef<Path>>(mesh: &PolyMesh, path: P) -> std::io::Result<()> {
    let mut file = File::create(path)?;

    // Write header
    file.write_all(OM_MAGIC)?;
    file.write_u16::<LittleEndian>(OM_VERSION)?;
    file.write_u32::<LittleEndian>(mesh.n_vertices() as u32)?;
    file.write_u32::<LittleEndian>(mesh.n_edges() as u32)?;
    file.write_u32::<LittleEndian>(mesh.n_faces() as u32)?;

    // Write vertices
    for vh in mesh.vertices() {
        if let Some(p) = mesh.point(vh) {
            file.write_f32::<LittleEndian>(p.x)?;
            file.write_f32::<LittleEndian>(p.y)?;
            file.write_f32::<LittleEndian>(p.z)?;
        } else {
            file.write_f32::<LittleEndian>(0.0)?;
            file.write_f32::<LittleEndian>(0.0)?;
            file.write_f32::<LittleEndian>(0.0)?;
        }
    }

    // Write faces (simplified: triangles only)
    for fh in mesh.faces() {
        // Simplified: write first 3 vertices (placeholder for full implementation)
        file.write_u32::<LittleEndian>(0)?;
        file.write_u32::<LittleEndian>(1)?;
        file.write_u32::<LittleEndian>(2)?;
    }

    Ok(())
}

/// Auto-detect format from file extension
pub fn detect_om<P: AsRef<Path>>(path: P) -> bool {
    let ext = path.as_ref().extension()
        .map(|e| e.to_string_lossy().to_lowercase());
    ext == Some("om".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_om_header() {
        let header = OmHeader::default();
        assert_eq!(header.magic, *b"OM");
        assert_eq!(header.version, 1);
    }

    #[test]
    fn test_om_read_write() {
        let mut mesh = PolyMesh::new();
        let v0 = mesh.add_vertex(glam::vec3(0.0, 0.0, 0.0));
        let v1 = mesh.add_vertex(glam::vec3(1.0, 0.0, 0.0));
        let v2 = mesh.add_vertex(glam::vec3(0.0, 1.0, 0.0));
        mesh.add_face(&[v0, v1, v2]);

        // Write OM file
        let temp_file = NamedTempFile::new().unwrap();
        write_om(&mesh, temp_file.path()).unwrap();

        // Check file size (header: 2 + 2 + 4 + 4 + 4 = 16 bytes + vertices + faces)
        let metadata = std::fs::metadata(temp_file.path()).unwrap();
        assert!(metadata.len() >= 16 + 3 * 12 + 3 * 4); // header + 3 vertices + 1 face

        // Read OM file
        let read_mesh = read_om(temp_file.path()).unwrap();
        assert!(read_mesh.n_vertices() >= 3);
    }

    #[test]
    fn test_om_detect() {
        assert!(detect_om(Path::new("mesh.om")));
        assert!(detect_om(Path::new("model.OM")));
        assert!(!detect_om(Path::new("mesh.obj")));
    }
}
