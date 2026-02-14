# RustMesh

Rust port of OpenMesh - A versatile geometric data structure for representing and manipulating polygonal meshes.

## Overview

RustMesh provides a native Rust implementation of mesh data structures and operations, inspired by OpenMesh. It supports:
- Vertex, edge, halfedge, and face representations
- Polygonal mesh types
- Efficient iteration and circulation
- File I/O for common mesh formats (OFF, OBJ)

## Example

```rust
use rustmesh::{PolyMesh, Vec3};

let mut mesh = PolyMesh::new();

// Add vertices
let v0 = mesh.add_vertex(Vec3::new(0.0, 0.0, 0.0));
let v1 = mesh.add_vertex(Vec3::new(1.0, 0.0, 0.0));
let v2 = mesh.add_vertex(Vec3::new(0.0, 1.0, 0.0));

// Add a face
mesh.add_face(&[v0, v1, v2]);

// Iterate over vertices
for v in mesh.vertices() {
    let point = mesh.point(v);
    println!("Vertex at {:?}", point);
}
```

## Features Implemented

### Core Data Structures
- ✅ Handles (VertexHandle, HalfedgeHandle, EdgeHandle, FaceHandle)
- ✅ Items (Vertex, Halfedge, Edge, Face)
- ✅ ArrayKernel (storage layer)
- ✅ PolyConnectivity (connectivity relations)
- ✅ Geometry (geometric operations: bounding box, triangle area/normal)

### File I/O
- ✅ OFF file reading/writing
- ✅ OBJ file reading/writing
- ✅ Format auto-detection

### Circulators
- ✅ Vertex iterator
- ✅ Edge iterator
- ✅ Face iterator
- ✅ Halfedge iterator
- ✅ Vertex-vertex circulator (1-ring)
- ✅ Vertex-face circulator

## Features Pending (vs OpenMesh)

### High Priority
- [ ] **OFF file writing** - Complete face output
- [ ] **OBJ file writing** - Complete face output
- [ ] **PLY file format** - Polygon/Lexile format support
- [ ] **STL file format** - Stereolithography format support
- [ ] **face_vertices()** - Get vertices of a face

### Medium Priority
- [ ] **TriConnectivity** - Triangle mesh specialization
- [ ] **AttribKernel** - Attribute management (normals, colors, texcoords)
- [ ] **More circulators** - Edge-face, face-vertex, etc.
- [ ] **SmartHandles** - Automatic connectivity updates

### Lower Priority
- [ ] **Binary formats** - Binary OFF/PLY/STL support
- [ ] **Compression** - Compressed mesh formats
- [ ] **Performance optimization** - Memory pools, SIMD

## File Format Support

| Format | Read | Write | Notes |
|--------|------|-------|-------|
| OFF    | ✅   | ⚠️    | Basic support, color support incomplete |
| OBJ    | ✅   | ⚠️    | Basic support, UV/normals incomplete |
| PLY    | ❌   | ❌    | Not implemented |
| STL    | ❌   | ❌    | Not implemented |

## Project Structure

```
src/
├── lib.rs          # Main module exports
├── handles.rs      # Handle types
├── items.rs        # Mesh item types (Vertex, Edge, Halfedge, Face)
├── kernel.rs       # ArrayKernel storage layer
├── connectivity.rs # PolyConnectivity implementation
├── geometry.rs     # Geometric operations
└── io.rs           # File I/O (OFF, OBJ)
```

## Building

```bash
cargo build
cargo test
cargo bench
```

## Roadmap

See [ROADMAP.md](ROADMAP.md) for detailed development plans.

## References

- [OpenMesh Documentation](https://www.openmesh.org/)
- [OpenMesh GitHub](https://github.com/OpenMesh/Core)
