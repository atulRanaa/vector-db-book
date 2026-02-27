---
title: "Chapter 13 — Building an HNSW Store in Rust"
description: Memory-mapped graph storage, unsafe optimizations, SIMD distance kernels, and concurrent access in Rust.
tags:
  - implementation
  - rust
  - hnsw
---

# 13. Building an HNSW Store in Rust

This chapter walks through the design of a **production-quality HNSW index in Rust**, focusing on memory layout, safety trade-offs, and performance engineering.

---

## 13.1 Why Rust for Vector Indexes?

| Feature | C++ | Rust | Go/Java |
|---------|-----|------|---------|
| Memory safety | Manual | Compile-time guaranteed | GC |
| Performance | Fastest | ~Same as C++ | 2–5× slower |
| Concurrency | Threads + locks | Ownership + `Send`/`Sync` | Goroutines |
| GC pauses | None | None | **Unpredictable** latency spikes |
| Ecosystem | Mature | Growing rapidly | Large |

!!! tip "GC is the enemy of p99 latency"
    A 50ms GC pause in a Go/Java vector database can violate SLAs. Rust and C++ avoid this entirely.

---

## 13.2 Memory Layout Design

### Flat Vector Storage

Store all vectors in a **contiguous, memory-mapped `Vec<f32>`**:

```rust
// --8<-- [start:vector_storage]
/// Contiguous vector storage — cache-friendly, mmap-able.
pub struct VectorStorage {
    data: memmap2::MmapMut,  // memory-mapped file
    dim: usize,
    count: usize,
}

impl VectorStorage {
    /// Get vector by ID — zero-copy from mmap.
    #[inline]
    pub fn get(&self, id: usize) -> &[f32] {
        let offset = id * self.dim * std::mem::size_of::<f32>();
        let bytes = &self.data[offset..offset + self.dim * 4];
        // SAFETY: data is aligned and initialized during build
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, self.dim) }
    }
}
// --8<-- [end:vector_storage]
```

### Graph Adjacency Storage

Store neighbor lists in a **flat array** with fixed-width entries (no heap allocation per node):

```rust
// --8<-- [start:graph_layer]
/// Fixed-width adjacency list for one layer.
/// Layout: [count, n0, n1, ..., n_{M-1}] per node, each u32.
pub struct GraphLayer {
    data: Vec<u32>,
    max_neighbors: usize,
    entry_width: usize,  // = 1 + max_neighbors
}

impl GraphLayer {
    pub fn neighbors(&self, node: u32) -> &[u32] {
        let offset = node as usize * self.entry_width;
        let count = self.data[offset] as usize;
        &self.data[offset + 1..offset + 1 + count]
    }

    pub fn add_neighbor(&mut self, node: u32, neighbor: u32) {
        let offset = node as usize * self.entry_width;
        let count = self.data[offset] as usize;
        if count < self.max_neighbors {
            self.data[offset + 1 + count] = neighbor;
            self.data[offset] += 1;
        }
    }
}
// --8<-- [end:graph_layer]
```

This layout has **zero heap fragmentation** and is trivially serializable to disk.

---

## 13.3 SIMD Distance Kernels in Rust

Use the `std::arch` intrinsics for platform-specific SIMD:

```rust
// --8<-- [start:simd_l2]
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// AVX2-accelerated L2 squared distance.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn l2_squared_avx2(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = _mm256_setzero_ps();
    let chunks = a.len() / 8;
    for i in 0..chunks {
        let va = _mm256_loadu_ps(a.as_ptr().add(i * 8));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
        let diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }
    // Horizontal sum
    let hi = _mm256_extractf128_ps(sum, 1);
    let lo = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(lo, hi);
    let sum128 = _mm_hadd_ps(sum128, sum128);
    let sum128 = _mm_hadd_ps(sum128, sum128);
    let mut result = _mm_cvtss_f32(sum128);
    // Remainder
    for i in (chunks * 8)..a.len() {
        let d = a[i] - b[i];
        result += d * d;
    }
    result
}
// --8<-- [end:simd_l2]
```

---

## 13.4 Concurrent Search with `RwLock`

Multiple readers can search simultaneously; writers take an exclusive lock:

```rust
use std::sync::RwLock;

pub struct ConcurrentHNSW {
    graph: RwLock<Vec<GraphLayer>>,
    vectors: VectorStorage,
}

impl ConcurrentHNSW {
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(f32, u32)> {
        let graph = self.graph.read().unwrap();  // shared lock
        // ... beam search using graph ...
        todo!()
    }

    pub fn insert(&self, vector: &[f32]) -> u32 {
        let mut graph = self.graph.write().unwrap();  // exclusive lock
        // ... insert into graph ...
        todo!()
    }
}
```

!!! warning "Write starvation"
    Under heavy read load, writers may starve. Use a `parking_lot::RwLock` which provides writer-preferring fairness.

---

## 13.5 Persistence and Recovery

### On-Disk Format

```
index.hnsw
├── header (8 bytes: magic, version)
├── metadata (dim, M, ef, entry_point, max_layer)
├── vectors.bin (n × dim × 4 bytes, mmap'd)
├── graph_layer_0.bin (n × entry_width × 4 bytes)
├── graph_layer_1.bin (...)
└── ...
```

### WAL for Crash Recovery

1. Append `(INSERT, id, vector)` to WAL
2. Insert into in-memory graph
3. On flush, write graph layers to disk
4. On startup, replay WAL entries after last checkpoint

---

## 13.6 Optimization Checklist

| Optimization | Impact | Complexity |
|-------------|--------|-----------|
| SIMD distance kernels | 5–8× throughput | Medium |
| Memory-mapped vectors | Zero-copy, OS-managed paging | Low |
| Flat adjacency arrays | Cache-friendly, no allocator overhead | Medium |
| `RwLock` for concurrency | Linear read scaling | Low |
| Prefetch neighbor vectors | 10–30% latency reduction | Medium |
| PQ re-ranking | 10–50× memory reduction | High |

---

## References

1. Malkov, Y. A., & Yashunin, D. A. (2020). *HNSW*. IEEE TPAMI.
2. Qdrant source code (Rust HNSW implementation). https://github.com/qdrant/qdrant
