---
title: "Part III — Implementation Deep-Dive"
description: Hands-on implementation guides — Rust, CUDA, transactions, and benchmarks.
---

# Part III — Implementation Deep-Dive

Roll up your sleeves. This section provides hands-on implementation walkthroughs of real vector database components, from a complete HNSW store in Rust to GPU-accelerated quantized search.

## Chapters

| # | Chapter | Key Topics |
|---|---------|------------|
| 13 | [HNSW Store in Rust](hnsw-rust.md) | Memory-mapped adjacency lists, unsafe optimizations |
| 14 | [PQ-IVF on GPUs](pq-ivf-gpu.md) | CUDA kernels, fused distance computation |
| 15 | [Transactional Schemes](transactional-schemes.md) | MVCC adaptations, compaction scheduling |
| 16 | [Elastic Scaling](elastic-scaling.md) | Split/merge rebalancing, hotspot mitigation |
| 17 | [Benchmark Harness](benchmark-harness.md) | ANN-Benchmarks, synthetic data, latency suites |

!!! warning "Prerequisites"
    Chapters 13–14 assume familiarity with Rust and CUDA respectively. Chapter 15 assumes database transaction fundamentals.
