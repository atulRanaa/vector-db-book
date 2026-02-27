---
title: "API Reference (C++ and Rust)"
description: Core algorithm implementations spanning distances, locality-sensitive hashing, and HNSW graphs.
tags:
  - api
  - reference
  - cpp
  - rust
---

# API Reference (Algorithms)

This reference documents the core algorithmic implementations that back the theoretical explanations throughout the book. All implementations are thoroughly commented specifically for educational reading.

---

## 1. C++ Distance Metrics

The backbone of any vector database is mathematical distance calculation. For production scale, raw scalar `for` loops are too slow, and AVX2/AVX-512 SIMD (Single Instruction, Multiple Data) intrinsically accelerated kernels are required.

```cpp title="src/cpp/distances.cpp"
--8<-- "src/cpp/distances.cpp"
```

---

## 2. C++ Locality-Sensitive Hashing (LSH)

Locality-Sensitive Hashing uses random hyperplane projection to build probabilistic buckets. Similar vectors have a mathematically high probability of landing in the same bucket via identical binary signature patterns.

```cpp title="src/cpp/lsh.hpp"
--8<-- "src/cpp/lsh.hpp"
```

---

## 3. C++ Product Quantization (PQ)

Product Quantization aggressively chunks high-dimensional vectors into $M$ sub-spaces, and trains a `k-means` dictionary codebook for each chunk. Memory footprint shrinks up to 96x.

```cpp title="src/cpp/pq.hpp"
--8<-- "src/cpp/pq.hpp"
```

---

## 4. C++ HNSW Graph

Hierarchical Navigable Small World graphs are multi-layer proximity graphs. Search jumps layer by layer (skipping massive distances like an interstate highway) until dropping to Layer 0 to execute dense local navigation.

```cpp title="src/cpp/hnsw.hpp"
--8<-- "src/cpp/hnsw.hpp"
```

---

## 5. C++ Inverted File Index (IVF)

IVF uses k-means to slice the entire valid embedding space into distinct "Voronoi Cells". During an incoming query, the database routes the query to the `nprobe` nearest cell centroids, entirely skipping computation in faraway regions.

```cpp title="src/cpp/ivf.hpp"
--8<-- "src/cpp/ivf.hpp"
```
