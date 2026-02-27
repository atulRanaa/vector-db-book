---
title: "Part I — Foundations"
description: Mathematical and algorithmic foundations of vector databases.
---

# Part I — Foundations

This section builds the theoretical and algorithmic bedrock needed to understand vector databases. We start with the geometry of high-dimensional spaces, work through the major families of approximate nearest neighbor algorithms, and finish with the practical engineering of data ingestion pipelines.

## Chapters

| # | Chapter | Key Topics |
|---|---------|------------|
| 1 | [High-Dimensional Geometry](high-dimensional-geometry.md) | Vector spaces, norms, cosine similarity, curse of dimensionality, dimensionality reduction |
| 2 | [ANN Algorithms](ann-algorithms.md) | KD-Trees, LSH, HNSW, NSG, Product Quantization, Annoy, ScaNN, DiskANN |
| 3 | [Index-Storage Trade-offs](index-storage-tradeoffs.md) | Memory hierarchies, cache-oblivious layouts, quantization vs. recall |
| 4 | [Query Semantics & Similarities](query-semantics.md) | k-NN vs. range search, hybrid predicates, multi-vector queries |
| 5 | [Data Ingestion & Vectorization](data-ingestion.md) | Transformers, word2vec, multimodal embeddings, batch vs. streaming |

!!! info "Prerequisites"
    Familiarity with linear algebra (vectors, matrices, inner products) and basic algorithm analysis (Big-O) is assumed. Appendix A provides a concise math refresher.
