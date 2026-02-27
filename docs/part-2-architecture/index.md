---
title: "Part II — System Architecture"
description: How vector databases are built — from storage engines to distributed systems.
---

# Part II — System Architecture

This section takes you inside the architecture of production vector databases. You'll understand how the core components fit together, how data is stored and replicated, and how hybrid search combines vector similarity with traditional filtering.

## Chapters

| # | Chapter | Key Topics |
|---|---------|------------|
| 6 | [Core Components](core-components.md) | Ingest, index builder, query engine, storage, scheduler, WAL |
| 7 | [Storage Engines](storage-engines.md) | LSM trees, columnar layouts, delta-merge for mutable vectors |
| 8 | [Distributed Vector Stores](distributed-stores.md) | Sharding, consistency, replication, cloud-native DB comparison |
| 9 | [Real-Time Update Handling](realtime-updates.md) | Dynamic indexes, HNSW layer fan-out, graph-merge |
| 10 | [Hybrid Search](hybrid-search.md) | Score fusion, BM25 + ANN rerank, metadata filtering |
| 11 | [Hardware Acceleration](hardware-acceleration.md) | SIMD, GPU, FPGA, NUMA pinning, RDMA |
| 12 | [Observability & Operations](observability-operations.md) | Latency histograms, index health, auto-tuning |
