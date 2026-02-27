---
title: Home
description: "Vector Databases — Theory, Engineering, and Frontiers: A comprehensive technical book on vector database internals."
---

# Vector Databases — Theory, Engineering, and Frontiers

Welcome to the open-source technical book on **vector database internals**. This resource covers everything from the mathematical foundations of high-dimensional search to production-grade distributed system architectures.

---

## :material-book-open-variant: What You'll Learn

<div class="grid cards" markdown>

-   :material-math-compass:{ .lg .middle } **Part I — Foundations**

    ---

    High-dimensional geometry, ANN algorithms (HNSW, LSH, PQ), index–storage trade-offs, query semantics, and vectorization pipelines.

    [:octicons-arrow-right-24: Start reading](part-1-foundations/index.md)

-   :material-server-network:{ .lg .middle } **Part II — System Architecture**

    ---

    Core components, storage engines, distributed sharding, hybrid search, hardware acceleration (SIMD/GPU/FPGA), and observability.

    [:octicons-arrow-right-24: Start reading](part-2-architecture/index.md)

-   :material-code-braces:{ .lg .middle } **Part III — Implementation Deep-Dive**

    ---

    Build an HNSW store in Rust, GPU-accelerated PQ-IVF, transactional vector inserts, elastic scaling, and benchmark harnesses.

    [:octicons-arrow-right-24: Start reading](part-3-implementation/index.md)

-   :material-rocket-launch:{ .lg .middle } **Part IV — Advanced Topics & Research Frontiers**

    ---

    Privacy-preserving search, continual learning, Gen-AI agent pipelines, and future directions (learned indexing, vector SQL).

    [:octicons-arrow-right-24: Start reading](part-4-advanced/index.md)

</div>

---

## :material-target: Who This Book Is For

- **ML/AI Engineers** building retrieval-augmented generation (RAG) pipelines
- **Database Engineers** designing or evaluating vector storage systems
- **Systems Programmers** interested in high-performance search infrastructure
- **Researchers** exploring the frontier of approximate nearest neighbor search
- **Students** seeking a rigorous yet practical treatment of the field

## :material-hammer-wrench: How to Use This Book

Each chapter is self-contained — you can read linearly or jump to any topic of interest. Chapters include:

- **Mathematical notation** rendered with MathJax ($L_2$, cosine similarity, etc.)
- **Code examples** in Python, Rust, and C++ with copy-to-clipboard
- **Architecture diagrams** built with Mermaid
- **Admonitions** highlighting key insights, warnings, and practical tips

## :material-github: Contributing

This is an open-source project. Contributions are welcome! See the [GitHub repository](https://github.com/ranaatul/vector-db-book) for how to contribute.
