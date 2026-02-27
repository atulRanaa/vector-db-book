<div align="center">
  <img src="https://raw.githubusercontent.com/atulRanaa/vector-db-book/main/docs/assets/logo.png" alt="Vector DB Book Logo" width="120" style="border-radius: 20%; margin-bottom: 20px;" onerror="this.style.display='none'">
  
  <h1>Vector Databases</h1>
  <h3>Theory, Engineering, and Frontiers</h3>

  <p>
    <em>A comprehensive, open-source technical book demystifying the internals of vector search at the billion-scale.</em>
  </p>

  <p>
    <a href="https://atulRanaa.github.io/vector-db-book/"><img src="https://img.shields.io/badge/Read_Online-Website-blue?style=for-the-badge&logo=bookstack&logoColor=white" alt="Read Online"></a>
    <a href="https://github.com/atulRanaa/vector-db-book/actions/workflows/deploy.yml"><img src="https://img.shields.io/github/actions/workflow/status/atulRanaa/vector-db-book/deploy.yml?branch=main&style=for-the-badge&logo=githubactions" alt="Build Status"></a>
    <a href="https://creativecommons.org/licenses/by-sa/4.0/"><img src="https://img.shields.io/badge/License-CC_BY--SA_4.0-lightgrey?style=for-the-badge" alt="License: CC BY-SA 4.0"></a>
  </p>
</div>

---

## 🎯 Our Mission

As AI applications scale rapidly through Retrieval-Augmented Generation (RAG) and Agentic patterns, vector search has become the fundamental memory backbone of modern infrastructure. However, the internal mechanics of these databases — like HNSW tuning, Product Quantization, and distributed sharding — are often treated as black boxes.

This book exists to demystify vector search. It is written for software engineers, data scientists, and systems architects who want to transcend API calling and actually understand **how their data is stored, compressed, distributed, and searched** at scale.

## 🛠️ Hands-on "Build Your Own DB" Track

Unlike purely theoretical texts, this book emphasizes engineering implementation. Woven throughout the chapters is a progressive C++ assignment track where you build a toy vector database from scratch:

- **Ch 0:** Brute-force `L2` distance mathematics
- **Ch 2:** K-Means Clustering and IVF indexing
- **Ch 3:** Scalar Quantization (`float32` → `uint8` compression)
- **Ch 6:** Write-Ahead Logs (WAL) for durability
- **Ch 7:** Segmented File Store & immutable flushing
- **Ch 8:** Distributed Scatter-Gather search across Shards
- **Ch 9:** Tombstone soft-deletions via bitmaps
- **Ch 10:** Hybrid Search (Pre- vs. Post-filtering)
- **Ch 16:** Shard Split Simulation & hash routing
- **Ch 17:** Recall-vs-QPS Benchmark Harness

*All source code for the algorithms and exercises can be found in the [`src/`](src/) directory.*

---

## 📖 Chapter Breakdown

### Part I — Foundations
| Chapter | Topics Covered |
|---------|--------|
| **1. High-Dimensional Geometry** | Vector spaces, norms, cosine similarity, curse of dimensionality, PCA, UMAP |
| **2. ANN Algorithms** | KD-Trees, LSH, HNSW, NSG, DiskANN, Product Quantization, Annoy, ScaNN |
| **3. Index-Storage Trade-offs** | Memory hierarchies, quantization vs. recall curves, index sizing |
| **4. Query Semantics** | k-NN, range search, hybrid predicates, multi-vector (ColBERT) |
| **5. Data Ingestion** | Transformers, CLIP, multimodal embeddings, batch/streaming pipelines |

### Part II — System Architecture
| Chapter | Topics Covered |
|---------|--------|
| **6. Core Components** | WAL, index builder, query engine, scheduler |
| **7. Storage Engines** | LSM trees, columnar layouts, delta-merge |
| **8. Distributed Stores** | Sharding, replication, architecture comparison |
| **9. Real-Time Updates** | Dynamic HNSW, graph-merge, segment architecture |
| **10. Hybrid Search** | BM25+ANN rerank, RRF, Roaring Bitmaps, Percolation |
| **11. Hardware Acceleration** | SIMD intrinsics, GPU architectures, FPGA, NUMA, RDMA |
| **12. Observability** | Latency histograms, auto-tuning, capacity planning |

### Part III — Implementation Deep-Dive
| Chapter | Topics Covered |
|---------|--------|
| **13. HNSW Store in Rust** | Memory-mapped graphs, unsafe optimizations |
| **14. PQ-IVF on GPUs** | CUDA kernels, fused distance computation |
| **15. Transactional Schemes** | MVCC for vectors, compaction scheduling |
| **16. Elastic Scaling** | Shard split/merge, zero-downtime migration |
| **17. Benchmark Harness** | ANN-Benchmarks methodology, SIFT1M, GloVe |

### Part IV — Advanced Topics
| Chapter | Topics Covered |
|---------|--------|
| **18. Privacy** | Homomorphic encryption, MPC, differential privacy, TEEs |
| **19. Continual Learning** | Embedding updates, backward compatibility, drift |
| **20. Gen-AI Agents** | RAG pipelines, agentic memory, HyDE, evaluation frameworks |
| **21. Future Directions** | Learned indexes, Apache Iceberg, ADBC, neuromorphic hardware |
| **22. Open Source DBs** | Deep dive into LanceDB, Vespa, Vald, USearch, pgvector |
| **23. Case Studies** | Spotify, Pinterest, OpenAI infrastructure patterns |
| **24. In-Database ML** | PostgresML, Featureform, Virtual Feature Stores |

---

## 🚀 Getting Started Locally

### Prerequisites
- Python 3.9+
- Git

### Local Development Setup

```bash
# 1. Clone the repository
git clone https://github.com/atulRanaa/vector-db-book.git
cd vector-db-book

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # Or `venv\Scripts\activate` on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Serve the book locally (with live reload)
mkdocs serve
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser to view the live rendering of the book.

---

## 🤝 Contributing

This book is fundamentally an **open-source community project**. The technology surrounding Vector Databases evolves incredibly fast, and we rely on practitioners to help keep the content accurate and cutting-edge.

Contributions of all sizes are welcome and deeply appreciated:
- Fixing typos or broken links
- Improving unclear explanations (ELI5 analogies are highly valued!)
- Adding or refining C++/Rust/Python code examples
- Writing new sections on emerging research

### How to Contribute

1. **Fork** the repository on GitHub.
2. **Clone** your fork locally and create a feature branch (`git checkout -b feature/explain-binary-quantization`).
3. **Edit** the Markdown files in the `docs/` directory.
4. **Test** locally by running `mkdocs serve`.
5. **Commit & Push** your changes.
6. **Open a Pull Request** against the `main` branch.

All contributors will be recognized in the repository and future releases.

---

## 📝 License

We believe fundamental knowledge should be freely accessible.

- The **Text Content** (`docs/`, markdown diagrams, explanations) is licensed under the [Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).
- The **Code Examples** (`src/`, scripts, test cases) are licensed under the permissive [MIT License](LICENSE), allowing you to freely integrate the algorithms into your own projects.
