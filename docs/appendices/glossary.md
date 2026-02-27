---
title: "Appendix B — Glossary"
description: Definitions of key terms used throughout the book.
tags:
  - appendix
  - glossary
---

# Appendix B — Glossary

| Term | Definition |
|------|-----------|
| **ADC** | Asymmetric Distance Computation — PQ search where the query is not quantized |
| **ANN** | Approximate Nearest Neighbor — finding near-optimal closest vectors in sub-linear time |
| **AVX2** | Advanced Vector Extensions 2 — Intel SIMD instruction set (256-bit registers) |
| **Beam search** | Graph traversal maintaining a priority queue of $ef$ best candidates |
| **BM25** | Best Matching 25 — probabilistic lexical ranking function (keyword search) |
| **CLIP** | Contrastive Language-Image Pretraining — model producing shared text/image embeddings |
| **Codebook** | Set of centroid vectors used in quantization (typically $K = 256$ entries) |
| **ColBERT** | Contextualized Late Interaction over BERT — multi-vector retrieval model |
| **Compaction** | Background process that merges small segments and removes tombstones |
| **Cosine distance** | $1 - \cos(\theta)$ — distance derived from cosine similarity |
| **Cosine similarity** | $\frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\|\|\mathbf{y}\|}$ — angular similarity between vectors |
| **Cross-encoder** | Model that scores (query, document) pairs jointly — more accurate than bi-encoder |
| **Curse of dimensionality** | Phenomenon where distances concentrate in high dimensions |
| **Dense embedding** | Fixed-length float vector from a neural encoder (e.g., 768-dim BERT) |
| **Differential privacy** | Privacy guarantee via calibrated noise addition |
| **DiskANN** | Microsoft's disk-resident graph-based ANN index (Vamana algorithm) |
| **ef_construction** | HNSW parameter: beam width during index building |
| **ef_search** | HNSW parameter: beam width during query |
| **Embedding** | Dense vector representation of data (text, image, etc.) |
| **FAISS** | Facebook AI Similarity Search — Meta's vector search library |
| **FMA** | Fused Multiply-Add — SIMD instruction: $a \leftarrow a + b \times c$ |
| **FP16** | 16-bit floating point |
| **FP32** | 32-bit floating point (standard) |
| **Ground truth** | Exact nearest neighbors computed by brute-force |
| **Hamming distance** | Number of bit positions where two binary vectors differ |
| **HBM** | High Bandwidth Memory — GPU memory technology |
| **HNSW** | Hierarchical Navigable Small World — multi-layer graph-based ANN algorithm |
| **Homomorphic encryption** | Encryption scheme allowing computation on ciphertext |
| **HyDE** | Hypothetical Document Embeddings — retrieve using LLM-generated answer |
| **INT8 (SQ8)** | 8-bit integer scalar quantization |
| **IVF** | Inverted File Index — partition-based ANN using Voronoi cells |
| **JL Lemma** | Johnson-Lindenstrauss — random projection preserves distances |
| **k-NN** | k-Nearest Neighbors — find $k$ closest vectors to a query |
| **KD-tree** | Space-partitioning tree using coordinate-aligned splits |
| **LSH** | Locality-Sensitive Hashing — hash-based ANN algorithm |
| **LSM tree** | Log-Structured Merge tree — write-optimized storage structure |
| **M** | HNSW parameter: maximum connections per node per layer |
| **Matryoshka embeddings** | Embeddings where any prefix of dimensions is a valid embedding |
| **MaxSim** | Maximum similarity — ColBERT scoring: per-token max cosine, then sum |
| **MIPS** | Maximum Inner Product Search |
| **mmap** | Memory-mapped file I/O — let OS manage file-to-memory paging |
| **MMD** | Maximum Mean Discrepancy — statistical test for distribution shift |
| **MTEB** | Massive Text Embedding Benchmark |
| **MVCC** | Multi-Version Concurrency Control — snapshot-based isolation |
| **nDCG** | Normalized Discounted Cumulative Gain — ranking quality metric |
| **nlist** | IVF parameter: number of Voronoi cells (clusters) |
| **nprobe** | IVF parameter: number of cells searched at query time |
| **NSG** | Navigating Spreading-out Graph — optimized proximity graph |
| **NUMA** | Non-Uniform Memory Access — multi-socket server memory topology |
| **PCA** | Principal Component Analysis — linear dimensionality reduction |
| **PQ** | Product Quantization — subspace decomposition for compression |
| **RAG** | Retrieval-Augmented Generation — LLM + vector search |
| **RDMA** | Remote Direct Memory Access — kernel-bypass network |
| **Recall@k** | Fraction of true $k$ nearest neighbors found by ANN |
| **RRF** | Reciprocal Rank Fusion — rank combination method |
| **Segment** | Self-contained, immutable index unit (sealed after reaching threshold) |
| **SIMD** | Single Instruction, Multiple Data — parallel CPU instructions |
| **Sparse vector** | Vector with mostly zero entries (e.g., TF-IDF, BM25) |
| **SQ4** | 4-bit scalar quantization |
| **t-SNE** | t-distributed Stochastic Neighbor Embedding — nonlinear visualization |
| **TEE** | Trusted Execution Environment — hardware-isolated enclave |
| **Tombstone** | Marker indicating a deleted vector (physical removal deferred) |
| **UMAP** | Uniform Manifold Approximation and Projection — nonlinear reduction |
| **Vamana** | Graph algorithm behind DiskANN |
| **VP-tree** | Vantage-Point tree — metric space partitioning |
| **WAL** | Write-Ahead Log — durability mechanism for writes |
