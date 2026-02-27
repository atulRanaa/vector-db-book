---
title: "Appendix C — Bibliography"
description: Annotated bibliography of foundational papers in vector search and related fields.
tags:
  - appendix
  - bibliography
---

# Appendix C — Annotated Bibliography

---

## Foundational Algorithms

1. **Indyk, P., & Motwani, R. (1998).** *Approximate Nearest Neighbors: Towards Removing the Curse of Dimensionality.* STOC.
    — Introduced the theoretical framework for LSH and $(1+\epsilon)$-approximate NN.

2. **Malkov, Y. A., & Yashunin, D. A. (2020).** *Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs.* IEEE TPAMI.
    — The HNSW paper. Basis for most production vector database indexes.

3. **Jégou, H., Douze, M., & Schmid, C. (2011).** *Product Quantization for Nearest Neighbor Search.* IEEE TPAMI.
    — Introduced PQ and ADC. Foundation of IVF-PQ used in FAISS.

4. **Johnson, J., Douze, M., & Jégou, H. (2019).** *Billion-scale Similarity Search with GPUs.* IEEE TBD.
    — FAISS paper. GPU-accelerated IVF-PQ and flat search.

5. **Subramanya, S. J., et al. (2019).** *DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node.* NeurIPS.
    — SSD-resident graph search. Single-machine billion-scale.

6. **Guo, R., et al. (2020).** *Accelerating Large-Scale Inference with Anisotropic Vector Quantization.* ICML.
    — ScaNN: learned quantization that optimizes ranking, not just reconstruction.

---

## Geometry and Theory

7. **Aggarwal, C. C., Hinneburg, A., & Keim, D. A. (2001).** *On the Surprising Behavior of Distance Metrics in High Dimensional Spaces.* ICDT.
    — Empirical study of distance concentration.

8. **Johnson, W. B., & Lindenstrauss, J. (1984).** *Extensions of Lipschitz Mappings into a Hilbert Space.* Contemporary Mathematics.
    — The JL lemma: random projections preserve distances.

9. **Blum, A., Hopcroft, J., & Kannan, R. (2020).** *Foundations of Data Science.* Cambridge University Press.
    — Textbook chapter on high-dimensional geometry.

---

## Embeddings and NLP

10. **Reimers, N., & Gurevych, I. (2019).** *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP.
    — Made BERT practical for semantic similarity search.

11. **Radford, A., et al. (2021).** *Learning Transferable Visual Models From Natural Language Supervision.* ICML.
    — CLIP: shared text-image embedding space.

12. **Muennighoff, N., et al. (2023).** *MTEB: Massive Text Embedding Benchmark.* EACL.
    — Standard benchmark for comparing embedding models.

13. **Kusupati, A., et al. (2022).** *Matryoshka Representation Learning.* NeurIPS.
    — Variable-dimension embeddings: any prefix is a valid embedding.

---

## RAG and Applications

14. **Lewis, P., et al. (2020).** *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS.
    — Introduced RAG: combining retrieval with generation.

15. **Gao, L., et al. (2023).** *Precise Zero-Shot Dense Retrieval without Relevance Labels.* ACL.
    — HyDE: generate hypothetical answers for better retrieval.

16. **Khattab, O., & Zaharia, M. (2020).** *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT.* SIGIR.
    — Multi-vector retrieval with MaxSim scoring.

---

## Systems

17. **Wang, J., et al. (2021).** *Milvus: A Purpose-Built Vector Data Management System.* SIGMOD.
    — Architecture of a production vector database.

18. **Wei, J., et al. (2023).** *Filtered-DiskANN: Graph Algorithms for Approximate Nearest Neighbor Search with Filters.* WWW.
    — Combining metadata filtering with graph-based ANN.

---

## Privacy and Security

19. **Morris, J. X., et al. (2023).** *Text Embeddings Reveal (Almost) As Much As Text.* EMNLP.
    — Demonstrates embedding inversion attacks.

20. **Dwork, C. (2006).** *Differential Privacy.* ICALP.
    — Foundational framework for privacy-preserving data analysis.

---

## Benchmarks

21. **Aumüller, M., Bernhardsson, E., & Faithfull, A. (2020).** *ANN-Benchmarks: A Benchmarking Tool for Approximate Nearest Neighbor Algorithms.* Information Systems.
    — Standard evaluation framework for ANN algorithms.

22. **Simhadri, H., et al. (2022).** *Results of the NeurIPS'21 Challenge on Billion-Scale ANN Search.* NeurIPS.
    — Big-ANN-Benchmarks: billion-scale evaluation.
