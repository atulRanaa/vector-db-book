---
title: "Chapter 12 — Observability & Operations"
description: Latency histograms, recall regression, capacity planning, auto-tuning, and production runbooks.
tags:
  - architecture
  - observability
  - operations
---

# 12. Observability & Operations

Running a vector database in production requires monitoring, alerting, and capacity planning tools tailored to ANN workloads.

---

## 12.1 Key Metrics

### The Four Golden Signals for Vector Search

| Signal | Metric | Alert Threshold |
|--------|--------|----------------|
| **Latency** | p50, p95, p99 query latency | p99 > 100ms |
| **Traffic** | Queries/sec, inserts/sec | Spike > 3× baseline |
| **Errors** | Failed queries, timeouts | Error rate > 0.1% |
| **Saturation** | Memory usage, CPU, disk I/O | Memory > 85% |

### Vector-Specific Metrics

| Metric | Description | Why It Matters |
|--------|------------|---------------|
| **Recall@k** | Fraction of true k-NN found | Quality regression detection |
| **Distance ratio** | $d_{\text{approx}} / d_{\text{exact}}$ | ANN quality per-query |
| **Candidates scanned** | Vectors evaluated per query | Early termination indicator |
| **Segment count** | Number of active segments | Compaction needed? |
| **Tombstone ratio** | Deleted / total vectors | Index degradation |
| **Write amplification** | Bytes written / bytes inserted | Storage efficiency |

---

## 12.2 Recall Monitoring

### Synthetic Ground-Truth Testing

Periodically run exact brute-force search on a sample and compare with ANN results:

$$
\text{Online recall} = \frac{1}{|Q|} \sum_{q \in Q} \frac{|\text{ANN}(q, k) \cap \text{BF}(q, k)|}{k}
$$

!!! warning "Recall can silently degrade"
    As data distribution shifts (new embedding model, different content), index quality may drop without any error signals. Automated recall monitoring is essential.

---

## 12.3 Capacity Planning

### Memory Estimation

$$
\text{Required RAM} = n \cdot \left( d \cdot b + M_{\text{avg}} \cdot 8 + \text{meta}_{\text{avg}} \right) \cdot (1 + \text{overhead})
$$

where:
- $n$ = number of vectors
- $b$ = bytes per dimension (4 for FP32, 1 for INT8)
- $M_{\text{avg}}$ = average edges per node
- $\text{overhead}$ ≈ 10–20%

### QPS Estimation

$$
\text{QPS}_{\text{max}} = \frac{\text{threads} \times \text{IPC}}{\text{dist\_computations\_per\_query} \times \text{cycles\_per\_dist}}
$$

Rule of thumb: 1 core ≈ 500–2000 QPS for 1M vectors at 768 dimensions (HNSW, ef_search=50).

---

## 12.4 Auto-Tuning

### Adaptive ef_search

Monitor recall and latency, automatically adjust $\texttt{ef\_search}$:

```
if recall < target:
    ef_search += 10
elif p99_latency > budget:
    ef_search -= 5
```

### Adaptive nprobe (IVF)

Similar for IVF indexes: increase $n_{\text{probe}}$ when recall drops, decrease when latency is too high.

---

## 12.5 Production Operations

### Common Runbook Items

| Scenario | Action |
|----------|--------|
| p99 latency spike | Check segment count, compaction status, memory pressure |
| Recall dropped | Check data distribution shift, run synthetic recall test |
| Memory growing | Check tombstone ratio, trigger compaction |
| Cold start slow | Pre-warm critical segments, increase boot parallelism |
| Disk space full | Compact, archive old segments, promote SQ8 |

---

## References

1. Google SRE Book. *Monitoring Distributed Systems*. https://sre.google/sre-book/monitoring-distributed-systems/
