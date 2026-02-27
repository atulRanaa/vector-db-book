---
title: "Chapter 1 — High-Dimensional Geometry Refresher"
description: Vector spaces, norms, similarity measures, curse of dimensionality, and dimensionality reduction explained clearly.
tags:
  - foundations
  - geometry
  - dimensionality
---

# 1. High-Dimensional Geometry Refresher

Vector databases operate in **high-dimensional spaces** — typically ranging from 64 to 4,096 dimensions depending on the neural network generating the embeddings. For many developers, 3D space is intuitive, but visualising 1000-dimensional space breaks our standard mental models. 

This chapter builds geometric intuition for these high-dimensional spaces. We'll introduce the distance metrics that underpin similarity search, and explain a phenomenon called the "Curse of Dimensionality" which forces us to rethink how search algorithms work.

---

## 1.1 Vector Spaces and Representations

At its simplest, a **vector** $\mathbf{x} \in \mathbb{R}^d$ is just an ordered list of $d$ numbers. In the context of vector databases, these lists of numbers represent complex concepts like words, images, or entire documents.

There are two main types of vectors you will encounter:

- **Dense embeddings**: These are fixed-length lists of floating-point numbers produced by deep learning models (e.g., 768-dim from BERT, or 1536-dim from OpenAI's `text-embedding-3-small` model). Most of the values are non-zero.
- **Sparse vectors**: These are extremely high-dimensional lists (often >100,000 dimensions) where almost all the values are exactly zero. Examples include TF-IDF or BM25 term weights, where each dimension represents a specific word in an entire language vocabulary.

!!! note "ELI5: Why dense embeddings dominate"
    Imagine you are matching the sentence *"The king ruled the country"* with *"The monarch governed the nation"*. 
    
    If you use **sparse vectors** (which just check if exact words match), the overlap is basically zero because the English words used are completely different. 
    
    If you use **dense embeddings**, a neural network understands the *meaning*. It projects both sentences to the exact same neighbourhood in a mathematical space because they mean the same thing. Dense embeddings capture **semantic similarity**, not just lexical overlap.

### Vector as a Point in Space

If a 2D vector like `[3, 4]` represents a point on a flat sheet of paper, a 768-dimensional vector represents a point in a 768-dimensional universe. 

Nearest neighbor search in a vector database is fundamentally a geometric problem: *If my query is a specific point in this universe, how do I quickly find the dataset points that are physically closest to it?*

---

## 1.2 Norms and Distance Metrics

To figure out which points are "closest," we need a way to measure the distance between two vectors. There are several ways to do this, depending on what kind of data you have.

### The Minkowski Family

The **Minkowski distance** is a generalized mathematical formula that covers many standard ways of measuring distance. Its equation of order $p$ is:

$$
d_p(\mathbf{x}, \mathbf{y}) = \left( \sum_{i=1}^{d} |x_i - y_i|^p \right)^{1/p}
$$

By changing the value of $p$, we get different common distance metrics:

| Metric | $p$ | Formula | Real-World Analogy | Use case |
|--------|-----|---------|---------------------|----------|
| Manhattan ($L_1$) | 1 | $\sum \|x_i - y_i\|$ | Driving a taxi through city blocks. You can only move along a grid. | Sparse data, robust to outliers |
| **Euclidean ($L_2$)** | 2 | $\sqrt{\sum (x_i - y_i)^2}$ | Taking a helicopter straight from A to B (as the crow flies). | The most common default distance |
| Chebyshev ($L_\infty$) | $\infty$ | $\max_i \|x_i - y_i\|$ | The distance is just the single largest difference across any one dimension. | Worst-case difference analysis |

### Cosine Similarity

While Euclidean distance measures the physical distance between points, **Cosine Similarity** measures the **angle** between the two vectors, completely ignoring how "long" the vectors are.

$$
\text{cos}(\mathbf{x}, \mathbf{y}) = \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\| \, \|\mathbf{y}\|} = \frac{\sum_{i=1}^{d} x_i y_i}{\sqrt{\sum x_i^2} \cdot \sqrt{\sum y_i^2}}
$$

*   $\text{cos} = 1$: The vectors point in the exact same direction.
*   $\text{cos} = 0$: The vectors are exactly at a 90-degree angle (orthogonal/unrelated).
*   $\text{cos} = -1$: The vectors point in exactly opposite directions.

!!! tip "ELI5: Cosine Distance vs. Similarity"
    If you're writing a book review, and Alice writes a 1-paragraph positive review while Bob writes a 10-page positive review, the *Euclidean* distance between their vectors might be huge because Bob's vector is much longer (larger magnitude). 
    However, the *Cosine Similarity* will recognize they are both pointing in the "positive" direction and score them highly similar. 
    
    *Note: Databases usually compute **cosine distance** ($1 - \text{cos}$) because search algorithms inherently look for the "smallest" distance.*

### Inner Product (Dot Product)

$$
\text{IP}(\mathbf{x}, \mathbf{y}) = \sum_{i=1}^{d} x_i \, y_i
$$

The Inner Product multiplies matching dimensions and adds them up. It is widely used in **Maximum Inner Product Search (MIPS)**, which is crucial for recommendation systems where the vectors represent user profiles and item profiles, and the dot product represents the "score" or likelihood a user clicks an item.

### Hamming Distance

For binary vectors (lists that only contain 0s and 1s), we use the Hamming Distance:

$$
d_H(\mathbf{x}, \mathbf{y}) = \sum_{i=1}^{d} \mathbb{1}[x_i \neq y_i]
$$

This simply counts how many positions are different. It is incredibly fast for computers to calculate using low-level XOR CPU instructions.

### Implementation

Here is a C++ implementation of these core distance metrics. Notice how heavily optimized the AVX2 SIMD version is compared to the naive loop.

```{.cpp file=src/cpp/distances.cpp}
```

??? example "Key implementation: L2 distance (click to expand)"
    ```cpp
    --8<-- "src/cpp/distances.cpp:l2_naive"
    ```

    For SIMD-accelerated versions, see [Chapter 11 — Hardware Acceleration](../part-2-architecture/hardware-acceleration.md).

---

## 1.3 Curse of Dimensionality

### The Problem

As the number of dimensions $d$ grows, several deeply counter-intuitive mathematical phenomena start taking place. This is collectively known as the **Curse of Dimensionality**, and it fundamentally breaks standard database indexing techniques.

**1. Distance concentration**: As dimensions increase to the thousands, the mathematical difference between the "farthest" point in your dataset and the "nearest" point in your dataset vanishes.

$$
\lim_{d \to \infty} \frac{d_{\max} - d_{\min}}{d_{\min}} \to 0
$$

**ELI5**: Imagine searching a massive library for a book on "Cats". In a 3-dimensional system, the cat book is 1 foot away, and a book on cars is 100 feet away. Easy. But in 1000-dimensional space, the cat book is 10.0 feet away, and the car book is 10.1 feet away. Everything is basically the same distance apart, making it incredibly hard to confidently say which book is the "nearest".

**2. Volume of hyperspheres vanishes**: The volume of an N-dimensional sphere goes to **almost zero** as dimensions grow toward infinity. Almost all the volume of the space ends up pushed into a razor-thin mathematically dense shell near the surface. 

**3. Orthogonality**: If you pick two completely random vectors in high-dimensional space, they are almost guaranteed to be perfectly mathematically orthogonal (at a 90-degree angle to each other).

!!! warning "Why does this matter?"
    Traditional databases use "Tree" shapes (like B-Trees or KD-trees) to chop up space and quickly find records. Because high-dimensional space is mostly empty, and everything is roughly the exact same distance from everything else, **Tree algorithms completely collapse**. They end up having to scan every single point anyway (a brute-force $O(n \cdot d)$ scan). This is why we absolutely need specialized approximate algorithms (Chapter 2).

### Empirical Visualization

```mermaid
flowchart TD
    %% Define the progression of the curse
    subgraph Low Dim ["Low Dimensions (2D / 3D)"]
        direction LR
        L1(Data is spread out) --> L2(Clear nearest neighbors)
        L2 --> L3(KD-Trees work effectively)
    end

    subgraph High Dim ["High Dimensions (768D+)"]
        direction LR
        H1(Distances cluster near median val) --> H2(All points seem equidistant)
        H2 --> H3(Trees degrade to O/N.Brute Force)
    end
    
    Low Dim ==>|Dimensionality increases| High Dim
    
    High Dim --> Solution[Solution: Approximate Nearest Neighbor Algorithms]
    
    style Low Dim fill:#e1f5fe,stroke:#03a9f4,stroke-width:2px;
    style High Dim fill:#ffebee,stroke:#f44336,stroke-width:2px;
    style Solution fill:#e8f5e9,stroke:#4caf50,stroke-width:3px;
```

---

## 1.4 Dimensionality Reduction

If high dimensions break our math, can we just shrink the vectors back down to low dimensions before saving them? Yes, but it always comes with trade-offs.

### Principal Component Analysis (PCA)

PCA mathematically squeezes the data to preserve the directions holding the **maximum variance** (the most amount of distinct signal).

$$
\mathbf{z} = W_k^T (\mathbf{x} - \boldsymbol{\mu})
$$

| Pros | Cons |
|------|------|
| Highly optimal linear reduction | It only detects linear shapes, missing complex curved relationships |
| Fast and mathematically proven | Requires computing the covariance matrix of your entire dataset upfront |

### Johnson-Lindenstrauss Lemma

!!! info "Theorem (JL Lemma)"
    For any $\epsilon \in (0, 1)$ and any set of $n$ points in $\mathbb{R}^d$, there exists a linear map $f: \mathbb{R}^d \to \mathbb{R}^k$ with: $k = O\left(\frac{\log n}{\epsilon^2}\right)$ such that for all pairs of vectors, the distances between them in the low dimensional space are mathematically guaranteed to be almost identical to their distances in the high dimensional space.

**ELI5**: The JL Lemma proves a mathematical miracle — if you multiply your 1000-dimensional dataset by a completely random grid of numbers (a Gaussian matrix), the dataset will compress down to a few hundred dimensions, and the relative distance between *every single point* will stay essentially exactly the same. You don't even have to train a model to do this compression; random math does it for you.

### UMAP and t-SNE

These are non-linear methods used purely for **visualization**. If you want to view your 1000-dimensional embeddings as a 2D scatterplot graph on a computer screen, you use UMAP or t-SNE. 

!!! caution "Not for search"
    UMAP and t-SNE deliberately warp and distort absolute mathematical distances to make the 2D visual clusters look pretty to humans. You should **never** run vector searches over UMAP-reduced vectors.

---

## 1.5 Measure Concentration and Its Implications

Because of these mathematical quirks, our standard approaches to building indexes fail. The design of modern vector databases is a direct reaction to these phenomena:

| Phenomenon | Why it hurts | How Vector DBs fix it |
|-----------|------------|----------|
| **Distance concentration** | Finding the absolute true closest point requires checking everything | They accept finding an **approximate** closest point instead (Ch 2) |
| **Sphere volume goes to zero** | Grid-based and tree-based partitioning algorithms fail | They build **Navigable Proximity Graphs** (HNSW) instead |
| **Data has redundant dimensions** | Storing floats is expensive for RAM | They use **Product Quantization** (PQ) compression (Ch 3) |

---

## References

1. Aggarwal, C. C., Hinneburg, A., & Keim, D. A. (2001). *On the Surprising Behavior of Distance Metrics in High Dimensional Spaces*. ICDT.
2. Johnson, W. B., & Lindenstrauss, J. (1984). *Extensions of Lipschitz mappings into a Hilbert space*. Contemporary Mathematics, 26, 189–206.
3. Blum, A., Hopcroft, J., & Kannan, R. (2020). *Foundations of Data Science*. Cambridge University Press. Chapter 2: High-Dimensional Space.
4. Dasgupta, S., & Gupta, A. (2003). *An elementary proof of a theorem of Johnson and Lindenstrauss*. Random Structures & Algorithms.
