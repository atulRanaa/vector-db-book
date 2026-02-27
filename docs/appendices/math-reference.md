---
title: "Appendix A — Math Reference"
description: Quick reference sheets for linear algebra, probability, and information theory used throughout the book.
tags:
  - appendix
  - math
---

# Appendix A — Math Reference

---

## A.1 Linear Algebra

### Vectors and Matrices

| Notation | Meaning |
|----------|---------|
| $\mathbf{x} \in \mathbb{R}^d$ | Column vector of $d$ real numbers |
| $\|\mathbf{x}\|_p$ | $L_p$ norm: $\left(\sum_i |x_i|^p\right)^{1/p}$ |
| $\|\mathbf{x}\|_2$ | Euclidean norm: $\sqrt{\sum x_i^2}$ |
| $\mathbf{x} \cdot \mathbf{y}$ | Dot product: $\sum_i x_i y_i$ |
| $X \in \mathbb{R}^{n \times d}$ | Matrix: $n$ rows, $d$ columns |
| $X^T$ | Transpose |
| $X^{-1}$ | Inverse (if square and non-singular) |
| $\text{tr}(A)$ | Trace: $\sum_i a_{ii}$ |
| $\det(A)$ | Determinant |

### Eigendecomposition

For symmetric $A \in \mathbb{R}^{d \times d}$:

$$
A = Q \Lambda Q^T, \quad Q^T Q = I
$$

where $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_d)$ are eigenvalues and columns of $Q$ are eigenvectors.

### SVD (Singular Value Decomposition)

Any $X \in \mathbb{R}^{n \times d}$ can be decomposed:

$$
X = U \Sigma V^T
$$

where $U \in \mathbb{R}^{n \times n}$, $\Sigma \in \mathbb{R}^{n \times d}$ (diagonal), $V \in \mathbb{R}^{d \times d}$. PCA uses the top-$k$ right singular vectors (columns of $V$).

---

## A.2 Distance Metrics Summary

| Metric | Formula | Range | Properties |
|--------|---------|-------|-----------|
| Euclidean ($L_2$) | $\sqrt{\sum (x_i - y_i)^2}$ | $[0, \infty)$ | Metric |
| Manhattan ($L_1$) | $\sum |x_i - y_i|$ | $[0, \infty)$ | Metric |
| Chebyshev ($L_\infty$) | $\max_i |x_i - y_i|$ | $[0, \infty)$ | Metric |
| Cosine distance | $1 - \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\|\|\mathbf{y}\|}$ | $[0, 2]$ | Semimetric |
| Inner product | $\sum x_i y_i$ | $(-\infty, \infty)$ | Not a distance |
| Hamming | $\sum \mathbb{1}[x_i \neq y_i]$ | $[0, d]$ | Metric |
| Jaccard | $1 - \frac{|A \cap B|}{|A \cup B|}$ | $[0, 1]$ | Metric |

---

## A.3 Probability and Statistics

### Key Distributions

| Distribution | PDF/PMF | Mean | Variance |
|-------------|---------|------|----------|
| $\mathcal{N}(\mu, \sigma^2)$ | $\frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | $\mu$ | $\sigma^2$ |
| $\text{Uniform}(a, b)$ | $\frac{1}{b-a}$ | $\frac{a+b}{2}$ | $\frac{(b-a)^2}{12}$ |
| $\text{Geometric}(p)$ | $(1-p)^{k-1} p$ | $\frac{1}{p}$ | $\frac{1-p}{p^2}$ |

### Useful Concentration Inequalities

**Hoeffding's inequality**: For bounded independent random variables $X_i \in [a_i, b_i]$:

$$
Pr\left[\left|\frac{1}{n}\sum X_i - \mathbb{E}\left[\frac{1}{n}\sum X_i\right]\right| \geq t\right] \leq 2 \exp\left(-\frac{2n^2 t^2}{\sum (b_i - a_i)^2}\right)
$$

**Johnson-Lindenstrauss** (simplified): Random projection into $k = O(\epsilon^{-2} \log n)$ dimensions preserves pairwise distances within $(1 \pm \epsilon)$.

---

## A.4 Information Theory

| Measure | Formula | Use in Vector DBs |
|---------|---------|-------------------|
| Entropy | $H(X) = -\sum p(x) \log p(x)$ | Quantization codebook quality |
| KL Divergence | $D_{KL}(P\|Q) = \sum p(x) \log \frac{p(x)}{q(x)}$ | Distribution drift detection |
| Mutual Information | $I(X;Y) = H(X) - H(X|Y)$ | Feature selection for PQ subspaces |

---

## A.5 Complexity Notation

| Notation | Meaning | Example |
|----------|---------|---------|
| $O(f)$ | Upper bound (worst case) | Brute-force k-NN: $O(n \cdot d)$ |
| $\Omega(f)$ | Lower bound | Any comparison-based search: $\Omega(\log n)$ |
| $\Theta(f)$ | Tight bound | Sorted array search: $\Theta(\log n)$ |
| $\tilde{O}(f)$ | Ignoring log factors | HNSW query: $\tilde{O}(d \cdot M)$ |
