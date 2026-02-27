/**
 * Product Quantization (PQ) for vector compression and approximate search.
 *
 * Decomposes d-dimensional vectors into M subspaces and quantizes
 * each independently using k-means clustering.
 *
 * Compile: g++ -std=c++17 -O3 -o pq pq.hpp
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

// --8<-- [start:product_quantizer]
/**
 * Product Quantizer.
 *
 * Compresses d-dimensional float vectors into M bytes (one code per subspace).
 *
 * Compression ratio: d × 4 bytes → M bytes (e.g. 128-dim × 4B = 512B → 8B).
 *
 * Workflow:
 *   1. train()  — learn M codebooks of K=256 centroids each via k-means
 *   2. encode() — compress vectors to PQ codes (M uint8 per vector)
 *   3. search() — approximate distance using ADC lookup tables
 *
 * Asymmetric Distance Computation (ADC):
 *   dist(q, x̃)² ≈ Σ_m || q^(m) - c_{code_m}^(m) ||²
 *   Precompute a (M × K) distance table, then M lookups per candidate.
 */
class ProductQuantizer {
public:
  ProductQuantizer(size_t dim, size_t M = 8, size_t K = 256)
      : dim_(dim), M_(M), K_(K), ds_(dim / M) {
    assert(dim % M == 0 && "dim must be divisible by M");
    codebooks_.resize(
        M, std::vector<std::vector<float>>(K, std::vector<float>(ds_)));
  }

  /**
   * Train codebooks with k-means on each subspace.
   */
  void train(const std::vector<std::vector<float>> &data, size_t n_iter = 25) {
    std::mt19937 rng(42);
    size_t n = data.size();

    for (size_t m = 0; m < M_; ++m) {
      // Extract sub-vectors for subspace m
      std::vector<std::vector<float>> sub(n, std::vector<float>(ds_));
      for (size_t i = 0; i < n; ++i) {
        for (size_t d = 0; d < ds_; ++d) {
          sub[i][d] = data[i][m * ds_ + d];
        }
      }

      // Initialize centroids randomly
      std::vector<size_t> indices(n);
      std::iota(indices.begin(), indices.end(), 0);
      std::shuffle(indices.begin(), indices.end(), rng);
      for (size_t k = 0; k < K_; ++k) {
        codebooks_[m][k] = sub[indices[k % n]];
      }

      // k-means iterations
      std::vector<size_t> assignments(n);
      for (size_t iter = 0; iter < n_iter; ++iter) {
        // Assign
        for (size_t i = 0; i < n; ++i) {
          float best_dist = std::numeric_limits<float>::max();
          for (size_t k = 0; k < K_; ++k) {
            float d = l2_sq(sub[i], codebooks_[m][k]);
            if (d < best_dist) {
              best_dist = d;
              assignments[i] = k;
            }
          }
        }

        // Update centroids
        std::vector<std::vector<float>> sums(K_, std::vector<float>(ds_, 0));
        std::vector<size_t> counts(K_, 0);
        for (size_t i = 0; i < n; ++i) {
          size_t k = assignments[i];
          counts[k]++;
          for (size_t d = 0; d < ds_; ++d) {
            sums[k][d] += sub[i][d];
          }
        }
        for (size_t k = 0; k < K_; ++k) {
          if (counts[k] > 0) {
            for (size_t d = 0; d < ds_; ++d) {
              codebooks_[m][k][d] = sums[k][d] / counts[k];
            }
          }
        }
      }
    }
    trained_ = true;
  }

  /**
   * Encode vectors to PQ codes (M uint8 per vector).
   */
  std::vector<std::vector<uint8_t>>
  encode(const std::vector<std::vector<float>> &data) const {
    assert(trained_);
    size_t n = data.size();
    std::vector<std::vector<uint8_t>> codes(n, std::vector<uint8_t>(M_));

    for (size_t i = 0; i < n; ++i) {
      for (size_t m = 0; m < M_; ++m) {
        float best_dist = std::numeric_limits<float>::max();
        uint8_t best_k = 0;
        for (size_t k = 0; k < K_; ++k) {
          float d = 0;
          for (size_t dd = 0; dd < ds_; ++dd) {
            float diff = data[i][m * ds_ + dd] - codebooks_[m][k][dd];
            d += diff * diff;
          }
          if (d < best_dist) {
            best_dist = d;
            best_k = static_cast<uint8_t>(k);
          }
        }
        codes[i][m] = best_k;
      }
    }
    return codes;
  }

  /**
   * Decode PQ codes back to approximate vectors.
   */
  std::vector<float> decode(const std::vector<uint8_t> &code) const {
    assert(trained_);
    std::vector<float> vec(dim_);
    for (size_t m = 0; m < M_; ++m) {
      for (size_t d = 0; d < ds_; ++d) {
        vec[m * ds_ + d] = codebooks_[m][code[m]][d];
      }
    }
    return vec;
  }
  // --8<-- [end:product_quantizer]

  // --8<-- [start:adc_search]
  /**
   * Asymmetric Distance Computation (ADC) search.
   *
   * 1. Build distance table: dist_table[m][k] = ||q_m - c_mk||²
   * 2. For each database code, sum M table lookups
   * 3. Return top-k nearest
   *
   * Complexity: O(M·K·ds) to build table + O(n·M) to scan
   */
  struct SearchResult {
    float distance;
    size_t id;
    bool operator<(const SearchResult &o) const {
      return distance < o.distance;
    }
  };

  std::vector<SearchResult>
  search_adc(const std::vector<float> &query,
             const std::vector<std::vector<uint8_t>> &codes, size_t k) const {
    assert(trained_);
    size_t n = codes.size();

    // Precompute distance table: M × K
    std::vector<std::vector<float>> dist_table(M_, std::vector<float>(K_));
    for (size_t m = 0; m < M_; ++m) {
      for (size_t kk = 0; kk < K_; ++kk) {
        float d = 0;
        for (size_t dd = 0; dd < ds_; ++dd) {
          float diff = query[m * ds_ + dd] - codebooks_[m][kk][dd];
          d += diff * diff;
        }
        dist_table[m][kk] = d;
      }
    }

    // Compute approximate distances via table lookups
    std::vector<SearchResult> results(n);
    for (size_t i = 0; i < n; ++i) {
      float d = 0;
      for (size_t m = 0; m < M_; ++m) {
        d += dist_table[m][codes[i][m]];
      }
      results[i] = {std::sqrt(d), i};
    }

    // Partial sort for top-k
    k = std::min(k, n);
    std::partial_sort(results.begin(), results.begin() + k, results.end());
    results.resize(k);
    return results;
  }
  // --8<-- [end:adc_search]

private:
  static float l2_sq(const std::vector<float> &a, const std::vector<float> &b) {
    float s = 0;
    for (size_t i = 0; i < a.size(); ++i) {
      float d = a[i] - b[i];
      s += d * d;
    }
    return s;
  }

  size_t dim_, M_, K_, ds_;
  bool trained_ = false;
  // codebooks_[m][k][d] = centroid value
  std::vector<std::vector<std::vector<float>>> codebooks_;
};
