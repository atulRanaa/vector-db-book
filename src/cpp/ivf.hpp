/**
 * Inverted File Index (IVF) for approximate nearest neighbor search.
 *
 * Partitions vector space into Voronoi cells using k-means,
 * then searches only the nprobe nearest cells at query time.
 *
 * Compile: g++ -std=c++17 -O3 -o ivf ivf.hpp
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

// --8<-- [start:ivf_index]
/**
 * Inverted File Index.
 *
 * Search complexity: O(nprobe × n/nlist × d) vs O(n × d) brute-force.
 *
 * Parameters:
 *   nlist  — number of Voronoi cells (centroids)
 *   nprobe — number of cells to search (trade-off: recall vs speed)
 */
class IVFIndex {
public:
  struct SearchResult {
    float distance;
    size_t id;
    bool operator<(const SearchResult &o) const {
      return distance < o.distance;
    }
  };

  IVFIndex(size_t dim, size_t nlist = 100, size_t nprobe = 10)
      : dim_(dim), nlist_(nlist), nprobe_(nprobe) {
    inverted_lists_.resize(nlist);
  }

  /**
   * Train centroids using k-means.
   */
  void train(const std::vector<std::vector<float>> &data, size_t n_iter = 20) {
    size_t n = data.size();
    centroids_.resize(nlist_, std::vector<float>(dim_));

    std::mt19937 rng(42);
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);

    for (size_t c = 0; c < nlist_; ++c) {
      centroids_[c] = data[indices[c % n]];
    }

    std::vector<size_t> assignments(n);
    for (size_t iter = 0; iter < n_iter; ++iter) {
      // Assign to nearest centroid
      for (size_t i = 0; i < n; ++i) {
        float best = std::numeric_limits<float>::max();
        for (size_t c = 0; c < nlist_; ++c) {
          float d = l2_sq(data[i], centroids_[c]);
          if (d < best) {
            best = d;
            assignments[i] = c;
          }
        }
      }
      // Update centroids
      std::vector<std::vector<float>> sums(nlist_, std::vector<float>(dim_, 0));
      std::vector<size_t> counts(nlist_, 0);
      for (size_t i = 0; i < n; ++i) {
        counts[assignments[i]]++;
        for (size_t d = 0; d < dim_; ++d)
          sums[assignments[i]][d] += data[i][d];
      }
      for (size_t c = 0; c < nlist_; ++c) {
        if (counts[c] > 0) {
          for (size_t d = 0; d < dim_; ++d)
            centroids_[c][d] = sums[c][d] / counts[c];
        }
      }
    }
    trained_ = true;
  }

  /**
   * Add vectors to the inverted lists.
   */
  void add(const std::vector<std::vector<float>> &data) {
    assert(trained_);
    vectors_ = data;
    for (auto &list : inverted_lists_)
      list.clear();

    for (size_t i = 0; i < data.size(); ++i) {
      float best = std::numeric_limits<float>::max();
      size_t best_c = 0;
      for (size_t c = 0; c < nlist_; ++c) {
        float d = l2_sq(data[i], centroids_[c]);
        if (d < best) {
          best = d;
          best_c = c;
        }
      }
      inverted_lists_[best_c].push_back(i);
    }
  }

  /**
   * Search for k approximate nearest neighbors.
   *
   * 1. Find nprobe nearest centroids
   * 2. Scan vectors in those cells
   * 3. Return top-k
   */
  std::vector<SearchResult> search(const std::vector<float> &query,
                                   size_t k) const {
    assert(trained_);

    // Find nearest centroids
    std::vector<std::pair<float, size_t>> centroid_dists(nlist_);
    for (size_t c = 0; c < nlist_; ++c) {
      centroid_dists[c] = {l2_sq(query, centroids_[c]), c};
    }
    std::partial_sort(centroid_dists.begin(),
                      centroid_dists.begin() + std::min(nprobe_, nlist_),
                      centroid_dists.end());

    // Collect and score candidates
    std::vector<SearchResult> candidates;
    for (size_t p = 0; p < std::min(nprobe_, nlist_); ++p) {
      size_t cell = centroid_dists[p].second;
      for (size_t idx : inverted_lists_[cell]) {
        float d = std::sqrt(l2_sq(query, vectors_[idx]));
        candidates.push_back({d, idx});
      }
    }

    k = std::min(k, candidates.size());
    if (k > 0) {
      std::partial_sort(candidates.begin(), candidates.begin() + k,
                        candidates.end());
      candidates.resize(k);
    }
    return candidates;
  }

  void set_nprobe(size_t nprobe) { nprobe_ = nprobe; }
  size_t size() const { return vectors_.size(); }

private:
  static float l2_sq(const std::vector<float> &a, const std::vector<float> &b) {
    float s = 0;
    for (size_t i = 0; i < a.size(); ++i) {
      float d = a[i] - b[i];
      s += d * d;
    }
    return s;
  }

  size_t dim_, nlist_, nprobe_;
  bool trained_ = false;
  std::vector<std::vector<float>> centroids_;
  std::vector<std::vector<size_t>> inverted_lists_;
  std::vector<std::vector<float>> vectors_;
};
// --8<-- [end:ivf_index]
