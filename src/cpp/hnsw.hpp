/**
 * Hierarchical Navigable Small World (HNSW) graph index.
 *
 * A complete C++ implementation of the HNSW algorithm.
 * Reference: Malkov & Yashunin, "Efficient and Robust Approximate
 *            Nearest Neighbor Search Using HNSW Graphs" (2020)
 *
 * Compile: g++ -std=c++17 -O3 -o hnsw hnsw.hpp
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>
#include <queue>
#include <random>
#include <unordered_set>
#include <vector>

// --8<-- [start:hnsw_index]
/**
 * HNSW Index for approximate nearest neighbor search.
 *
 * Builds a multi-layer navigable small-world graph.
 * - Layer 0 is the densest (all nodes present)
 * - Higher layers are exponentially sparser
 * - Search descends from top layer greedily, then does
 *   beam search at layer 0.
 *
 * Template parameters are avoided for clarity; uses float vectors.
 *
 * Key parameters:
 *   M               — max edges per node per layer (default 16)
 *   M_max0          — max edges at layer 0 (default 2*M)
 *   ef_construction  — beam width during build
 *   ef_search        — beam width during query
 *   mL              — level generation factor = 1/ln(M)
 */
class HNSWIndex {
public:
  struct SearchResult {
    float distance;
    size_t id;
    bool operator>(const SearchResult &o) const {
      return distance > o.distance;
    }
    bool operator<(const SearchResult &o) const {
      return distance < o.distance;
    }
  };

  HNSWIndex(size_t dim, size_t M = 16, size_t ef_construction = 200,
            size_t ef_search = 50)
      : dim_(dim), M_(M), M_max0_(2 * M), ef_construction_(ef_construction),
        ef_search_(ef_search), mL_(1.0 / std::log(static_cast<double>(M))),
        entry_point_(NONE), max_layer_(0), rng_(42), uniform_(0.0, 1.0) {}

  /**
   * Insert a single vector into the index.
   *
   * Algorithm (simplified):
   * 1. Draw random level l ~ Geometric(mL)
   * 2. From entry point, greedily descend to layer l+1
   * 3. At each layer [l..0], beam-search for ef_construction
   *    neighbors and add bidirectional edges
   */
  size_t insert(const std::vector<float> &vec) {
    size_t id = vectors_.size();
    vectors_.push_back(vec);

    int level = random_level();

    // Grow graph layers if needed
    while (static_cast<int>(graph_.size()) <= level) {
      graph_.emplace_back();
    }

    // First element
    if (entry_point_ == NONE) {
      entry_point_ = id;
      max_layer_ = level;
      for (int l = 0; l <= level; ++l) {
        if (graph_[l].size() <= id)
          graph_[l].resize(id + 1);
      }
      return id;
    }

    // Ensure node has adjacency lists at all its layers
    for (int l = 0; l <= level; ++l) {
      if (graph_[l].size() <= id)
        graph_[l].resize(id + 1);
    }

    size_t current = entry_point_;

    // Phase 1: Greedy descent from max_layer to level+1
    for (int l = max_layer_; l > level; --l) {
      auto nearest = search_layer(vec, current, 1, l);
      if (!nearest.empty())
        current = nearest[0].id;
    }

    // Phase 2: Insert at layers [min(level, max_layer)..0]
    for (int l = std::min(level, max_layer_); l >= 0; --l) {
      auto candidates = search_layer(vec, current, ef_construction_, l);
      size_t M_max = (l == 0) ? M_max0_ : M_;
      auto neighbors = select_neighbors(candidates, M_max);

      // Add bidirectional edges
      for (auto &nb : neighbors) {
        graph_[l][id].push_back(nb.id);
        graph_[l][nb.id].push_back(id);

        // Prune neighbor if over capacity
        if (graph_[l][nb.id].size() > M_max) {
          prune(nb.id, l, M_max);
        }
      }

      if (!candidates.empty())
        current = candidates[0].id;
    }

    if (level > max_layer_) {
      entry_point_ = id;
      max_layer_ = level;
    }

    return id;
  }

  /**
   * Search for k approximate nearest neighbors.
   *
   * 1. Greedy descent from top layer to layer 1
   * 2. Beam search at layer 0 with ef = max(ef_search, k)
   * 3. Return top-k results sorted by distance
   */
  std::vector<SearchResult> search(const std::vector<float> &query,
                                   size_t k) const {
    if (entry_point_ == NONE)
      return {};

    size_t current = entry_point_;

    // Descend from top
    for (int l = max_layer_; l > 0; --l) {
      auto nearest = search_layer(query, current, 1, l);
      if (!nearest.empty())
        current = nearest[0].id;
    }

    size_t ef = std::max(ef_search_, k);
    auto results = search_layer(query, current, ef, 0);

    // Take sqrt for actual Euclidean distances
    if (results.size() > k)
      results.resize(k);
    for (auto &r : results) {
      r.distance = std::sqrt(r.distance);
    }
    return results;
  }

  /**
   * Bulk insert all vectors.
   */
  void build(const std::vector<std::vector<float>> &vectors) {
    for (const auto &v : vectors)
      insert(v);
  }

  size_t size() const { return vectors_.size(); }
  size_t num_layers() const { return graph_.size(); }

  void set_ef_search(size_t ef) { ef_search_ = ef; }

private:
  static constexpr size_t NONE = std::numeric_limits<size_t>::max();

  float distance_sq(const std::vector<float> &a,
                    const std::vector<float> &b) const {
    float sum = 0.0f;
    for (size_t i = 0; i < dim_; ++i) {
      float d = a[i] - b[i];
      sum += d * d;
    }
    return sum;
  }

  int random_level() {
    return static_cast<int>(-std::log(uniform_(rng_)) * mL_);
  }

  /**
   * Beam search in a single layer.
   * Returns up to ef nearest elements, sorted by distance (ascending).
   */
  std::vector<SearchResult> search_layer(const std::vector<float> &query,
                                         size_t entry, size_t ef,
                                         int layer) const {
    if (layer >= static_cast<int>(graph_.size()))
      return {};
    if (entry >= graph_[layer].size())
      return {};

    std::unordered_set<size_t> visited;
    visited.insert(entry);

    float d = distance_sq(query, vectors_[entry]);

    // candidates: min-heap (closest first)
    std::priority_queue<SearchResult, std::vector<SearchResult>,
                        std::greater<SearchResult>>
        candidates;
    // results: max-heap (farthest first, for bounding)
    std::priority_queue<SearchResult> results;

    candidates.push({d, entry});
    results.push({d, entry});

    while (!candidates.empty()) {
      auto [c_dist, c_id] = candidates.top();
      candidates.pop();

      float farthest = results.top().distance;
      if (c_dist > farthest)
        break;

      // Expand neighbors
      if (c_id < graph_[layer].size()) {
        for (size_t nb : graph_[layer][c_id]) {
          if (visited.count(nb))
            continue;
          visited.insert(nb);

          float nb_dist = distance_sq(query, vectors_[nb]);
          farthest = results.top().distance;

          if (nb_dist < farthest || results.size() < ef) {
            candidates.push({nb_dist, nb});
            results.push({nb_dist, nb});
            if (results.size() > ef)
              results.pop();
          }
        }
      }
    }

    // Extract sorted results
    std::vector<SearchResult> out;
    out.reserve(results.size());
    while (!results.empty()) {
      out.push_back(results.top());
      results.pop();
    }
    std::sort(out.begin(), out.end());
    return out;
  }

  std::vector<SearchResult>
  select_neighbors(const std::vector<SearchResult> &candidates,
                   size_t M) const {
    auto sorted = candidates;
    std::sort(sorted.begin(), sorted.end());
    if (sorted.size() > M)
      sorted.resize(M);
    return sorted;
  }

  void prune(size_t node, int layer, size_t M_max) {
    auto &adj = graph_[layer][node];
    std::vector<SearchResult> scored;
    for (size_t nb : adj) {
      scored.push_back({distance_sq(vectors_[node], vectors_[nb]), nb});
    }
    std::sort(scored.begin(), scored.end());
    adj.clear();
    for (size_t i = 0; i < std::min(M_max, scored.size()); ++i) {
      adj.push_back(scored[i].id);
    }
  }

  size_t dim_;
  size_t M_, M_max0_;
  size_t ef_construction_, ef_search_;
  double mL_;
  size_t entry_point_;
  int max_layer_;

  std::mt19937 rng_;
  std::uniform_real_distribution<double> uniform_;

  std::vector<std::vector<float>> vectors_;
  // graph_[layer][node_id] = list of neighbor IDs
  std::vector<std::vector<std::vector<size_t>>> graph_;
};
// --8<-- [end:hnsw_index]
