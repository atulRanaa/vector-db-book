/**
 * Locality-Sensitive Hashing (LSH) for approximate nearest neighbor search.
 *
 * Implements random-hyperplane LSH (cosine similarity) and
 * p-stable distribution LSH (Euclidean distance).
 *
 * Compile: g++ -std=c++17 -O3 -o lsh lsh.cpp
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// --8<-- [start:hash_signature]
/**
 * Hash signature — a sequence of bits/ints that represents
 * a vector's position in a hash table.
 */
struct HashSignature {
  std::vector<int> bits;

  bool operator==(const HashSignature &other) const {
    return bits == other.bits;
  }
};

struct HashSignatureHash {
  size_t operator()(const HashSignature &sig) const {
    size_t seed = sig.bits.size();
    for (auto &b : sig.bits) {
      seed ^= std::hash<int>{}(b) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};
// --8<-- [end:hash_signature]

// --8<-- [start:random_hyperplane_lsh]
/**
 * Random Hyperplane LSH for cosine similarity.
 *
 * Each hash function: h(x) = sign(r · x)
 * where r is a random Gaussian vector.
 *
 * Collision probability:
 *   Pr[h(x) = h(y)] = 1 - θ(x,y) / π
 *
 * Parameters:
 *   dim         — dimensionality of input vectors
 *   num_tables  — number of hash tables L (more tables → higher recall)
 *   num_hashes  — hash bits per table k (more bits → higher precision)
 */
class RandomHyperplaneLSH {
public:
  RandomHyperplaneLSH(size_t dim, size_t num_tables = 10, size_t num_hashes = 8)
      : dim_(dim), num_tables_(num_tables), num_hashes_(num_hashes) {
    std::mt19937 rng(42);
    std::normal_distribution<float> normal(0.0f, 1.0f);

    // Generate random hyperplanes for each table
    hyperplanes_.resize(num_tables);
    for (size_t t = 0; t < num_tables; ++t) {
      hyperplanes_[t].resize(num_hashes * dim);
      for (auto &val : hyperplanes_[t]) {
        val = normal(rng);
      }
    }

    tables_.resize(num_tables);
  }

  /**
   * Build the index from a set of vectors.
   */
  void build(const std::vector<std::vector<float>> &vectors) {
    vectors_ = vectors;
    for (auto &table : tables_)
      table.clear();

    for (size_t i = 0; i < vectors.size(); ++i) {
      for (size_t t = 0; t < num_tables_; ++t) {
        auto sig = hash(vectors[i], t);
        tables_[t][sig].push_back(i);
      }
    }
  }

  /**
   * Query for k approximate nearest neighbors.
   *
   * 1. Hash query into each table
   * 2. Collect all candidate IDs from matching buckets
   * 3. Re-rank candidates by exact cosine similarity
   */
  std::vector<size_t> query(const std::vector<float> &q, size_t k) const {
    std::unordered_set<size_t> candidates;

    for (size_t t = 0; t < num_tables_; ++t) {
      auto sig = hash(q, t);
      auto it = tables_[t].find(sig);
      if (it != tables_[t].end()) {
        for (size_t idx : it->second) {
          candidates.insert(idx);
        }
      }
    }

    // Re-rank by cosine similarity
    std::vector<std::pair<float, size_t>> scored;
    scored.reserve(candidates.size());
    for (size_t idx : candidates) {
      float sim = cosine_sim(q, vectors_[idx]);
      scored.push_back({-sim, idx}); // negate for ascending sort
    }
    std::sort(scored.begin(), scored.end());

    std::vector<size_t> result;
    for (size_t i = 0; i < std::min(k, scored.size()); ++i) {
      result.push_back(scored[i].second);
    }
    return result;
  }

  size_t num_vectors() const { return vectors_.size(); }

private:
  HashSignature hash(const std::vector<float> &vec, size_t table_idx) const {
    HashSignature sig;
    sig.bits.resize(num_hashes_);
    for (size_t h = 0; h < num_hashes_; ++h) {
      float dot = 0.0f;
      const float *plane = &hyperplanes_[table_idx][h * dim_];
      for (size_t d = 0; d < dim_; ++d) {
        dot += plane[d] * vec[d];
      }
      sig.bits[h] = (dot > 0.0f) ? 1 : 0;
    }
    return sig;
  }

  static float cosine_sim(const std::vector<float> &a,
                          const std::vector<float> &b) {
    float dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
      dot += a[i] * b[i];
      na += a[i] * a[i];
      nb += b[i] * b[i];
    }
    float denom = std::sqrt(na) * std::sqrt(nb);
    return (denom > 1e-10f) ? dot / denom : 0.0f;
  }

  size_t dim_, num_tables_, num_hashes_;
  std::vector<std::vector<float>> hyperplanes_; // [table][hash*dim + d]
  std::vector<
      std::unordered_map<HashSignature, std::vector<size_t>, HashSignatureHash>>
      tables_;
  std::vector<std::vector<float>> vectors_;
};
// --8<-- [end:random_hyperplane_lsh]

// --8<-- [start:euclidean_lsh]
/**
 * p-stable distribution LSH for Euclidean distance.
 *
 * Hash function: h(x) = floor((a · x + b) / w)
 * where a ~ N(0,I), b ~ Uniform(0,w).
 *
 * Parameters:
 *   bucket_width — width of hash buckets (w)
 */
class EuclideanLSH {
public:
  EuclideanLSH(size_t dim, size_t num_tables = 10, size_t num_hashes = 8,
               float bucket_width = 4.0f)
      : dim_(dim), num_tables_(num_tables), num_hashes_(num_hashes),
        w_(bucket_width) {
    std::mt19937 rng(123);
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_real_distribution<float> uniform(0.0f, bucket_width);

    projections_.resize(num_tables);
    offsets_.resize(num_tables);
    for (size_t t = 0; t < num_tables; ++t) {
      projections_[t].resize(num_hashes * dim);
      offsets_[t].resize(num_hashes);
      for (auto &v : projections_[t])
        v = normal(rng);
      for (auto &v : offsets_[t])
        v = uniform(rng);
    }
    tables_.resize(num_tables);
  }

  void build(const std::vector<std::vector<float>> &vectors) {
    vectors_ = vectors;
    for (auto &table : tables_)
      table.clear();

    for (size_t i = 0; i < vectors.size(); ++i) {
      for (size_t t = 0; t < num_tables_; ++t) {
        auto sig = hash(vectors[i], t);
        tables_[t][sig].push_back(i);
      }
    }
  }

  std::vector<size_t> query(const std::vector<float> &q, size_t k) const {
    std::unordered_set<size_t> candidates;
    for (size_t t = 0; t < num_tables_; ++t) {
      auto sig = hash(q, t);
      auto it = tables_[t].find(sig);
      if (it != tables_[t].end()) {
        for (size_t idx : it->second)
          candidates.insert(idx);
      }
    }

    std::vector<std::pair<float, size_t>> scored;
    for (size_t idx : candidates) {
      scored.push_back({l2_squared(q, vectors_[idx]), idx});
    }
    std::sort(scored.begin(), scored.end());

    std::vector<size_t> result;
    for (size_t i = 0; i < std::min(k, scored.size()); ++i) {
      result.push_back(scored[i].second);
    }
    return result;
  }

private:
  HashSignature hash(const std::vector<float> &vec, size_t table_idx) const {
    HashSignature sig;
    sig.bits.resize(num_hashes_);
    for (size_t h = 0; h < num_hashes_; ++h) {
      float proj = offsets_[table_idx][h];
      const float *a = &projections_[table_idx][h * dim_];
      for (size_t d = 0; d < dim_; ++d)
        proj += a[d] * vec[d];
      sig.bits[h] = static_cast<int>(std::floor(proj / w_));
    }
    return sig;
  }

  static float l2_squared(const std::vector<float> &a,
                          const std::vector<float> &b) {
    float sum = 0;
    for (size_t i = 0; i < a.size(); ++i) {
      float d = a[i] - b[i];
      sum += d * d;
    }
    return sum;
  }

  size_t dim_, num_tables_, num_hashes_;
  float w_;
  std::vector<std::vector<float>> projections_;
  std::vector<std::vector<float>> offsets_;
  std::vector<
      std::unordered_map<HashSignature, std::vector<size_t>, HashSignatureHash>>
      tables_;
  std::vector<std::vector<float>> vectors_;
};
// --8<-- [end:euclidean_lsh]
