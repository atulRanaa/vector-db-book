/**
 * Test suite for HNSW, LSH, IVF, and PQ implementations.
 *
 * Compile & run:
 *   g++ -std=c++17 -O2 -I../../src/cpp -o test_algorithms test_algorithms.cpp
 * && ./test_algorithms
 */

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "../../src/cpp/hnsw.hpp"
#include "../../src/cpp/ivf.hpp"
#include "../../src/cpp/lsh.hpp"
#include "../../src/cpp/pq.hpp"

static int tests_passed = 0;
static int tests_failed = 0;

void check(bool condition, const std::string &name) {
  if (condition) {
    std::cout << "  ✓ " << name << std::endl;
    tests_passed++;
  } else {
    std::cout << "  ✗ FAILED: " << name << std::endl;
    tests_failed++;
  }
}

// Generate random dataset
std::vector<std::vector<float>> generate_data(size_t n, size_t d,
                                              unsigned seed = 42) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<std::vector<float>> data(n, std::vector<float>(d));
  for (auto &v : data)
    for (auto &x : v)
      x = dist(rng);
  return data;
}

// Brute-force ground truth
std::vector<size_t> brute_force_knn(const std::vector<float> &query,
                                    const std::vector<std::vector<float>> &data,
                                    size_t k) {
  std::vector<std::pair<float, size_t>> dists;
  for (size_t i = 0; i < data.size(); ++i) {
    float d = 0;
    for (size_t j = 0; j < query.size(); ++j) {
      float diff = query[j] - data[i][j];
      d += diff * diff;
    }
    dists.push_back({d, i});
  }
  std::partial_sort(dists.begin(), dists.begin() + k, dists.end());
  std::vector<size_t> result;
  for (size_t i = 0; i < k; ++i)
    result.push_back(dists[i].second);
  return result;
}

// Compute recall@k
float compute_recall(const std::vector<size_t> &approx,
                     const std::vector<size_t> &exact, size_t k) {
  std::set<size_t> truth(exact.begin(),
                         exact.begin() + std::min(k, exact.size()));
  int found = 0;
  for (size_t i = 0; i < std::min(k, approx.size()); ++i) {
    if (truth.count(approx[i]))
      found++;
  }
  return static_cast<float>(found) / k;
}

// ────────────── HNSW Tests ──────────────

void test_hnsw_basic() {
  std::cout << "\n[test_hnsw_basic]" << std::endl;

  HNSWIndex idx(4, /*M=*/8, /*ef_construction=*/100, /*ef_search=*/50);

  // Insert 3 vectors
  idx.insert({1, 0, 0, 0});
  idx.insert({0, 1, 0, 0});
  idx.insert({1, 1, 0, 0});

  check(idx.size() == 3, "inserted 3 vectors");

  auto results = idx.search({1, 0, 0, 0}, 1);
  check(!results.empty(), "search returns results");
  check(results[0].id == 0, "nearest to [1,0,0,0] is itself");
  check(results[0].distance < 1e-6f, "distance to itself is ~0");
}

void test_hnsw_recall() {
  std::cout << "\n[test_hnsw_recall]" << std::endl;

  const size_t n = 1000, d = 32, k = 10;
  auto data = generate_data(n, d);

  HNSWIndex idx(d, /*M=*/16, /*ef_construction=*/200, /*ef_search=*/100);
  idx.build(data);
  check(idx.size() == n, "built index with 1000 vectors");

  // Test recall on 10 random queries
  float total_recall = 0;
  auto queries = generate_data(10, d, 999);
  for (const auto &q : queries) {
    auto approx = idx.search(q, k);
    auto exact = brute_force_knn(q, data, k);
    std::vector<size_t> approx_ids;
    for (auto &r : approx)
      approx_ids.push_back(r.id);
    total_recall += compute_recall(approx_ids, exact, k);
  }
  float avg_recall = total_recall / 10;
  check(avg_recall >= 0.7f,
        "recall@10 ≥ 0.7 (got " + std::to_string(avg_recall) + ")");
}

// ────────────── LSH Tests ──────────────

void test_lsh_cosine() {
  std::cout << "\n[test_lsh_cosine]" << std::endl;

  const size_t n = 500, d = 16, k = 5;
  auto data = generate_data(n, d);

  RandomHyperplaneLSH lsh(d, /*num_tables=*/15, /*num_hashes=*/6);
  lsh.build(data);
  check(lsh.num_vectors() == n, "built cosine LSH index");

  auto results = lsh.query(data[0], k);
  check(!results.empty(), "query returns candidates");

  // First result should be the query itself
  bool found_self = false;
  for (auto id : results)
    if (id == 0)
      found_self = true;
  check(found_self, "query vector found in its own results");
}

void test_lsh_euclidean() {
  std::cout << "\n[test_lsh_euclidean]" << std::endl;

  const size_t n = 500, d = 16, k = 5;
  auto data = generate_data(n, d);

  EuclideanLSH lsh(d, /*num_tables=*/15, /*num_hashes=*/6,
                   /*bucket_width=*/4.0f);
  lsh.build(data);

  auto results = lsh.query(data[0], k);
  check(!results.empty(), "Euclidean LSH returns candidates");
}

// ────────────── PQ Tests ──────────────

void test_pq_encode_decode() {
  std::cout << "\n[test_pq_encode_decode]" << std::endl;

  const size_t n = 200, d = 16, M = 4;
  auto data = generate_data(n, d);

  ProductQuantizer pq(d, M, /*K=*/64);
  pq.train(data, /*n_iter=*/15);

  auto codes = pq.encode(data);
  check(codes.size() == n, "encoded all vectors");
  check(codes[0].size() == M, "code length = M");

  // Decode and check reconstruction error is bounded
  float total_error = 0;
  for (size_t i = 0; i < n; ++i) {
    auto reconstructed = pq.decode(codes[i]);
    for (size_t j = 0; j < d; ++j) {
      float diff = data[i][j] - reconstructed[j];
      total_error += diff * diff;
    }
  }
  float avg_error = total_error / n;
  check(avg_error < 50.0f, "avg reconstruction error < 50 (got " +
                               std::to_string(avg_error) + ")");
}

void test_pq_search() {
  std::cout << "\n[test_pq_search]" << std::endl;

  const size_t n = 300, d = 16, M = 4, k = 5;
  auto data = generate_data(n, d);

  ProductQuantizer pq(d, M, /*K=*/64);
  pq.train(data, 15);
  auto codes = pq.encode(data);

  auto results = pq.search_adc(data[0], codes, k);
  check(results.size() == k, "ADC returns k results");
  check(results[0].id == 0, "closest to query is itself");
}

// ────────────── IVF Tests ──────────────

void test_ivf_basic() {
  std::cout << "\n[test_ivf_basic]" << std::endl;

  const size_t n = 500, d = 16, k = 10;
  auto data = generate_data(n, d);

  IVFIndex ivf(d, /*nlist=*/20, /*nprobe=*/5);
  ivf.train(data, /*n_iter=*/10);
  ivf.add(data);
  check(ivf.size() == n, "IVF index has correct size");

  auto results = ivf.search(data[0], k);
  check(!results.empty(), "IVF search returns results");
  check(results[0].id == 0, "nearest is the query itself");
}

void test_ivf_recall() {
  std::cout << "\n[test_ivf_recall]" << std::endl;

  const size_t n = 1000, d = 32, k = 10;
  auto data = generate_data(n, d);

  IVFIndex ivf(d, /*nlist=*/50, /*nprobe=*/10);
  ivf.train(data, 15);
  ivf.add(data);

  float total_recall = 0;
  auto queries = generate_data(10, d, 999);
  for (const auto &q : queries) {
    auto approx = ivf.search(q, k);
    auto exact = brute_force_knn(q, data, k);
    std::vector<size_t> approx_ids;
    for (auto &r : approx)
      approx_ids.push_back(r.id);
    total_recall += compute_recall(approx_ids, exact, k);
  }
  float avg_recall = total_recall / 10;
  check(avg_recall >= 0.5f,
        "IVF recall@10 ≥ 0.5 (got " + std::to_string(avg_recall) + ")");
}

// ────────────── Main ──────────────

int main() {
  std::cout << "=== Vector Database Algorithm Tests ===" << std::endl;

  test_hnsw_basic();
  test_hnsw_recall();
  test_lsh_cosine();
  test_lsh_euclidean();
  test_pq_encode_decode();
  test_pq_search();
  test_ivf_basic();
  test_ivf_recall();

  std::cout << "\n--- Results: " << tests_passed << " passed, " << tests_failed
            << " failed ---" << std::endl;
  return tests_failed > 0 ? 1 : 0;
}
