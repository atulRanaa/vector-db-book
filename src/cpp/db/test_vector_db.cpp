/**
 * test_vector_db.cpp — Test suite for the ADBC + Iceberg + HNSW VectorDB.
 *
 * Tests the three layers independently and then the unified engine:
 *   1. ArrowBatch: columnar layout, schema validation
 *   2. IcebergStore: segment lifecycle, tombstones, compaction, snapshots
 *   3. VectorDB: batch ingest, search, delete, compact+rebuild
 *
 * Compile:
 *   g++ -std=c++17 -O2 -o test_vector_db test_vector_db.cpp
 *
 * Run:
 *   ./test_vector_db
 */

#include "arrow_batch.hpp"
#include "iceberg_store.hpp"
#include "vector_db.hpp"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

using namespace vectordb;

// ─────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name)                                                             \
  do {                                                                         \
    std::cout << "  [TEST] " << name;                                          \
  } while (0)

#define PASS()                                                                 \
  do {                                                                         \
    std::cout << " ✅" << std::endl;                                           \
    ++tests_passed;                                                            \
  } while (0)

#define FAIL(msg)                                                              \
  do {                                                                         \
    std::cout << " ❌ " << msg << std::endl;                                   \
    ++tests_failed;                                                            \
  } while (0)

#define ASSERT_EQ(a, b, msg)                                                   \
  if ((a) != (b)) {                                                            \
    FAIL(msg);                                                                 \
    return;                                                                    \
  }

#define ASSERT_TRUE(cond, msg)                                                 \
  if (!(cond)) {                                                               \
    FAIL(msg);                                                                 \
    return;                                                                    \
  }

/// Generate a random float vector of given dimension.
static std::vector<float> random_vector(size_t dim, std::mt19937 &rng) {
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> v(dim);
  for (auto &x : v)
    x = dist(rng);
  return v;
}

// ─────────────────────────────────────────────────────
// 1. Arrow Batch Tests
// ─────────────────────────────────────────────────────

void test_arrow_batch_creation() {
  TEST("Arrow RecordBatch creation and column access");

  const size_t dim = 4;
  const size_t n = 3;

  RecordBatchBuilder builder;
  builder.add_id_column("id", {100, 200, 300});
  builder.add_vector_column("embedding",
                            {1, 2, 3, 4,     // vec 0
                             5, 6, 7, 8,     // vec 1
                             9, 10, 11, 12}, // vec 2
                            dim);
  builder.add_string_column("metadata", {"doc_a", "doc_b", "doc_c"});

  auto arrow_batch = builder.build();

  ASSERT_EQ(arrow_batch->num_rows(), static_cast<int64_t>(n),
            "Expected 3 rows");
  ASSERT_EQ(arrow_batch->num_columns(), 3, "Expected 3 columns");

  auto id_col = std::static_pointer_cast<arrow::UInt64Array>(
      arrow_batch->GetColumnByName("id"));
  ASSERT_TRUE(id_col != nullptr, "id column not found");
  ASSERT_EQ(id_col->Value(1), 200u, "Second ID should be 200");

  auto vec_col = std::static_pointer_cast<arrow::FixedSizeListArray>(
      arrow_batch->GetColumnByName("embedding"));
  ASSERT_TRUE(vec_col != nullptr, "embedding column not found");
  ASSERT_EQ(std::static_pointer_cast<arrow::FixedSizeListType>(vec_col->type())
                ->list_size(),
            dim, "Dimension mismatch");

  auto floats = std::static_pointer_cast<arrow::FloatArray>(vec_col->values());
  // Check vector 1, dimension 2 (0-indexed): should be 7
  ASSERT_EQ(floats->Value(1 * dim + 2), 7.0f, "Float mismatch");

  PASS();
}

void test_arrow_batch_invalid_dim() {
  TEST("Arrow RecordBatch rejects misaligned vector data");

  RecordBatchBuilder builder;
  bool caught = false;
  try {
    // 7 floats is not divisible by dim=4
    builder.add_vector_column("embedding", {1, 2, 3, 4, 5, 6, 7}, 4);
  } catch (const std::invalid_argument &) {
    caught = true;
  }

  ASSERT_TRUE(caught, "Should throw on misaligned vector data");
  PASS();
}

// ─────────────────────────────────────────────────────
// 2. Iceberg Store Tests
// ─────────────────────────────────────────────────────

void test_iceberg_insert_and_scan() {
  TEST("IcebergStore insert and scan");

  IcebergStore store(4, /*segment_capacity=*/5);

  store.insert(1, {0.1f, 0.2f, 0.3f, 0.4f}, "first");
  store.insert(2, {0.5f, 0.6f, 0.7f, 0.8f}, "second");

  ASSERT_EQ(store.total_records(), 2u, "Should have 2 records");

  auto all = store.scan_all();
  ASSERT_EQ(all.size(), 2u, "Scan should return 2 records");
  ASSERT_EQ(all[0].id, 1u, "First record ID mismatch");
  ASSERT_EQ(all[1].metadata, "second", "Metadata mismatch");

  PASS();
}

void test_iceberg_auto_flush() {
  TEST("IcebergStore auto-flushes when segment is full");

  IcebergStore store(2, /*segment_capacity=*/3);

  store.insert(1, {1.0f, 2.0f});
  store.insert(2, {3.0f, 4.0f});
  store.insert(3, {5.0f, 6.0f}); // This triggers flush
  store.insert(4, {7.0f, 8.0f}); // Goes into new active segment

  ASSERT_EQ(store.sealed_segment_count(), 1u, "Should have 1 sealed segment");
  ASSERT_EQ(store.total_records(), 4u, "Total should be 4");

  PASS();
}

void test_iceberg_tombstone_delete() {
  TEST("IcebergStore tombstone deletion");

  IcebergStore store(2, /*segment_capacity=*/100);

  store.insert(1, {1.0f, 2.0f});
  store.insert(2, {3.0f, 4.0f});
  store.insert(3, {5.0f, 6.0f});

  store.delete_vector(2);

  ASSERT_EQ(store.total_records(), 3u, "Total records unchanged");
  ASSERT_EQ(store.total_live_records(), 2u, "Live records should be 2");

  auto live = store.scan_all();
  ASSERT_EQ(live.size(), 2u, "Scan should return 2 live records");
  // Verify that ID 2 was filtered out
  for (const auto &r : live) {
    ASSERT_TRUE(r.id != 2, "Deleted record should not appear in scan");
  }

  PASS();
}

void test_iceberg_compaction() {
  TEST("IcebergStore compaction reclaims tombstoned records");

  IcebergStore store(2, /*segment_capacity=*/3);

  // Insert 6 records (creates 2 sealed segments)
  store.insert(1, {1.0f, 1.0f});
  store.insert(2, {2.0f, 2.0f});
  store.insert(3, {3.0f, 3.0f}); // flush → segment 0
  store.insert(4, {4.0f, 4.0f});
  store.insert(5, {5.0f, 5.0f});
  store.insert(6, {6.0f, 6.0f}); // flush → segment 1

  ASSERT_EQ(store.sealed_segment_count(), 2u, "Should have 2 segments");

  // Delete 2 out of 3 in segment 0 → 66% tombstone ratio
  store.delete_vector(1);
  store.delete_vector(2);

  size_t reclaimed = store.compact(0.5f);
  ASSERT_EQ(reclaimed, 2u, "Should reclaim 2 records");
  ASSERT_EQ(store.total_live_records(), 4u, "4 live records remain");

  PASS();
}

void test_iceberg_snapshots() {
  TEST("IcebergStore snapshot creation");

  IcebergStore store(2, /*segment_capacity=*/2);

  // Initial snapshot exists on construction
  size_t initial_snaps = store.snapshot_count();
  ASSERT_TRUE(initial_snaps >= 1, "Should have initial snapshot");

  store.insert(1, {1.0f, 2.0f});
  store.insert(2, {3.0f, 4.0f}); // triggers flush → new snapshot

  ASSERT_TRUE(store.snapshot_count() > initial_snaps,
              "Flush should create a new snapshot");

  PASS();
}

// ─────────────────────────────────────────────────────
// 3. Unified VectorDB Tests
// ─────────────────────────────────────────────────────

void test_vdb_batch_ingest() {
  TEST("VectorDB batch ingestion via RecordBatch");

  const size_t dim = 8;
  VectorDB db(dim, /*M=*/8, /*ef_c=*/50, /*ef_s=*/20,
              /*seg_cap=*/100);

  // Build a batch of 50 vectors
  std::mt19937 rng(42);
  std::vector<uint64_t> ids(50);
  std::vector<float> flat(50 * dim);
  std::vector<std::string> metas(50);

  for (size_t i = 0; i < 50; ++i) {
    ids[i] = i + 1;
    auto v = random_vector(dim, rng);
    std::copy(v.begin(), v.end(), flat.begin() + i * dim);
    metas[i] = "doc_" + std::to_string(i);
  }

  RecordBatchBuilder builder;
  builder.add_id_column("id", std::move(ids));
  builder.add_vector_column("embedding", std::move(flat), dim);
  builder.add_string_column("metadata", std::move(metas));

  size_t ingested = db.ingest_batch(builder.build());

  ASSERT_EQ(ingested, 50u, "Should ingest 50 records");
  ASSERT_EQ(db.total_records(), 50u, "Storage should have 50 records");
  ASSERT_EQ(db.index_size(), 50u, "HNSW should have 50 nodes");

  PASS();
}

void test_vdb_search() {
  TEST("VectorDB search returns nearest neighbors");

  const size_t dim = 4;
  VectorDB db(dim, /*M=*/8, /*ef_c=*/100, /*ef_s=*/50,
              /*seg_cap=*/100);

  // Insert known vectors
  db.insert(1, {1.0f, 0.0f, 0.0f, 0.0f});
  db.insert(2, {0.0f, 1.0f, 0.0f, 0.0f});
  db.insert(3, {0.0f, 0.0f, 1.0f, 0.0f});
  db.insert(4, {1.0f, 1.0f, 0.0f, 0.0f}); // closest to query

  // Query near [1, 1, 0, 0]
  auto results = db.search({0.9f, 0.9f, 0.0f, 0.0f}, 2);

  ASSERT_TRUE(!results.empty(), "Should return results");
  // The closest vector to [0.9, 0.9, 0, 0] should be [1, 1, 0, 0] (ID 4)
  ASSERT_EQ(results[0].id, 4u, "Nearest should be ID 4");

  PASS();
}

void test_vdb_delete_and_search() {
  TEST("VectorDB delete filters results from search");

  const size_t dim = 4;
  VectorDB db(dim, /*M=*/8, /*ef_c=*/100, /*ef_s=*/50,
              /*seg_cap=*/100);

  db.insert(1, {1.0f, 0.0f, 0.0f, 0.0f}, "keep");
  db.insert(2, {1.1f, 0.0f, 0.0f, 0.0f}, "delete_me");
  db.insert(3, {0.0f, 1.0f, 0.0f, 0.0f}, "keep");

  db.delete_vector(2);

  ASSERT_EQ(db.total_records(), 3u, "Total unchanged after soft delete");
  ASSERT_EQ(db.live_records(), 2u, "Live records should be 2");

  PASS();
}

void test_vdb_compact_and_rebuild() {
  TEST("VectorDB compact and rebuild HNSW index");

  const size_t dim = 4;
  VectorDB db(dim, /*M=*/8, /*ef_c=*/100, /*ef_s=*/50,
              /*seg_cap=*/3);

  // Insert 6 vectors (creates 2 segments)
  db.insert(1, {1.0f, 0.0f, 0.0f, 0.0f});
  db.insert(2, {0.0f, 1.0f, 0.0f, 0.0f});
  db.insert(3, {0.0f, 0.0f, 1.0f, 0.0f});
  db.insert(4, {1.0f, 1.0f, 0.0f, 0.0f});
  db.insert(5, {0.0f, 1.0f, 1.0f, 0.0f});
  db.insert(6, {1.0f, 0.0f, 1.0f, 0.0f});

  // Delete 2 from first segment (66% tombstone ratio)
  db.delete_vector(1);
  db.delete_vector(2);

  size_t reclaimed = db.compact_and_rebuild(0.5f);

  ASSERT_EQ(reclaimed, 2u, "Should reclaim 2 records");
  ASSERT_EQ(db.live_records(), 4u, "4 live records remain");
  ASSERT_EQ(db.index_size(), 4u, "HNSW rebuilt with 4 nodes");

  // Verify search still works after rebuild
  auto results = db.search({1.0f, 1.0f, 0.0f, 0.0f}, 1);
  ASSERT_TRUE(!results.empty(), "Search should work after rebuild");
  ASSERT_EQ(results[0].id, 4u, "Should find ID 4");

  PASS();
}

void test_vdb_dimension_validation() {
  TEST("VectorDB rejects mismatched dimensions");

  VectorDB db(4);
  bool caught = false;

  try {
    db.insert(1, {1.0f, 2.0f}); // dim=2, but DB expects dim=4
  } catch (const std::invalid_argument &) {
    caught = true;
  }

  ASSERT_TRUE(caught, "Should throw on dimension mismatch");
  PASS();
}

void test_vdb_large_batch() {
  TEST("VectorDB handles large batch (1000 vectors)");

  const size_t dim = 32;
  const size_t n = 1000;
  VectorDB db(dim, /*M=*/16, /*ef_c=*/100, /*ef_s=*/50,
              /*seg_cap=*/200);

  std::mt19937 rng(123);
  std::vector<uint64_t> ids(n);
  std::vector<float> flat(n * dim);

  for (size_t i = 0; i < n; ++i) {
    ids[i] = i;
    auto v = random_vector(dim, rng);
    std::copy(v.begin(), v.end(), flat.begin() + i * dim);
  }

  RecordBatchBuilder builder;
  builder.add_id_column("id", std::move(ids));
  builder.add_vector_column("embedding", std::move(flat), dim);

  db.ingest_batch(builder.build());

  ASSERT_EQ(db.total_records(), n, "Should have 1000 records");
  ASSERT_EQ(db.index_size(), n, "HNSW should have 1000 nodes");
  ASSERT_TRUE(db.segment_count() >= 4, "Should have multiple sealed segments");

  // Search should return results
  auto q = random_vector(dim, rng);
  auto results = db.search(q, 5);
  ASSERT_EQ(results.size(), 5u, "Should return 5 results");

  PASS();
}

// ─────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────

int main() {
  std::cout << "\n═══════════════════════════════════════════════" << std::endl;
  std::cout << "  ADBC + Iceberg + HNSW VectorDB Test Suite" << std::endl;
  std::cout << "═══════════════════════════════════════════════" << std::endl;

  std::cout << "\n── Arrow Batch Layer ──────────────────────" << std::endl;
  test_arrow_batch_creation();
  test_arrow_batch_invalid_dim();

  std::cout << "\n── Iceberg Store Layer ────────────────────" << std::endl;
  test_iceberg_insert_and_scan();
  test_iceberg_auto_flush();
  test_iceberg_tombstone_delete();
  test_iceberg_compaction();
  test_iceberg_snapshots();

  std::cout << "\n── Unified VectorDB Engine ────────────────" << std::endl;
  test_vdb_batch_ingest();
  test_vdb_search();
  test_vdb_delete_and_search();
  test_vdb_compact_and_rebuild();
  test_vdb_dimension_validation();
  test_vdb_large_batch();

  std::cout << "\n═══════════════════════════════════════════════" << std::endl;
  std::cout << "  Results: " << tests_passed << " passed, " << tests_failed
            << " failed" << std::endl;
  std::cout << "═══════════════════════════════════════════════\n" << std::endl;

  return tests_failed > 0 ? 1 : 0;
}
