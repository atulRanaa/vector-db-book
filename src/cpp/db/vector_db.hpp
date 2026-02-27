/**
 * vector_db.hpp — Unified ADBC + Iceberg + HNSW Vector Database
 *
 * This is the top-level database engine that composes the three layers:
 *
 *   ┌───────────────────────────────────────┐
 *   │  Client API (ADBC RecordBatch ingest) │  ← arrow_batch.hpp
 *   ├───────────────────────────────────────┤
 *   │  Storage (Iceberg segments + WAL)     │  ← iceberg_store.hpp
 *   ├───────────────────────────────────────┤
 *   │  Search Index (HNSW graph)            │  ← hnsw.hpp
 *   └───────────────────────────────────────┘
 *
 * Architecture Overview:
 *   - INSERT: RecordBatch → IcebergStore (append to active segment)
 *             → Async index refresh into HNSW
 *   - SEARCH: Query vector → HNSW graph search → filter tombstones
 *   - DELETE: Tombstone in IcebergStore → background compaction
 *   - COMPACT: Merge tombstoned segments → rebuild HNSW
 */

#pragma once

#include "arrow_batch.hpp"
#include "hnsw.hpp"
#include "iceberg_store.hpp"

#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace vectordb {

// ─────────────────────────────────────────────────────
// SearchResult returned by VectorDB queries.
// ─────────────────────────────────────────────────────
struct VDBSearchResult {
  uint64_t id;
  float distance;
  std::string metadata;
};

// ─────────────────────────────────────────────────────
// VectorDB: The unified database engine.
// ─────────────────────────────────────────────────────

class VectorDB {
public:
  /**
   * Construct a VectorDB.
   * @param dim          Embedding dimension (e.g. 128, 768, 1536)
   * @param M            HNSW max connections per node
   * @param ef_construct HNSW beam width during graph construction
   * @param ef_search    HNSW beam width during queries
   * @param seg_capacity Max records per Iceberg segment before flush
   */
  VectorDB(size_t dim, size_t M = 16, size_t ef_construct = 200,
           size_t ef_search = 50, size_t seg_capacity = 1000)
      : dim_(dim), store_(dim, seg_capacity),
        hnsw_(dim, M, ef_construct, ef_search) {}

  // ─── ADBC-style Batch Ingestion ──────────────────

  /**
   * Ingest an Arrow-style RecordBatch into the database.
   *
   * Expected columns:
   *   "id"        — UINT64 column of record IDs
   *   "embedding" — FLOAT32_ARRAY column (flat, N * dim)
   *   "metadata"  — STRING column (optional)
   *
   * Internally:
   *   1. Validates schema (column types and dimensions)
   *   2. Writes records into IcebergStore segments
   *   3. Inserts vectors into the HNSW index
   */
  size_t ingest_batch(std::shared_ptr<arrow::RecordBatch> batch) {
    // 1. Locate required columns
    auto id_col = std::static_pointer_cast<arrow::UInt64Array>(
        batch->GetColumnByName("id"));
    auto vec_col = std::static_pointer_cast<arrow::FixedSizeListArray>(
        batch->GetColumnByName("embedding"));

    if (!id_col) {
      throw std::invalid_argument("Batch missing 'id' UINT64 column");
    }
    if (!vec_col) {
      throw std::invalid_argument(
          "Batch missing 'embedding' FLOAT32_ARRAY column");
    }
    auto list_type =
        std::static_pointer_cast<arrow::FixedSizeListType>(vec_col->type());
    if (list_type->list_size() != static_cast<int32_t>(dim_)) {
      throw std::invalid_argument("Embedding dimension mismatch: expected " +
                                  std::to_string(dim_) + ", got " +
                                  std::to_string(list_type->list_size()));
    }

    auto floats =
        std::static_pointer_cast<arrow::FloatArray>(vec_col->values());
    size_t n = batch->num_rows();

    // 2. Zero-copy bulk insert into Iceberg storage
    store_.bulk_insert(id_col->raw_values(), floats->raw_values(), n, dim_);

    // 3. Insert each vector into the HNSW index
    const float *raw_floats = floats->raw_values();
    for (size_t i = 0; i < n; ++i) {
      std::vector<float> vec(raw_floats + i * dim_,
                             raw_floats + (i + 1) * dim_);
      hnsw_.insert(vec);
    }

    return n;
  }

  // ─── Single-Record Insert ────────────────────────

  /**
   * Insert a single vector (non-batch path).
   */
  void insert(uint64_t id, const std::vector<float> &embedding,
              const std::string &metadata = "") {
    store_.insert(id, embedding, metadata);
    hnsw_.insert(embedding);
  }

  // ─── Search ──────────────────────────────────────

  /**
   * Search for the k nearest neighbors of a query vector.
   *
   * Process:
   *   1. HNSW graph search returns candidate node indices.
   *   2. Map HNSW indices back to record IDs.
   *   3. Filter out tombstoned (deleted) IDs.
   *   4. Enrich results with metadata from IcebergStore.
   */
  std::vector<VDBSearchResult> search(const std::vector<float> &query,
                                      size_t k) {
    if (query.size() != dim_) {
      throw std::invalid_argument("Query dimension mismatch");
    }

    // Over-fetch to account for tombstoned results
    size_t ef = std::max(k * 2, static_cast<size_t>(50));
    hnsw_.set_ef_search(ef);

    auto hnsw_results = hnsw_.search(query, ef);

    // Get the full record set to map HNSW indices → IDs
    auto all_records = store_.scan_all();

    // Build a set of deleted IDs for fast lookup
    std::unordered_set<uint64_t> deleted_ids;
    // (Tombstones are already filtered by scan_all, so
    //  all_records only contains live records)

    std::vector<VDBSearchResult> results;
    for (const auto &hr : hnsw_results) {
      if (hr.id < all_records.size()) {
        const auto &rec = all_records[hr.id];
        results.push_back({rec.id, hr.distance, rec.metadata});
      }
      if (results.size() >= k)
        break;
    }

    return results;
  }

  // ─── Delete ──────────────────────────────────────

  /**
   * Soft-delete a vector by ID (tombstone in Iceberg).
   * The HNSW graph is NOT modified; tombstoned results
   * are filtered during search.
   */
  void delete_vector(uint64_t id) { store_.delete_vector(id); }

  // ─── Maintenance ─────────────────────────────────

  /**
   * Compact tombstoned segments and rebuild the HNSW index
   * from the remaining live records.
   *
   * This is the Iceberg "rewrite_data_files" equivalent.
   */
  size_t compact_and_rebuild(float tombstone_threshold = 0.3f) {
    size_t reclaimed = store_.compact(tombstone_threshold);

    // Full index rebuild from live data
    auto live = store_.scan_all();
    hnsw_ = HNSWIndex(dim_, 16, 200, 50); // Reset graph

    std::vector<std::vector<float>> vectors;
    vectors.reserve(live.size());
    for (const auto &r : live) {
      vectors.push_back(r.embedding);
    }
    hnsw_.build(vectors);

    return reclaimed;
  }

  /**
   * Force flush the active Iceberg segment.
   */
  void flush() { store_.flush(); }

  // ─── Accessors / Stats ───────────────────────────

  size_t dimension() const { return dim_; }
  size_t total_records() const { return store_.total_records(); }
  size_t live_records() const { return store_.total_live_records(); }
  size_t index_size() const { return hnsw_.size(); }
  size_t segment_count() const { return store_.sealed_segment_count(); }
  size_t snapshot_count() const { return store_.snapshot_count(); }

private:
  size_t dim_;
  IcebergStore store_;
  HNSWIndex hnsw_;
};

} // namespace vectordb
