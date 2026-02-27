/**
 * iceberg_store.hpp — Apache Iceberg-style Segment Storage
 *
 * This version uses actual Apache Parquet files generated via Arrow,
 * simulating an Iceberg object storage layer. Live (mutable) segments
 * are held in memory until flushed. Once flushed, the segments become
 * immutable Parquet files. Tombstones are held in memory.
 */

#pragma once

#include "arrow_batch.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <parquet/exception.h>

namespace vectordb {

// ─────────────────────────────────────────────────────
// VectorRecord: A single row in the table.
// ─────────────────────────────────────────────────────

struct VectorRecord {
  uint64_t id;
  std::vector<float> embedding;
  std::string metadata;
};

// ─────────────────────────────────────────────────────
// Segment: Mutable Data
// ─────────────────────────────────────────────────────

struct Segment {
  int segment_id;
  std::vector<VectorRecord> records;    // The actual data
  std::unordered_set<uint64_t> deleted; // Tombstone set

  size_t live_count() const { return records.size() - deleted.size(); }
};

// ─────────────────────────────────────────────────────
// SealedSegment: Immutable Parquet file metadata
// ─────────────────────────────────────────────────────

struct SealedSegment {
  int segment_id;
  std::string filepath;
  size_t num_records;
  std::unordered_set<uint64_t> deleted;

  size_t live_count() const { return num_records - deleted.size(); }
  float tombstone_ratio() const {
    if (num_records == 0)
      return 0.0f;
    return static_cast<float>(deleted.size()) / static_cast<float>(num_records);
  }
};

// ─────────────────────────────────────────────────────
// Snapshot: An Iceberg table snapshot (manifest list)
// ─────────────────────────────────────────────────────

struct Snapshot {
  int snapshot_id;
  int64_t timestamp_ms;
  std::vector<int> segment_ids; // Which segments are live
};

// ─────────────────────────────────────────────────────
// IcebergStore: The Table-Format Storage Manager
// ─────────────────────────────────────────────────────

class IcebergStore {
public:
  explicit IcebergStore(size_t dim, size_t segment_capacity = 1000,
                        const std::string &data_dir = "/tmp/vectordb")
      : dim_(dim), segment_capacity_(segment_capacity), data_dir_(data_dir),
        next_segment_id_(0) {
    if (!std::filesystem::exists(data_dir_)) {
      std::filesystem::create_directories(data_dir_);
    }

    // Start with an empty active segment
    active_segment_.segment_id = next_segment_id_++;
    commit_snapshot();
  }

  // ─── Write Path ──────────────────────────────────

  void insert(uint64_t id, const std::vector<float> &embedding,
              const std::string &metadata = "") {
    std::lock_guard<std::mutex> lock(mu_);
    if (embedding.size() != dim_)
      throw std::invalid_argument("Dimension mismatch");

    active_segment_.records.push_back({id, embedding, metadata});

    if (active_segment_.records.size() >= segment_capacity_) {
      flush_active_segment_locked();
    }
  }

  void bulk_insert(const uint64_t *ids, const float *vectors, size_t count,
                   size_t dim) {
    std::lock_guard<std::mutex> lock(mu_);
    if (dim != dim_)
      throw std::invalid_argument("Dimension mismatch in bulk insert");

    for (size_t i = 0; i < count; ++i) {
      std::vector<float> emb(vectors + i * dim, vectors + (i + 1) * dim);
      active_segment_.records.push_back({ids[i], std::move(emb), ""});

      if (active_segment_.records.size() >= segment_capacity_) {
        flush_active_segment_locked();
      }
    }
  }

  void delete_vector(uint64_t id) {
    std::lock_guard<std::mutex> lock(mu_);

    for (auto &r : active_segment_.records) {
      if (r.id == id) {
        active_segment_.deleted.insert(id);
        return;
      }
    }
    for (auto &seg : sealed_segments_) {
      // In a real Iceberg system, we'd use a Bloom filter or search the files.
      // Here we assume any soft-delete might apply to a sealed segment.
      // But we can't easily check Parquet here without scanning it.
      // We'll just append it to all sealed_segments' tombstone sets.
      // This is a naive approximation of positional/equality deletes.
      seg.deleted.insert(id);
    }
  }

  void flush() {
    std::lock_guard<std::mutex> lock(mu_);
    flush_active_segment_locked();
  }

  // ─── Read Path ───────────────────────────────────

  std::vector<VectorRecord> scan_all() const {
    std::lock_guard<std::mutex> lock(mu_);
    std::vector<VectorRecord> result;

    // Scan sealed segments from Parquet files
    for (const auto &seg : sealed_segments_) {
      auto records = read_parquet(seg.filepath, seg.deleted);
      result.insert(result.end(), std::make_move_iterator(records.begin()),
                    std::make_move_iterator(records.end()));
    }

    // Scan active segment
    for (const auto &r : active_segment_.records) {
      if (active_segment_.deleted.find(r.id) == active_segment_.deleted.end()) {
        result.push_back(r);
      }
    }

    return result;
  }

  // ─── Compaction ──────────────────────────────────

  size_t compact(float tombstone_threshold = 0.3f) {
    std::lock_guard<std::mutex> lock(mu_);

    std::vector<SealedSegment> clean;
    std::vector<VectorRecord> to_merge;
    size_t reclaimed = 0;

    for (auto &seg : sealed_segments_) {
      if (seg.tombstone_ratio() >= tombstone_threshold) {
        // Read live records from Parquet
        auto records = read_parquet(seg.filepath, seg.deleted);
        size_t initial_count = seg.num_records;
        reclaimed += (initial_count - records.size());
        to_merge.insert(to_merge.end(),
                        std::make_move_iterator(records.begin()),
                        std::make_move_iterator(records.end()));

        // Optionally delete the old file
        std::filesystem::remove(seg.filepath);
      } else {
        clean.push_back(std::move(seg));
      }
    }

    if (!to_merge.empty()) {
      int new_seg_id = next_segment_id_++;
      std::string path =
          data_dir_ + "/segment_" + std::to_string(new_seg_id) + ".parquet";
      write_parquet(path, to_merge);

      SealedSegment merged;
      merged.segment_id = new_seg_id;
      merged.filepath = path;
      merged.num_records = to_merge.size();
      clean.push_back(std::move(merged));
    }

    sealed_segments_ = std::move(clean);
    commit_snapshot();
    return reclaimed;
  }

  // ─── Stats ───────────────────────────────────────

  size_t snapshot_count() const { return snapshots_.size(); }
  const Snapshot &get_snapshot(size_t idx) const { return snapshots_.at(idx); }

  size_t total_records() const {
    std::lock_guard<std::mutex> lock(mu_);
    size_t total = active_segment_.records.size();
    for (const auto &seg : sealed_segments_)
      total += seg.num_records;
    return total;
  }

  size_t total_live_records() const {
    std::lock_guard<std::mutex> lock(mu_);
    size_t total = active_segment_.live_count();
    for (const auto &seg : sealed_segments_)
      total += seg.live_count();
    return total;
  }

  size_t sealed_segment_count() const {
    std::lock_guard<std::mutex> lock(mu_);
    return sealed_segments_.size();
  }

  size_t dimension() const { return dim_; }

private:
  void flush_active_segment_locked() {
    if (active_segment_.records.empty())
      return;

    std::string path = data_dir_ + "/segment_" +
                       std::to_string(active_segment_.segment_id) + ".parquet";
    write_parquet(path, active_segment_.records);

    SealedSegment sealed;
    sealed.segment_id = active_segment_.segment_id;
    sealed.filepath = path;
    sealed.num_records = active_segment_.records.size();
    sealed.deleted = std::move(active_segment_.deleted);
    sealed_segments_.push_back(std::move(sealed));

    active_segment_ = Segment{};
    active_segment_.segment_id = next_segment_id_++;

    commit_snapshot();
  }

  void write_parquet(const std::string &path,
                     const std::vector<VectorRecord> &records) const {
    RecordBatchBuilder builder;
    std::vector<uint64_t> ids;
    std::vector<float> flat_vectors;
    std::vector<std::string> metas;

    ids.reserve(records.size());
    flat_vectors.reserve(records.size() * dim_);
    metas.reserve(records.size());

    for (const auto &r : records) {
      ids.push_back(r.id);
      flat_vectors.insert(flat_vectors.end(), r.embedding.begin(),
                          r.embedding.end());
      metas.push_back(r.metadata);
    }

    builder.add_id_column("id", ids);
    builder.add_vector_column("embedding", flat_vectors, dim_);
    builder.add_string_column("metadata", metas);

    auto batch = builder.build();
    auto table = arrow::Table::FromRecordBatches({batch}).ValueOrDie();

    std::shared_ptr<arrow::io::FileOutputStream> outfile;
    PARQUET_ASSIGN_OR_THROW(outfile, arrow::io::FileOutputStream::Open(path));
    PARQUET_THROW_NOT_OK(parquet::arrow::WriteTable(
        *table, arrow::default_memory_pool(), outfile, 1024));
  }

  std::vector<VectorRecord>
  read_parquet(const std::string &path,
               const std::unordered_set<uint64_t> &deleted) const {
    std::vector<VectorRecord> result;

    std::shared_ptr<arrow::io::ReadableFile> infile;
    PARQUET_ASSIGN_OR_THROW(infile, arrow::io::ReadableFile::Open(path));

    std::unique_ptr<parquet::arrow::FileReader> reader;
    PARQUET_ASSIGN_OR_THROW(
        reader, parquet::arrow::OpenFile(infile, arrow::default_memory_pool()));

    std::shared_ptr<arrow::Table> table;
    PARQUET_THROW_NOT_OK(reader->ReadTable(&table));

    // Combine chunks so we only deal with one dense Array
    auto combined_table = table->CombineChunksToBatch().ValueOrDie();
    if (!combined_table)
      return result;

    auto id_col = std::static_pointer_cast<arrow::UInt64Array>(
        combined_table->GetColumnByName("id"));
    auto vec_col = std::static_pointer_cast<arrow::FixedSizeListArray>(
        combined_table->GetColumnByName("embedding"));
    auto meta_col = std::static_pointer_cast<arrow::StringArray>(
        combined_table->GetColumnByName("metadata"));
    auto floats =
        std::static_pointer_cast<arrow::FloatArray>(vec_col->values());

    for (int64_t i = 0; i < combined_table->num_rows(); ++i) {
      uint64_t id = id_col->Value(i);
      if (deleted.find(id) == deleted.end()) {
        std::vector<float> vec(floats->raw_values() + i * dim_,
                               floats->raw_values() + (i + 1) * dim_);
        std::string meta = meta_col->GetString(i);
        result.push_back({id, std::move(vec), std::move(meta)});
      }
    }
    return result;
  }

  void commit_snapshot() {
    Snapshot snap;
    snap.snapshot_id = static_cast<int>(snapshots_.size());
    snap.timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();

    for (const auto &seg : sealed_segments_) {
      snap.segment_ids.push_back(seg.segment_id);
    }
    snapshots_.push_back(snap);
  }

  size_t dim_;
  size_t segment_capacity_;
  std::string data_dir_;
  int next_segment_id_;

  Segment active_segment_;
  std::vector<SealedSegment> sealed_segments_;
  std::vector<Snapshot> snapshots_;

  mutable std::mutex mu_;
};

} // namespace vectordb
