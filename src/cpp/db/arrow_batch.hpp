/**
 * arrow_batch.hpp — ADBC-style Columnar Batch for Vector Ingestion
 *
 * This version uses the actual Apache Arrow C++ library APIs to build
 * and manage columnar layouts, as opposed to custom mock structs.
 *
 * In a production system, these memory structures can be passed with
 * zero-copy over the Arrow C Data Interface.
 */

#pragma once

#include <arrow/api.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/record_batch.h>
#include <arrow/type.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace vectordb {

/**
 * A builder wrapper for arrow::RecordBatch.
 *
 * This simplifies the process of appending entire vectors into columnar
 * chunks before passing the resulting RecordBatch to the DB.
 */
class RecordBatchBuilder {
public:
  RecordBatchBuilder() = default;

  /// Add an ID column (uint64).
  void add_id_column(const std::string &name,
                     const std::vector<uint64_t> &ids) {
    arrow::UInt64Builder builder;
    if (!builder.AppendValues(ids.data(), ids.size()).ok()) {
      throw std::runtime_error("Failed to append ID values");
    }
    std::shared_ptr<arrow::Array> array;
    if (!builder.Finish(&array).ok()) {
      throw std::runtime_error("Failed to finish ID array");
    }
    fields_.push_back(arrow::field(name, arrow::uint64()));
    arrays_.push_back(array);
  }

  /// Add a vector column (FixedSizeList of Float32).
  void add_vector_column(const std::string &name,
                         const std::vector<float> &flat_vectors, size_t dim) {
    if (dim == 0 || flat_vectors.size() % dim != 0) {
      throw std::invalid_argument(
          "flat_vectors.size() must be divisible by dim");
    }

    auto value_type = arrow::float32();
    auto list_type = arrow::fixed_size_list(value_type, dim);

    std::shared_ptr<arrow::FloatBuilder> value_builder =
        std::make_shared<arrow::FloatBuilder>();
    arrow::FixedSizeListBuilder list_builder(arrow::default_memory_pool(),
                                             value_builder, list_type);

    size_t num_vectors = flat_vectors.size() / dim;
    if (!list_builder.AppendValues(num_vectors).ok()) {
      throw std::runtime_error("Failed to append to FixedSizeListBuilder");
    }
    if (!value_builder->AppendValues(flat_vectors.data(), flat_vectors.size())
             .ok()) {
      throw std::runtime_error("Failed to append float values");
    }

    std::shared_ptr<arrow::Array> array;
    if (!list_builder.Finish(&array).ok()) {
      throw std::runtime_error("Failed to finish vector array");
    }

    fields_.push_back(arrow::field(name, list_type));
    arrays_.push_back(array);
  }

  /// Add a metadata/string column.
  void add_string_column(const std::string &name,
                         const std::vector<std::string> &strings) {
    arrow::StringBuilder builder;
    if (!builder.AppendValues(strings).ok()) {
      throw std::runtime_error("Failed to append string values");
    }
    std::shared_ptr<arrow::Array> array;
    if (!builder.Finish(&array).ok()) {
      throw std::runtime_error("Failed to finish string array");
    }
    fields_.push_back(arrow::field(name, arrow::utf8()));
    arrays_.push_back(array);
  }

  /// Convert to true arrow::RecordBatch
  std::shared_ptr<arrow::RecordBatch> build() const {
    auto schema = arrow::schema(fields_);
    size_t num_rows = 0;
    if (!arrays_.empty()) {
      num_rows = arrays_[0]->length();
    }
    return arrow::RecordBatch::Make(schema, num_rows, arrays_);
  }

private:
  std::vector<std::shared_ptr<arrow::Field>> fields_;
  std::vector<std::shared_ptr<arrow::Array>> arrays_;
};

} // namespace vectordb
