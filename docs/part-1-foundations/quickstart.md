---
title: "Chapter 0 — Build a Vector DB from Scratch"
description: A curated, step-by-step exercise to build a functional vector database from scratch in C++.
tags:
  - foundations
  - exercise
  - cpp
---

# 0. Build a Vector DB from Scratch (Curated Exercise)

Before diving into the high-dimensional math of Chapter 1, the best way to understand a vector database is to build one yourself. 

In this curated exercise, we will build a functional, in-memory Vector Database from scratch using **C++**. It will feature:
1. Dense vector storage
2. Brute-force Euclidean distance search
3. A basic metadata filtering system
4. A naive index to speed up the search

---

## Step 1: Defining the Data Structures

First, we need to define what a "Record" in our database looks like. A record consists of a unique ID, the high-dimensional float vector (embedding), and some basic metadata.

Create a file named `VectorDB.hpp`:

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <stdexcept>

// Let's assume we are working with 128-dimensional vectors
constexpr size_t DIMENSIONS = 128;

struct Record {
    size_t id;
    std::vector<float> vector;
    std::string metadata_category; // e.g., "science", "sports"
};

class MiniVectorDB {
private:
    // Core storage: Maps an ID to its full Record
    std::unordered_map<size_t, Record> storage;
    size_t next_id = 1;

public:
    MiniVectorDB() = default;

    // We will implement methods here
};
```

---

## Step 2: The Ingestion Pipeline

We need a method to insert vectors into our database. In a real database like Milvus, this involves Write-Ahead Logs (WAL) and memory buffers (see Chapter 6). For our mini-DB, we just insert it into our hash map.

```cpp
    // Add inside MiniVectorDB class:
    
    size_t insert(const std::vector<float>& vec, const std::string& category = "") {
        if (vec.size() != DIMENSIONS) {
            throw std::invalid_argument("Vector dimension mismatch");
        }
        
        size_t current_id = next_id++;
        storage[current_id] = {current_id, vec, category};
        
        return current_id;
    }
```

---

## Step 3: Exact Distance Math (L2)

To search, we need a mathematical way to compare two vectors. We will use the squared Euclidean (L2) distance. We use *squared* L2 because computing the square root (`std::sqrt`) is computationally expensive and unnecessary when we only care about rank ordering.

```cpp
    // Add inside MiniVectorDB class:

    static float calculate_l2_squared(const std::vector<float>& a, const std::vector<float>& b) {
        float distance = 0.0f;
        // Naive loop. In a production DB, this loop is replaced entirely 
        // with AVX2/AVX-512 SIMD instructions (See Chapter 11).
        for (size_t i = 0; i < DIMENSIONS; ++i) {
            float diff = a[i] - b[i];
            distance += diff * diff;
        }
        return distance;
    }
```

---

## Step 4: Brute-Force Exact Search

Now we build the simplest possible search engine: Flat Search (also known as k-Nearest Neighbors or k-NN). It literally loops through *every single vector* in the database, calculates the math, and returns the top `k` results.

```cpp
    // Add inside MiniVectorDB class:

    struct SearchResult {
        size_t id;
        float distance;
        bool operator<(const SearchResult& other) const {
            return distance < other.distance; // We want minimum distance
        }
    };

    std::vector<SearchResult> search(const std::vector<float>& query, int k) {
        std::vector<SearchResult> results;
        results.reserve(storage.size());

        // 1. Scan every vector in the DB (O(N * d) complexity)
        for (const auto& [id, record] : storage) {
            float dist = calculate_l2_squared(query, record.vector);
            results.push_back({id, dist});
        }

        // 2. Sort by distance
        std::sort(results.begin(), results.end());

        // 3. Keep only the Top-K
        if (results.size() > k) {
            results.resize(k);
        }

        return results;
    }
```

**Why this is bad:** If `storage` has 1 billion vectors, this single query performs 128 billion floating-point subtractions. It takes seconds or minutes. 

---

## Step 5: Implementing Hybrid Metadata Filtering

Users rarely just search for "similar vectors". They search for "similar vectors WHERE category = 'science'". Let's add Pre-Filtering to our brute-force search.

```cpp
    std::vector<SearchResult> search_with_filter(
        const std::vector<float>& query, 
        int k, 
        const std::string& filter_category) 
    {
        std::vector<SearchResult> results;

        for (const auto& [id, record] : storage) {
            // Pre-Filter step: skip vectors that don't match the metadata
            if (record.metadata_category != filter_category) {
                continue; 
            }
            
            float dist = calculate_l2_squared(query, record.vector);
            results.push_back({id, dist});
        }

        std::sort(results.begin(), results.end());
        if (results.size() > k) results.resize(k);

        return results;
    }
```

---

## Step 6: Upgrading to an Approximate Index (IVF Simulation)

Scanning everything is too slow. Let's implement a rudimentary **Inverted File Index (IVF)**. (Covered extensively in Chapter 2).

Instead of scanning all vectors, we will partition the vectors into two "buckets" based on their first dimension. If `vector[0] > 0`, they go to Bucket 1. Otherwise, Bucket 2. When searching, we only scan the bucket that the query vector belongs to.

*Note: Real databases use k-means clustering to define buckets, but this logic is identical.*

```cpp
class FastMiniVectorDB {
private:
    std::unordered_map<size_t, Record> storage;
    size_t next_id = 1;

    // Our naive index: Two buckets (Lists of IDs)
    std::vector<size_t> bucket_positive;
    std::vector<size_t> bucket_negative;

public:
    size_t insert(const std::vector<float>& vec) {
        size_t id = next_id++;
        storage[id] = {id, vec, ""};
        
        // Update the Index during ingestion
        if (vec[0] > 0.0f) {
            bucket_positive.push_back(id);
        } else {
            bucket_negative.push_back(id);
        }
        return id;
    }

    std::vector<SearchResult> fast_search(const std::vector<float>& query, int k) {
        std::vector<SearchResult> results;
        
        // 1. Ask the index which bucket to search (Routing)
        const std::vector<size_t>* target_bucket = &bucket_negative;
        if (query[0] > 0.0f) {
            target_bucket = &bucket_positive;
        }

        // 2. Only compute distance for vectors inside this specific bucket
        // We have successfully skipped ~50% of the database math!
        for (size_t id : *target_bucket) {
            const auto& record = storage[id];
            float dist = MiniVectorDB::calculate_l2_squared(query, record.vector);
            results.push_back({id, dist});
        }

        std::sort(results.begin(), results.end());
        if (results.size() > k) results.resize(k);
        return results;
    }
};
```

### The Concept of Recall
Because `fast_search` didn't check the other bucket, it's possible the true mathematically closest vector was sitting right on the boundary line in the other bucket. The database missed it. This introduces the concept of **Approximate Nearest Neighbors (ANN)** and the metric of **Recall**. Our query was 2x faster, but mathematically imperfect. 

---

## Conclusion

You have successfully written a vector database. You implemented:
1. Ingestion (`insert`)
2. Vector Math (`calculate_l2_squared`)
3. Flat Search (`search`)
4. Hybrid Search (`search_with_filter`)
5. Partitioned Indexing (`fast_search`)

Every production vector database works on exactly these principles, just scaled up:
* `calculate_l2_squared` is replaced with `AVX-512 FMA SIMD` instructions.
* The 2-bucket index is replaced with an `HNSW` multi-layer connection graph.
* The `unordered_map` storage is replaced with `Memory-Mapped Columnar` files.

Read on to Chapter 1 to understand the geometry that necessitates these massive scale-ups.
