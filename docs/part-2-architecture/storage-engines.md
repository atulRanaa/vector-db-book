---
title: "Chapter 7 — Storage Engines"
description: An exploration of the physical storage formats underlying vector databases, comparing memory-mapped files, LSM-trees, and columnar layouts.
tags:
  - architecture
  - storage
  - IO
---

# 7. Storage Engines

In a vector database, the HNSW graph acts as the brain (the index routing map), but the **Storage Engine** is the muscle. It manages how the actual multi-gigabyte files of floats and metadata are written to and fetched from physical NVMe Solid State Drives (SSDs). 

Operating systems are incredibly complex when it comes to disk I/O. If a storage engine performs 100,000 tiny microscopic reads from random sectors of a hard drive every second, the OS kernel will lock up. This chapter explains how vector databases design their internal file structures to bypass these bottlenecks.

---

## 7.1 Memory-Mapping (mmap)

The vast majority of modern vector databases (including Qdrant and early Milvus) heavily utilize `mmap`, a POSIX system call that maps the bytes of a file sitting on an SSD directly into the memory space of the application.

### How it works
Normally, if a database wants to read vector ID #500 from a file, it asks the OS to read `file.bin`, the OS pulls the data from disk into Kernel RAM, and then copies it into the Database RAM. 
With `mmap`, the database completely bypasses manual reads. It just treats the `file.bin` on the hard drive as if it were a massive C++ array in RAM. When the software asks for `array[500]`, the OS silently intercepts the request, runs to the hard drive, pulls the block of data into a cache (Page Cache), and serves it.

**Why `mmap` is dominant in Vector DBs:**
1. **Zero manual memory management:** The OS decides what parts of the HNSW graph belong in fast RAM and what parts can be evicted to slow SSD based on exactly what vectors users have been querying recently.
2. **Crash resilience:** If the database process crashes, the data isn't corrupted; the OS just stops mapping it.

### The Limits of mmap
`mmap` is highly optimized for **read-heavy** workloads. However, if the database is constantly being written to (millions of inserts a second), `mmap` suffers severe performance degradation. For heavy ingest, we need specialized write architectures.

---

## 7.2 Log-Structured Merge (LSM) Trees

Relational databases traditionally use B-Trees (where you overwrite data exactly where it lives on disk). Because SSDs physically degrade when you overwrite the exact same sectors repeatedly, and random writes are slow, modern NoSQL and Vector databases (like Milvus and Pinecone) adapt **LSM-Trees** (Log-Structured Merge-Trees) for their scalar data and segment logistics.

### The Write Path (Append-Only)
In an LSM architecture, data is **never** updated in place. When a new vector is inserted, it is simply appended to the very end of an active file. Appending to the end of a file is the absolute fastest operation a hard drive can perform. 

```mermaid
flowchart TD
    subgraph RAM [Fast Memory]
        MemTable["MemTable<br/>Active In-Memory Segment"]
    end

    subgraph SSD [NVMe Disk Storage]
        direction TB
        L0["Level 0 Files<br/>Small, Fresh Flushes"]
        L1["Level 1 Files<br/>Medium Merged Segments"]
        L2["Level 2 Files<br/>Massive Historical Segments"]
    end

    Client[Client Insert] --> |Append to| MemTable
    MemTable -. "When Full (e.g., 64MB)" .-> |Flush to Disk| L0
    L0 -. "Background Compaction thread" .-> L1
    L1 -. "Background Compaction thread" .-> L2
    
    style RAM fill:#e8f5e9,stroke:#4caf50
    style SSD fill:#eceff1,stroke:#607d8b
```

### Compaction: The Necessary Evil
If you just kept appending vectors forever, you'd end up with 100,000 tiny files. Your read latency would skyrocket because you'd have to search every single file.
To fix this, a background thread constantly runs **Compaction**. It grabs ten small 60MB segments (Level 0 files), sorts them, merges them together, mathematically rebuilds a new HNSW graph for the combined dataset, and saves it as a brand new 600MB Level 1 segment. This uses heavy CPU, but keeps the disk organized for fast reads.

---

## 7.3 Columnar Layouts for Vector Math

If you read an entire record (Vector + Metadata) from disk sequentially:

```json
{
  "id": 1, 
  "vector": [0.1, 0.4, ...], 
  "author": "Alice", 
  "date": "2024-01-01"
}
```
This is a **Row-based** storage format. 

If your query engine needs to execute math on the vectors, it has to load the `author` and `date` strings into the CPU cache alongside the floats. This pollutes the microscopic incredibly fast L1 CPU caches with garbage text data that the math engine doesn't need, slowing down AVX operations.

Databases like **LanceDB** use **Columnar Storage**. They physically store all vectors clustered together in one file block, all authors in another block, and all dates in another. During a pure math search, the CPU simply blasts through the tightly-packed block of contiguous vectors with zero cache pollution.

---

# Assignment: Build a Segmented File Store Layer

In Chapter 6, you built a WAL to protect an isolated, in-memory hash map. In reality, memory maps cannot grow forever. In this assignment, you will implement an LSM-style `flush` mechanism that "Seals" a segment and writes it to an immutable binary file.

### Goal
Modify your `MiniVectorDB` to flush vectors from active memory into a `.segment` binary file when the RAM reaches a certain target threshold.

```cpp title="Exercise: Implement Segment Flushing"
#include <fstream>
#include <iostream>
#include <vector>

constexpr size_t THRESHOLD = 1000; // Vectors per segment

class SegmentManager {
private:
    std::unordered_map<size_t, Record> active_memory;
    int segment_counter = 0;

public:
    void insert(size_t id, const Record& r) {
        active_memory[id] = r;
        
        // 1. Check if we breached the RAM threshold
        if (active_memory.size() >= THRESHOLD) {
            flush_to_disk();
        }
    }

    // YOUR EXERCISE: Implement this function!
    void flush_to_disk() {
        std::string filename = "segment_" + std::to_string(segment_counter++) + ".bin";
        std::ofstream outfile(filename, std::ios::binary);

        // 2. Iterate through `active_memory`
        // 3. Serialize the `id` and `vector` bytes into the binary file
        // 4. Clear `active_memory` so RAM usage drops back to 0.
        
        /* Write your serialization code here */

        outfile.close();
        std::cout << "Sealed Segment: " << filename << std::endl;
    }

    // Further thinking: Now that data is on disk, how does your `search()`
    // function need to change? (Hint: mmap or file streams).
};
```

---

## References

1. O'Neil, P., et al. (1996). *The Log-Structured Merge-Tree (LSM-Tree)*. Acta Informatica.
2. LanceDB. *Lance Columnar Format Overview*.
