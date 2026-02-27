---
title: "Chapter 23 — Real-World Case Studies"
description: How Spotify, Pinterest, and OpenAI use vector search at scale, including experimentation code snippets in C++ and Rust.
tags:
  - advanced
  - case-studies
---

# 23. Real-World Case Studies

Vector databases are not just theoretical math exercises; they power the most critical features of the modern internet. This chapter profiles how three massive tech companies use vector search in production, and provides code snippets imitating their strategies so you can experiment with the concepts natively.

---

## 23.1 Spotify: Real-Time Audio Recommendations

### The Problem
Spotify has over 100 million tracks. When a user finishes listening to a heavy metal song, Spotify must instantly recommend the next song to play. They cannot use metadata alone (e.g., matching the genre tag `metal`); they want to recommend songs with similar acoustic profiles, tempos, and "vibes".

### The Solution: Annoy & Audio Embeddings
Spotify was one of the early pioneers of semantic vector search. They convert raw audio waveforms into dense embeddings via neural networks. To search the 100-million track database in under 50 milliseconds, they open-sourced a library called **Annoy** (Approximate Nearest Neighbors Oh Yeah).

Annoy uses **Random Projection Trees** (a variant of the KD-Trees discussed in Chapter 2) rather than HNSW. It recursively splits the data using random hyperplanes, creating a forest of trees that are heavily optimized for memory-mapped read-only sharing across multiple processes.

### Experiment: Random Projection Trees in C++

Here is an experimental snippet demonstrating how a single Split Node in an Annoy-style Random Projection Tree is structured. *Try compiling this and experimenting with how the `margin` determines which branch a vector takes.*

```cpp
#include <iostream>
#include <vector>
#include <numeric>

// An Annoy-style Random Projection Node
struct Node {
    std::vector<float> split_plane_normal;
    float split_bias;
    int left_child_index;
    int right_child_index;
};

// Computes dot product to determine side of the hyperplane
float get_margin(const Node& node, const std::vector<float>& item) {
    float margin = node.split_bias;
    for (size_t i = 0; i < item.size(); ++i) {
        margin += node.split_plane_normal[i] * item[i];
    }
    return margin;
}

int route_to_child(const Node& node, const std::vector<float>& query) {
    float margin = get_margin(node, query);
    // If margin is positive, go left. If negative, go right.
    if (margin > 0) return node.left_child_index;
    return node.right_child_index;
}

int main() {
    // 3-dimensional toy embedding of an audio track
    std::vector<float> audio_query = {0.8f, -0.4f, 1.2f}; 

    // A predefined split node
    Node root = {{1.0f, -1.0f, 0.5f}, 0.0f, 1, 2};

    std::cout << "Routing audio track to child node ID: " 
              << route_to_child(root, audio_query) << std::endl;
    // Computes: (1.0*0.8) + (-1.0*-0.4) + (0.5*1.2) = 0.8 + 0.4 + 0.6 = 1.8. 
    // Margin is positive, returns left child (1).
    return 0;
}
```

---

## 23.2 Pinterest: Visual Search & "Shop the Look"

### The Problem
When a Pinterest user sees a photo of a living room and clicks on a specific armchair in the picture, Pinterest must instantly scan billions of products across the internet to find visually similar armchairs to sell to the user. Keywords are useless here; the user doesn't know the designer's name or color code.

### The Solution: Multi-Modal HNSW
Pinterest generates embeddings of images using Convolutional Neural Networks (CNNs) and transformer models. They heavily rely on **HNSW** indexes for search. 
Crucially, Pinterest uses **Multi-Modal embeddings**. If a user types "mid-century leather chair" (text), the text is converted into the *exact same embedding space* as the images of the chairs. The database doesn't care if the query is text or a cropped image; it's just a 256-dimensional float vector.

### Experiment: HNSW Graph Initialization in Rust

Here is an experimental snippet imitating the core layer structure of Pinterest's HNSW implementation in Rust. *Notice how vectors are assigned a maximum layer based on a probability distribution.*

```rust
use rand::Rng;

const M: usize = 16;       // Max connections per node
const M_MAX: usize = 32;   // Max connections at Layer 0

struct HnswNode {
    id: usize,
    vector: Vec<f32>,
    max_layer: usize,
    // connections[layer][neighbour_index]
    connections: Vec<Vec<usize>>, 
}

fn assign_random_layer(mult: f64) -> usize {
    let mut rng = rand::thread_rng();
    let uniform: f64 = rng.gen_range(0.0001..1.0);
    // Exponential decay probability: 
    // Almost everyone gets layer 0. Very few get layer 4.
    let layer = (-uniform.ln() * mult).floor() as usize;
    layer
}

fn main() {
    // Multiplier often set to 1 / ln(M)
    let m_l = 1.0 / (16.0_f64).ln(); 
    let node_layer = assign_random_layer(m_l);
    
    let mut node = HnswNode {
        id: 1042,
        vector: vec![0.11, -0.9, 0.88], // "leather chair" embedding
        max_layer: node_layer,
        connections: vec![Vec::new(); node_layer + 1],
    };
    
    println!("Node {} injected at maximum layer: {}", node.id, node.max_layer);
}
```

---

## 23.3 OpenAI: Large-Scale RAG Tool Retrieval

### The Problem
When ChatGPT searches the web or accesses internal enterprise documents, it cannot stuff millions of PDF pages into its context window. It must isolate the 10 most highly relevant paragraphs to read before generating an answer.

### The Solution: Chunking and the RAG Pipeline
OpenAI relies heavily on **Retrieval-Augmented Generation (RAG)**. They do not embed entire books at once. Instead, they use a "Chunking" strategy:
1. Split the massive document into 300-token chunks.
2. Embed each chunk using the `text-embedding-3-small` model.
3. Store the chunk embeddings in a vector database (like Pinecone or Milvus).
4. When the user asks a question, embed the question, find the Top-5 closest chunks, and inject those chunks seamlessly into the LLM prompt.

### Experiment: RAG Semantic Chunk Scorer in C++

Here is an experimental snippet representing the final step of a RAG pipeline: after retrieving chunks from the Vector DB, we must rank them by relevance (cosine similarity) before passing them to the LLM. *Try adding new chunks to see how the LLM decides which paragraphs to "read".*

```cpp
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

// Calculate Cosine Similarity
float cosine_sim(const std::vector<float>& a, const std::vector<float>& b) {
    float dot = 0.0f, normA = 0.0f, normB = 0.0f;
    for(size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    return dot / (std::sqrt(normA) * std::sqrt(normB));
}

struct DocumentChunk {
    std::string text;
    std::vector<float> embedding;
    float relevance_score = 0.0f;
};

int main() {
    // User query: "How do black holes form?"
    std::vector<float> query_emb = {0.9f, -0.2f, 0.4f, 0.8f};

    // Chunks retrieved from the Vector DB
    std::vector<DocumentChunk> retrieved_chunks = {
        {"Astronomers observe supernova explosions...", {0.8f, -0.1f, 0.3f, 0.9f}},
        {"A recipe for black forest cake...",           {-0.5f, 0.8f, 0.1f, -0.2f}},
        {"Gravity condenses a dying star core...",      {0.9f, -0.3f, 0.5f, 0.7f}}
    };

    // Re-rank for the LLM Context Window
    for (auto& chunk : retrieved_chunks) {
        chunk.relevance_score = cosine_sim(query_emb, chunk.embedding);
    }
    
    std::sort(retrieved_chunks.begin(), retrieved_chunks.end(), 
        [](const DocumentChunk& a, const DocumentChunk& b) {
            return a.relevance_score > b.relevance_score;
        }
    );

    std::cout << "--- LLM Context Window Injection ---" << std::endl;
    for (const auto& chunk : retrieved_chunks) {
        std::cout << "Score [" << chunk.relevance_score << "] : " << chunk.text << std::endl;
    }

    return 0;
}
```
