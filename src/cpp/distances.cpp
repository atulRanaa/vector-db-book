/**
 * SIMD-optimized distance computations for vector search.
 *
 * Demonstrates how production vector databases accelerate
 * distance calculations using AVX2/AVX-512 intrinsics.
 *
 * Compile: g++ -O3 -mavx2 -mfma -o distances distances.cpp
 */

#include <cmath>
#include <cstddef>
#include <vector>
#include <algorithm>
#include <numeric>

// --8<-- [start:l2_naive]
/**
 * Naive L2 squared distance — scalar loop.
 * Complexity: O(d)
 */
float l2_distance_naive(const float* x, const float* y, size_t d) {
    float sum = 0.0f;
    for (size_t i = 0; i < d; ++i) {
        float diff = x[i] - y[i];
        sum += diff * diff;
    }
    return sum;
}
// --8<-- [end:l2_naive]


// --8<-- [start:l2_simd]
#ifdef __AVX2__
#include <immintrin.h>

/**
 * AVX2-optimized L2 squared distance.
 *
 * Processes 8 floats per iteration using 256-bit SIMD registers.
 * ~4-8x faster than scalar on modern CPUs.
 */
float l2_distance_avx2(const float* x, const float* y, size_t d) {
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;

    // Process 8 floats at a time
    for (; i + 8 <= d; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        __m256 diff = _mm256_sub_ps(vx, vy);
        sum = _mm256_fmadd_ps(diff, diff, sum);  // FMA: sum += diff * diff
    }

    // Horizontal sum of 8 floats in sum register
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float result = _mm_cvtss_f32(sum128);

    // Handle remaining elements
    for (; i < d; ++i) {
        float diff = x[i] - y[i];
        result += diff * diff;
    }

    return result;
}
#endif
// --8<-- [end:l2_simd]


// --8<-- [start:inner_product_simd]
#ifdef __AVX2__
/**
 * AVX2-optimized inner product.
 */
float inner_product_avx2(const float* x, const float* y, size_t d) {
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;

    for (; i + 8 <= d; i += 8) {
        __m256 vx = _mm256_loadu_ps(x + i);
        __m256 vy = _mm256_loadu_ps(y + i);
        sum = _mm256_fmadd_ps(vx, vy, sum);
    }

    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    float result = _mm_cvtss_f32(sum128);

    for (; i < d; ++i) {
        result += x[i] * y[i];
    }

    return result;
}
#endif
// --8<-- [end:inner_product_simd]


// --8<-- [start:brute_force_knn]
/**
 * Brute-force k-NN search — the baseline that all ANN
 * algorithms are compared against.
 *
 * Returns indices of k nearest vectors to query.
 */
struct SearchResult {
    size_t index;
    float distance;
    bool operator<(const SearchResult& other) const {
        return distance < other.distance;
    }
};

std::vector<SearchResult> brute_force_knn(
    const float* query,
    const float* database,  // row-major: n × d
    size_t n,
    size_t d,
    size_t k
) {
    std::vector<SearchResult> results(n);
    for (size_t i = 0; i < n; ++i) {
        results[i] = {i, l2_distance_naive(query, database + i * d, d)};
    }

    // Partial sort for top-k
    std::partial_sort(
        results.begin(),
        results.begin() + std::min(k, n),
        results.end()
    );

    results.resize(std::min(k, n));
    return results;
}
// --8<-- [end:brute_force_knn]
