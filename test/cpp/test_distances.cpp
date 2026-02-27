/**
 * Test suite for vector distance computations.
 *
 * Compile & run:
 *   g++ -std=c++17 -O2 -I../src/cpp -o test_distances
 * test/cpp/test_distances.cpp && ./test_distances
 */

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

// Forward declarations from distances.cpp (compiled inline for testing)
float l2_distance_naive(const float *x, const float *y, size_t d);

// Inline the naive implementation for self-contained test
float l2_distance_naive_test(const float *x, const float *y, size_t d) {
  float sum = 0.0f;
  for (size_t i = 0; i < d; ++i) {
    float diff = x[i] - y[i];
    sum += diff * diff;
  }
  return sum;
}

float cosine_sim(const float *x, const float *y, size_t d) {
  float dot = 0, nx = 0, ny = 0;
  for (size_t i = 0; i < d; ++i) {
    dot += x[i] * y[i];
    nx += x[i] * x[i];
    ny += y[i] * y[i];
  }
  float denom = std::sqrt(nx) * std::sqrt(ny);
  return (denom > 1e-10f) ? dot / denom : 0.0f;
}

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

void check_near(float a, float b, float tol, const std::string &name) {
  check(std::abs(a - b) < tol, name + " (got " + std::to_string(a) +
                                   ", expected " + std::to_string(b) + ")");
}

void test_l2_distance() {
  std::cout << "\n[test_l2_distance]" << std::endl;

  // Identical vectors → distance = 0
  float a[] = {1, 2, 3, 4};
  check_near(l2_distance_naive_test(a, a, 4), 0.0f, 1e-6f, "identical vectors");

  // Known distance: (1,0) to (0,1) → sqrt(2), squared = 2
  float x[] = {1.0f, 0.0f};
  float y[] = {0.0f, 1.0f};
  check_near(l2_distance_naive_test(x, y, 2), 2.0f, 1e-6f, "unit vectors L2^2");

  // 3-4-5 triangle
  float p[] = {0.0f, 0.0f};
  float q[] = {3.0f, 4.0f};
  check_near(l2_distance_naive_test(p, q, 2), 25.0f, 1e-6f,
             "3-4-5 triangle squared");

  // Zero vector
  float zero[] = {0, 0, 0};
  float v[] = {1, 1, 1};
  check_near(l2_distance_naive_test(zero, v, 3), 3.0f, 1e-6f, "from origin");
}

void test_cosine_similarity() {
  std::cout << "\n[test_cosine_similarity]" << std::endl;

  // Identical direction → 1.0
  float a[] = {1, 2, 3};
  check_near(cosine_sim(a, a, 3), 1.0f, 1e-6f, "identical vectors → 1.0");

  // Opposite → -1.0
  float b[] = {-1, -2, -3};
  check_near(cosine_sim(a, b, 3), -1.0f, 1e-6f, "opposite vectors → -1.0");

  // Orthogonal → 0.0
  float x[] = {1, 0};
  float y[] = {0, 1};
  check_near(cosine_sim(x, y, 2), 0.0f, 1e-6f, "orthogonal → 0.0");

  // Scaled vector → same similarity
  float c[] = {2, 4, 6};
  check_near(cosine_sim(a, c, 3), 1.0f, 1e-6f, "scaled → 1.0");
}

void test_symmetry() {
  std::cout << "\n[test_symmetry]" << std::endl;

  float x[] = {1.5f, -0.3f, 2.7f, 0.1f};
  float y[] = {-0.5f, 1.2f, 0.8f, 3.3f};

  check_near(l2_distance_naive_test(x, y, 4), l2_distance_naive_test(y, x, 4),
             1e-6f, "L2 symmetry");
  check_near(cosine_sim(x, y, 4), cosine_sim(y, x, 4), 1e-6f,
             "cosine symmetry");
}

void test_triangle_inequality() {
  std::cout << "\n[test_triangle_inequality]" << std::endl;

  float a[] = {0, 0};
  float b[] = {3, 0};
  float c[] = {3, 4};

  float d_ab = std::sqrt(l2_distance_naive_test(a, b, 2));
  float d_bc = std::sqrt(l2_distance_naive_test(b, c, 2));
  float d_ac = std::sqrt(l2_distance_naive_test(a, c, 2));

  check(d_ac <= d_ab + d_bc + 1e-6f,
        "triangle inequality: d(a,c) ≤ d(a,b) + d(b,c)");
}

void test_high_dimensional() {
  std::cout << "\n[test_high_dimensional]" << std::endl;

  const size_t d = 128;
  std::vector<float> x(d, 1.0f);
  std::vector<float> y(d, 0.0f);

  // L2^2 of (1,1,...,1) to origin = d
  check_near(l2_distance_naive_test(x.data(), y.data(), d),
             static_cast<float>(d), 1e-4f, "128-dim ones to zeros");

  // Cosine of parallel vectors = 1
  std::vector<float> z(d, 2.0f);
  check_near(cosine_sim(x.data(), z.data(), d), 1.0f, 1e-6f,
             "parallel 128-dim vectors");
}

int main() {
  std::cout << "=== Distance Metric Tests ===" << std::endl;

  test_l2_distance();
  test_cosine_similarity();
  test_symmetry();
  test_triangle_inequality();
  test_high_dimensional();

  std::cout << "\n--- Results: " << tests_passed << " passed, " << tests_failed
            << " failed ---" << std::endl;
  return tests_failed > 0 ? 1 : 0;
}
