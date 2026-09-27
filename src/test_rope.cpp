// Checkpoint-independent tests for the CPU RoPE implementation.
// These check the mathematics of the rotation, not just self-consistency.

#include "cpu_ops.hpp"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

constexpr float kTolerance = 1e-5f;

int failures = 0;

void expect_near(const char* label, float actual, float expected)
{
    if (!std::isfinite(actual) || std::abs(actual - expected) > kTolerance) {
        std::cout << "  FAIL " << label << ": expected " << expected
                  << ", got " << actual << '\n';
        failures++;
    }
}

// (3, 4) rotated by 90 degrees (cos 0, sin 1) must become (-4, 3).
void test_quarter_turn()
{
    std::cout << "quarter turn\n";
    float v[2] = {3.0f, 4.0f};
    const float c[1] = {0.0f};
    const float s[1] = {1.0f};
    rope(v, 1, 2, c, s);
    expect_near("a", v[0], -4.0f);
    expect_near("b", v[1], 3.0f);
}

// cos 1, sin 0 is the identity rotation.
void test_identity()
{
    std::cout << "identity\n";
    std::vector<float> v = {0.5f, -1.25f, 2.0f, 3.75f, -0.125f, 9.0f};
    const std::vector<float> original = v;
    const float c[3] = {1.0f, 1.0f, 1.0f};
    const float s[3] = {0.0f, 0.0f, 0.0f};
    rope(v.data(), 1, 6, c, s);
    for (std::size_t i = 0; i < v.size(); i++) {
        expect_near("element", v[i], original[i]);
    }
}

// A rotation never changes the length of the vector it rotates, so
// a^2 + b^2 must be unchanged for every pair.
void test_norm_preserved()
{
    std::cout << "norm preservation\n";
    constexpr int n_heads = 2;
    constexpr int head_size = 8;
    constexpr int half = head_size / 2;

    std::vector<float> v(n_heads * head_size);
    for (std::size_t i = 0; i < v.size(); i++) {
        v[i] = 0.37f * static_cast<float>(i) - 2.1f;
    }
    const std::vector<float> original = v;

    float c[half], s[half];
    for (int p = 0; p < half; p++) {
        const float angle = 0.7f * static_cast<float>(p + 1);
        c[p] = std::cos(angle);
        s[p] = std::sin(angle);
    }

    rope(v.data(), n_heads, head_size, c, s);

    for (int i = 0; i < n_heads * head_size; i += 2) {
        const float before = original[i] * original[i] + original[i + 1] * original[i + 1];
        const float after = v[i] * v[i] + v[i + 1] * v[i + 1];
        expect_near("pair norm", after, before);
    }
}

// Each pair index has its own angle, and that per-pair pattern must be
// applied identically to every head.  Give every head the same input and
// check every head against a hand-computed result.
void test_pattern_repeats_across_heads()
{
    std::cout << "pattern repeats across heads\n";
    constexpr int n_heads = 3;
    constexpr int head_size = 4;   // two pairs per head

    const float c[2] = {0.0f, 1.0f};   // pair 0: quarter turn, pair 1: identity
    const float s[2] = {1.0f, 0.0f};

    std::vector<float> v;
    for (int h = 0; h < n_heads; h++) {
        v.insert(v.end(), {1.0f, 2.0f, 5.0f, 6.0f});
    }

    rope(v.data(), n_heads, head_size, c, s);

    for (int h = 0; h < n_heads; h++) {
        const float* head = v.data() + h * head_size;
        expect_near("pair0 a", head[0], -2.0f);
        expect_near("pair0 b", head[1], 1.0f);
        expect_near("pair1 a", head[2], 5.0f);
        expect_near("pair1 b", head[3], 6.0f);
    }
}

// Q has more heads than K (grouped-query attention).  Rotating K with
// n_kv_heads must produce exactly what the first n_kv_heads of Q produce,
// and must not touch anything past K's end.
void test_fewer_kv_heads()
{
    std::cout << "fewer kv heads than q heads\n";
    constexpr int n_heads = 4;
    constexpr int n_kv_heads = 2;
    constexpr int head_size = 6;
    constexpr int half = head_size / 2;

    std::vector<float> q(n_heads * head_size);
    for (std::size_t i = 0; i < q.size(); i++) {
        q[i] = static_cast<float>((i * 7) % 11) - 5.0f;
    }

    // k = first n_kv_heads of q, plus a sentinel past the end.
    std::vector<float> k(q.begin(), q.begin() + n_kv_heads * head_size);
    k.push_back(123.0f);

    float c[half], s[half];
    for (int p = 0; p < half; p++) {
        c[p] = std::cos(0.3f * static_cast<float>(p + 1));
        s[p] = std::sin(0.3f * static_cast<float>(p + 1));
    }

    rope(q.data(), n_heads, head_size, c, s);
    rope(k.data(), n_kv_heads, head_size, c, s);

    for (int i = 0; i < n_kv_heads * head_size; i++) {
        expect_near("k vs q", k[i], q[i]);
    }
    expect_near("sentinel", k.back(), 123.0f);
}

// 7 heads x 37 pairs = 259 pairs, which is not a multiple of the CUDA
// block size.  On the CPU this simply checks an odd shape end to end.
void test_awkward_pair_count()
{
    std::cout << "awkward pair count\n";
    constexpr int n_heads = 7;
    constexpr int head_size = 74;
    constexpr int half = head_size / 2;

    std::vector<float> v(n_heads * head_size);
    for (std::size_t i = 0; i < v.size(); i++) {
        v[i] = 0.01f * static_cast<float>(i);
    }
    const std::vector<float> original = v;

    float c[half], s[half];
    for (int p = 0; p < half; p++) {
        c[p] = std::cos(0.05f * static_cast<float>(p));
        s[p] = std::sin(0.05f * static_cast<float>(p));
    }

    rope(v.data(), n_heads, head_size, c, s);

    for (int h = 0; h < n_heads; h++) {
        for (int p = 0; p < half; p++) {
            const int i = h * head_size + 2 * p;
            const float a = original[i];
            const float b = original[i + 1];
            expect_near("a", v[i], a * c[p] - b * s[p]);
            expect_near("b", v[i + 1], a * s[p] + b * c[p]);
        }
    }
}

} // namespace

int main()
{
    test_quarter_turn();
    test_identity();
    test_norm_preserved();
    test_pattern_repeats_across_heads();
    test_fewer_kv_heads();
    test_awkward_pair_count();

    if (failures == 0) {
        std::cout << "All Passed\n";
        return 0;
    }
    std::cout << failures << " failure(s)\n";
    return 1;
}
