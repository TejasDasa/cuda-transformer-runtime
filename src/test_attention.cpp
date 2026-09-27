// Checkpoint-independent tests for the CPU KV cache, softmax and
// single-token causal attention.  Expected values are hand-derived or
// computed by an independent double-precision reference, not copied from
// the implementation under test.

#include "attention_shape.hpp"
#include "cpu_ops.hpp"
#include "kv_cache.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <vector>

namespace {

constexpr float kTolerance = 1e-5f;

int failures = 0;

void expect_near(const char* label, double actual, double expected, double tol = kTolerance)
{
    if (!std::isfinite(actual) || std::abs(actual - expected) > tol) {
        std::cout << "  FAIL " << label << ": expected " << expected
                  << ", got " << actual << '\n';
        failures++;
    }
}

void expect_true(const char* label, bool condition)
{
    if (!condition) {
        std::cout << "  FAIL " << label << '\n';
        failures++;
    }
}

// Naive double-precision attention written independently of cpu_ops.
std::vector<double> reference_attention(
    const std::vector<float>& q,
    const std::vector<float>& k_cache,
    const std::vector<float>& v_cache,
    int n_heads, int n_kv_heads, int head_size, int pos
)
{
    const int kv_dim = n_kv_heads * head_size;
    std::vector<double> out(static_cast<std::size_t>(n_heads) * head_size, 0.0);
    for (int h = 0; h < n_heads; h++) {
        const int kv = h / (n_heads / n_kv_heads);
        std::vector<double> s(pos + 1);
        double m = -std::numeric_limits<double>::infinity();
        for (int t = 0; t <= pos; t++) {
            double dot = 0;
            for (int d = 0; d < head_size; d++) {
                dot += static_cast<double>(q[h * head_size + d]) *
                       k_cache[static_cast<std::size_t>(t) * kv_dim + kv * head_size + d];
            }
            s[t] = dot / std::sqrt(static_cast<double>(head_size));
            m = std::max(m, s[t]);
        }
        double sum = 0;
        for (int t = 0; t <= pos; t++) { s[t] = std::exp(s[t] - m); sum += s[t]; }
        for (int t = 0; t <= pos; t++) {
            for (int d = 0; d < head_size; d++) {
                out[h * head_size + d] += (s[t] / sum) *
                    v_cache[static_cast<std::size_t>(t) * kv_dim + kv * head_size + d];
            }
        }
    }
    return out;
}

// Deterministic pseudo-random filler in [-1, 1).
float filler(std::size_t i)
{
    return static_cast<float>((i * 2654435761u) % 2000) / 1000.0f - 1.0f;
}

struct Buffers {
    std::vector<float> q, k_cache, v_cache, output, scores, probs;
    Buffers(int n_heads, int n_kv_heads, int head_size, int seq_len)
        : q(static_cast<std::size_t>(n_heads) * head_size),
          k_cache(static_cast<std::size_t>(seq_len) * n_kv_heads * head_size),
          v_cache(k_cache.size()),
          output(q.size()),
          scores(static_cast<std::size_t>(n_heads) * seq_len),
          probs(scores.size())
    {
    }
};

// Position zero: exactly one visible row, so every probability is 1 and
// each head's output is its KV head's value vector.
void test_position_zero()
{
    std::cout << "position zero\n";
    const int n_heads = 4, n_kv_heads = 2, head_size = 5, seq_len = 3;
    Buffers b(n_heads, n_kv_heads, head_size, seq_len);
    for (std::size_t i = 0; i < b.q.size(); i++) b.q[i] = filler(i);
    for (std::size_t i = 0; i < b.k_cache.size(); i++) { b.k_cache[i] = filler(i + 100); b.v_cache[i] = filler(i + 200); }

    expect_true("attention accepted", attention(b.output.data(), b.scores.data(), b.probs.data(),
        b.q.data(), b.k_cache.data(), b.v_cache.data(), n_heads, n_kv_heads, head_size, seq_len, 0));

    for (int h = 0; h < n_heads; h++) {
        expect_near("prob", b.probs[h * seq_len + 0], 1.0);
        const int kv = h / 2;
        for (int d = 0; d < head_size; d++) {
            expect_near("output = V[kv]", b.output[h * head_size + d], b.v_cache[kv * head_size + d]);
        }
    }
}

// All-zero keys give equal scores, hence uniform probabilities and an
// output equal to the arithmetic mean of the visible value rows.
void test_equal_scores()
{
    std::cout << "equal scores\n";
    const int n_heads = 2, n_kv_heads = 1, head_size = 3, seq_len = 6, pos = 4;
    Buffers b(n_heads, n_kv_heads, head_size, seq_len);
    for (std::size_t i = 0; i < b.q.size(); i++) b.q[i] = filler(i);
    for (std::size_t i = 0; i < b.v_cache.size(); i++) b.v_cache[i] = filler(i + 50);
    // k_cache stays zero

    expect_true("attention accepted", attention(b.output.data(), b.scores.data(), b.probs.data(),
        b.q.data(), b.k_cache.data(), b.v_cache.data(), n_heads, n_kv_heads, head_size, seq_len, pos));

    for (int h = 0; h < n_heads; h++) {
        for (int t = 0; t <= pos; t++) {
            expect_near("score", b.scores[h * seq_len + t], 0.0);
            expect_near("uniform prob", b.probs[h * seq_len + t], 1.0 / (pos + 1));
        }
        for (int d = 0; d < head_size; d++) {
            double mean = 0;
            for (int t = 0; t <= pos; t++) mean += b.v_cache[t * head_size + d];
            mean /= (pos + 1);
            expect_near("mean of V", b.output[h * head_size + d], mean);
        }
    }
}

// head_size = 1, so the scale is 1.  Q = 1 against keys [0, ln 3] gives
// scores [0, ln 3] and probabilities [1, 3] / 4 = [0.25, 0.75].
// With values [10, 20] the output is 2.5 + 15 = 17.5.
void test_hand_checkable()
{
    std::cout << "hand-checkable nonuniform case\n";
    const int n_heads = 1, n_kv_heads = 1, head_size = 1, seq_len = 2, pos = 1;
    Buffers b(n_heads, n_kv_heads, head_size, seq_len);
    b.q[0] = 1.0f;
    b.k_cache = {0.0f, std::log(3.0f)};
    b.v_cache = {10.0f, 20.0f};

    expect_true("attention accepted", attention(b.output.data(), b.scores.data(), b.probs.data(),
        b.q.data(), b.k_cache.data(), b.v_cache.data(), n_heads, n_kv_heads, head_size, seq_len, pos));

    expect_near("score 0", b.scores[0], 0.0);
    expect_near("score 1", b.scores[1], std::log(3.0));
    expect_near("prob 0", b.probs[0], 0.25);
    expect_near("prob 1", b.probs[1], 0.75);
    expect_near("output", b.output[0], 17.5);
}

// Huge finite scores must not overflow: the stable softmax subtracts the
// maximum first.  [-1e4, 0, 1e4] -> [0, 0, 1]; all equal -1e4 -> uniform.
void test_large_scores()
{
    std::cout << "large finite scores\n";
    {
        const float s[3] = {-1e4f, 0.0f, 1e4f};
        float p[3];
        softmax(p, s, 3);
        double sum = 0;
        for (int i = 0; i < 3; i++) { expect_true("finite", std::isfinite(p[i])); sum += p[i]; }
        expect_near("sum", sum, 1.0);
        expect_near("p2", p[2], 1.0);
    }
    {
        const float s[4] = {-1e4f, -1e4f, -1e4f, -1e4f};
        float p[4];
        softmax(p, s, 4);
        double sum = 0;
        for (int i = 0; i < 4; i++) { expect_near("uniform", p[i], 0.25); sum += p[i]; }
        expect_near("sum", sum, 1.0);
    }
    // Through attention: q = 1e4, keys [-1, 0, 1] -> scores -1e4, 0, 1e4.
    {
        const int seq_len = 3, pos = 2;
        Buffers b(1, 1, 1, seq_len);
        b.q[0] = 1e4f;
        b.k_cache = {-1.0f, 0.0f, 1.0f};
        b.v_cache = {5.0f, 6.0f, 7.0f};
        expect_true("attention accepted", attention(b.output.data(), b.scores.data(), b.probs.data(),
            b.q.data(), b.k_cache.data(), b.v_cache.data(), 1, 1, 1, seq_len, pos));
        expect_near("prob 2", b.probs[2], 1.0);
        expect_near("output", b.output[0], 7.0);
    }
}

// Storing row 2 must leave rows 0 and 1 intact and rows 3+ untouched.
void test_cache_append()
{
    std::cout << "cache append preserves rows\n";
    const int kv_dim = 4, seq_len = 5;
    std::vector<float> cache(kv_dim * seq_len, -999.0f);   // sentinel
    std::vector<float> rows[3] = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
    for (int pos = 0; pos < 3; pos++) {
        expect_true("store accepted", kv_cache_store(cache.data(), rows[pos].data(), kv_dim, seq_len, pos));
    }
    for (int pos = 0; pos < 3; pos++) {
        for (int i = 0; i < kv_dim; i++) {
            expect_near("row content", cache[pos * kv_dim + i], rows[pos][i]);
        }
    }
    for (int i = 3 * kv_dim; i < seq_len * kv_dim; i++) {
        expect_near("untouched sentinel", cache[i], -999.0);
    }
    expect_true("store rejects pos = seq_len", !kv_cache_store(cache.data(), rows[0].data(), kv_dim, seq_len, seq_len));
    expect_true("store rejects negative pos", !kv_cache_store(cache.data(), rows[0].data(), kv_dim, seq_len, -1));
}

// Four query heads, two KV heads, head_size 2, position 1.  Keys are unit
// vectors arranged so that each query head, with a large one-hot query,
// attends almost entirely to one specific row of its own KV head:
//   head 0 -> kv 0 row 0     head 1 -> kv 0 row 1
//   head 2 -> kv 1 row 1     head 3 -> kv 1 row 0
void test_kv_head_mapping()
{
    std::cout << "query-to-KV head mapping\n";
    const int n_heads = 4, n_kv_heads = 2, head_size = 2, seq_len = 2, pos = 1;
    Buffers b(n_heads, n_kv_heads, head_size, seq_len);
    const float big = 100.0f;
    b.q = {big, 0,  0, big,  big, 0,  0, big};
    //             kv0      kv1
    b.k_cache = {  1, 0,    0, 1,     // row 0
                   0, 1,    1, 0 };   // row 1
    b.v_cache = { 10, 11,   20, 21,   // row 0
                  12, 13,   22, 23 }; // row 1

    expect_true("attention accepted", attention(b.output.data(), b.scores.data(), b.probs.data(),
        b.q.data(), b.k_cache.data(), b.v_cache.data(), n_heads, n_kv_heads, head_size, seq_len, pos));

    // Score gap is 100/sqrt(2) ~ 70, so the losing probability is ~e^-70.
    const float expected[8] = {10, 11,  12, 13,  22, 23,  20, 21};
    for (int i = 0; i < 8; i++) {
        expect_near("output", b.output[i], expected[i]);
    }
    expect_near("head0 prob row0", b.probs[0 * seq_len + 0], 1.0);
    expect_near("head1 prob row1", b.probs[1 * seq_len + 1], 1.0);
    expect_near("head2 prob row1", b.probs[2 * seq_len + 1], 1.0);
    expect_near("head3 prob row0", b.probs[3 * seq_len + 0], 1.0);
}

// Compares attention() against the double reference on a filled cache.
void check_against_reference(const char* label, int n_heads, int n_kv_heads, int head_size, int seq_len, int pos)
{
    Buffers b(n_heads, n_kv_heads, head_size, seq_len);
    for (std::size_t i = 0; i < b.q.size(); i++) b.q[i] = filler(i + 7);
    for (std::size_t i = 0; i < b.k_cache.size(); i++) { b.k_cache[i] = filler(i + 11); b.v_cache[i] = filler(i + 13); }

    expect_true("attention accepted", attention(b.output.data(), b.scores.data(), b.probs.data(),
        b.q.data(), b.k_cache.data(), b.v_cache.data(), n_heads, n_kv_heads, head_size, seq_len, pos));

    const std::vector<double> ref = reference_attention(b.q, b.k_cache, b.v_cache, n_heads, n_kv_heads, head_size, pos);
    for (std::size_t i = 0; i < ref.size(); i++) {
        expect_near(label, b.output[i], ref[i]);
    }
    for (int h = 0; h < n_heads; h++) {
        double sum = 0;
        for (int t = 0; t <= pos; t++) sum += b.probs[h * seq_len + t];
        expect_near("probabilities sum to 1", sum, 1.0);
    }
}

// Prime head size and an odd active length exercise every loop tail.
void test_awkward_shape()
{
    std::cout << "awkward head size and sequence length\n";
    check_against_reference("awkward vs reference", 3, 1, 37, 53, 52);
}

// Active length 300 is longer than any block size the kernels use.
void test_long_sequence()
{
    std::cout << "sequence longer than a block\n";
    check_against_reference("long vs reference", 2, 2, 8, 300, 299);
}

// Rows beyond pos hold NaN.  If anything read them the output would be
// NaN, so a finite result equal to the clean-cache result proves the
// causal bound.
void test_future_rows_ignored()
{
    std::cout << "future rows cannot leak\n";
    const int n_heads = 2, n_kv_heads = 1, head_size = 4, seq_len = 8, pos = 2;
    Buffers clean(n_heads, n_kv_heads, head_size, seq_len);
    for (std::size_t i = 0; i < clean.q.size(); i++) clean.q[i] = filler(i + 3);
    for (std::size_t i = 0; i < clean.k_cache.size(); i++) { clean.k_cache[i] = filler(i + 5); clean.v_cache[i] = filler(i + 9); }

    Buffers dirty = clean;
    const int kv_dim = n_kv_heads * head_size;
    for (std::size_t i = static_cast<std::size_t>(pos + 1) * kv_dim; i < dirty.k_cache.size(); i++) {
        dirty.k_cache[i] = std::numeric_limits<float>::quiet_NaN();
        dirty.v_cache[i] = std::numeric_limits<float>::quiet_NaN();
    }

    expect_true("clean accepted", attention(clean.output.data(), clean.scores.data(), clean.probs.data(),
        clean.q.data(), clean.k_cache.data(), clean.v_cache.data(), n_heads, n_kv_heads, head_size, seq_len, pos));
    expect_true("dirty accepted", attention(dirty.output.data(), dirty.scores.data(), dirty.probs.data(),
        dirty.q.data(), dirty.k_cache.data(), dirty.v_cache.data(), n_heads, n_kv_heads, head_size, seq_len, pos));

    for (std::size_t i = 0; i < clean.output.size(); i++) {
        expect_near("output unaffected", dirty.output[i], clean.output[i], 0.0);
    }
}

void test_rejections()
{
    std::cout << "position limits and invalid configuration\n";
    Buffers b(4, 2, 4, 8);
    auto run = [&](int n_heads, int n_kv_heads, int head_size, int seq_len, int pos) {
        return attention(b.output.data(), b.scores.data(), b.probs.data(),
            b.q.data(), b.k_cache.data(), b.v_cache.data(), n_heads, n_kv_heads, head_size, seq_len, pos);
    };
    expect_true("valid shape accepted", run(4, 2, 4, 8, 7));
    expect_true("pos = seq_len rejected", !run(4, 2, 4, 8, 8));
    expect_true("negative pos rejected", !run(4, 2, 4, 8, -1));
    expect_true("n_heads not multiple of n_kv_heads rejected", !run(3, 2, 4, 8, 0));
    expect_true("zero head_size rejected", !run(4, 2, 0, 8, 0));
    expect_true("zero n_kv_heads rejected", !run(4, 0, 4, 8, 0));
    expect_true("shape helper agrees", !attention_shape_valid(4, 2, 4, 8, 8));
}

void test_cache_state()
{
    std::cout << "cache state: sequential rows and reset\n";
    KvCacheState state(3, 4);
    expect_true("capacity", state.capacity_floats() == 12);
    expect_true("row offset", state.row_offset(2) == 8);
    expect_true("accepts 0 first", state.accepts(0));
    expect_true("rejects 1 before 0", !state.accepts(1));
    expect_true("covers nothing yet", !state.covers(0));
    state.advance();
    expect_true("covers 0", state.covers(0));
    expect_true("does not cover 1", !state.covers(1));
    expect_true("accepts 1 now", state.accepts(1));
    expect_true("rejects repeat of 0", !state.accepts(0));
    state.advance();
    state.advance();
    expect_true("full", state.length() == 3);
    expect_true("rejects pos = seq_len", !state.accepts(3));
    state.reset();
    expect_true("reset length", state.length() == 0);
    expect_true("reset covers nothing", !state.covers(0));
    expect_true("reset accepts 0", state.accepts(0));
}

} // namespace

int main()
{
    test_position_zero();
    test_equal_scores();
    test_hand_checkable();
    test_large_scores();
    test_cache_append();
    test_kv_head_mapping();
    test_awkward_shape();
    test_long_sequence();
    test_future_rows_ignored();
    test_rejections();
    test_cache_state();

    if (failures == 0) {
        std::cout << "All Passed\n";
        return 0;
    }
    std::cout << failures << " failure(s)\n";
    return 1;
}
