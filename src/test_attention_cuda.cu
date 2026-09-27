// Checkpoint-independent tests for the CUDA KV cache and attention
// kernels.  Each case runs on the GPU and is checked against hand-derived
// values or an independent double-precision reference, and then against
// the CPU implementation so the two never drift apart.

#include "attention_shape.hpp"
#include "cpu_ops.hpp"
#include "cuda_ops.hpp"
#include "device_buffer.hpp"

#include <cmath>
#include <cstddef>
#include <exception>
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

float filler(std::size_t i)
{
    return static_cast<float>((i * 2654435761u) % 2000) / 1000.0f - 1.0f;
}

// Naive double-precision attention, independent of both implementations.
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

struct Inputs {
    int n_heads, n_kv_heads, head_size, seq_len, pos;
    std::vector<float> q, k_cache, v_cache;

    Inputs(int nh, int nkv, int hs, int sl, int p)
        : n_heads(nh), n_kv_heads(nkv), head_size(hs), seq_len(sl), pos(p),
          q(static_cast<std::size_t>(nh) * hs),
          k_cache(static_cast<std::size_t>(sl) * nkv * hs),
          v_cache(k_cache.size())
    {
    }

    void fill(std::size_t seed)
    {
        for (std::size_t i = 0; i < q.size(); i++) q[i] = filler(i + seed);
        for (std::size_t i = 0; i < k_cache.size(); i++) {
            k_cache[i] = filler(i + seed + 100);
            v_cache[i] = filler(i + seed + 200);
        }
    }
};

struct Results {
    std::vector<float> output, scores, probs;
};

// Uploads the inputs (caches already populated on the host), runs the
// three attention kernels, and brings everything back.  All device memory
// is released when this returns, however it returns.
Results run_gpu(const Inputs& in)
{
    DeviceBuffer d_q(in.q.size());
    DeviceBuffer d_k(in.k_cache.size());
    DeviceBuffer d_v(in.v_cache.size());
    DeviceBuffer d_out(in.q.size());
    DeviceBuffer d_scores(static_cast<std::size_t>(in.n_heads) * in.seq_len);
    DeviceBuffer d_probs(d_scores.count());

    d_q.upload(in.q.data(), "upload q");
    d_k.upload(in.k_cache.data(), "upload k cache");
    d_v.upload(in.v_cache.data(), "upload v cache");

    // The kernels only write the causal prefix [0, pos] of each scratch
    // row.  Zero the rest so the full-buffer readback below never copies
    // uninitialised device memory (which Compute Sanitizer's initcheck
    // would otherwise flag, harmlessly, on the memcpy).
    check_cuda(cudaMemset(d_scores.data(), 0, d_scores.bytes()), "clear scores");
    check_cuda(cudaMemset(d_probs.data(), 0, d_probs.bytes()), "clear probs");

    check_cuda(
        attention_cuda(d_out.data(), d_scores.data(), d_probs.data(),
                       d_q.data(), d_k.data(), d_v.data(),
                       in.n_heads, in.n_kv_heads, in.head_size, in.seq_len, in.pos),
        "launch attention_cuda"
    );
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    Results r;
    r.output = d_out.download("download output");
    r.scores = d_scores.download("download scores");
    r.probs = d_probs.download("download probs");
    return r;
}

// Same inputs through the CPU implementation.
Results run_cpu(const Inputs& in)
{
    Results r;
    r.output.resize(in.q.size());
    r.scores.resize(static_cast<std::size_t>(in.n_heads) * in.seq_len);
    r.probs.resize(r.scores.size());
    expect_true("cpu attention accepted",
        attention(r.output.data(), r.scores.data(), r.probs.data(),
                  in.q.data(), in.k_cache.data(), in.v_cache.data(),
                  in.n_heads, in.n_kv_heads, in.head_size, in.seq_len, in.pos));
    return r;
}

// Output plus the active part of scores and probs must agree with the CPU.
void expect_matches_cpu(const char* label, const Inputs& in, const Results& gpu)
{
    const Results cpu = run_cpu(in);
    for (std::size_t i = 0; i < cpu.output.size(); i++) {
        expect_near(label, gpu.output[i], cpu.output[i]);
    }
    for (int h = 0; h < in.n_heads; h++) {
        for (int t = 0; t <= in.pos; t++) {
            const std::size_t i = static_cast<std::size_t>(h) * in.seq_len + t;
            expect_near("scores cpu/gpu", gpu.scores[i], cpu.scores[i], kTolerance + kTolerance * std::abs(cpu.scores[i]));
            expect_near("probs cpu/gpu", gpu.probs[i], cpu.probs[i]);
        }
    }
}

void test_position_zero()
{
    std::cout << "position zero\n";
    Inputs in(4, 2, 5, 3, 0);
    in.fill(1);
    const Results r = run_gpu(in);
    for (int h = 0; h < in.n_heads; h++) {
        expect_near("prob", r.probs[h * in.seq_len], 1.0);
        const int kv = h / 2;
        for (int d = 0; d < in.head_size; d++) {
            expect_near("output = V[kv]", r.output[h * in.head_size + d], in.v_cache[kv * in.head_size + d]);
        }
    }
    expect_matches_cpu("cpu/gpu", in, r);
}

void test_equal_scores()
{
    std::cout << "equal scores\n";
    Inputs in(2, 1, 3, 6, 4);
    in.fill(2);
    std::fill(in.k_cache.begin(), in.k_cache.end(), 0.0f);
    const Results r = run_gpu(in);
    for (int h = 0; h < in.n_heads; h++) {
        for (int t = 0; t <= in.pos; t++) {
            expect_near("uniform prob", r.probs[h * in.seq_len + t], 1.0 / (in.pos + 1));
        }
        for (int d = 0; d < in.head_size; d++) {
            double mean = 0;
            for (int t = 0; t <= in.pos; t++) mean += in.v_cache[t * in.head_size + d];
            expect_near("mean of V", r.output[h * in.head_size + d], mean / (in.pos + 1));
        }
    }
    expect_matches_cpu("cpu/gpu", in, r);
}

void test_hand_checkable()
{
    std::cout << "hand-checkable nonuniform case\n";
    Inputs in(1, 1, 1, 2, 1);
    in.q = {1.0f};
    in.k_cache = {0.0f, std::log(3.0f)};
    in.v_cache = {10.0f, 20.0f};
    const Results r = run_gpu(in);
    expect_near("score 1", r.scores[1], std::log(3.0));
    expect_near("prob 0", r.probs[0], 0.25);
    expect_near("prob 1", r.probs[1], 0.75);
    expect_near("output", r.output[0], 17.5);
}

void test_large_scores()
{
    std::cout << "large finite scores\n";
    // Softmax kernel directly on hand-picked rows, one of which is longer
    // than the block so the strided loops are also exercised.
    const int n_rows = 2, row_stride = 300, active_len = 300;
    std::vector<float> scores(static_cast<std::size_t>(n_rows) * row_stride, -1e4f);
    scores[0 * row_stride + 299] = 1e4f;   // row 0: one dominant entry at the tail
    // row 1: all -1e4 -> uniform

    DeviceBuffer d_scores(scores.size());
    DeviceBuffer d_probs(scores.size());
    d_scores.upload(scores.data(), "upload scores");
    check_cuda(softmax_rows_cuda(d_probs.data(), d_scores.data(), n_rows, row_stride, active_len), "launch softmax");
    check_cuda(cudaDeviceSynchronize(), "sync");
    const std::vector<float> probs = d_probs.download("download probs");

    double sum0 = 0, sum1 = 0;
    for (int t = 0; t < active_len; t++) {
        expect_true("finite", std::isfinite(probs[t]) && std::isfinite(probs[row_stride + t]));
        sum0 += probs[t];
        sum1 += probs[row_stride + t];
        expect_near("uniform row", probs[row_stride + t], 1.0 / active_len);
    }
    expect_near("dominant entry", probs[299], 1.0);
    expect_near("row 0 sums to 1", sum0, 1.0);
    expect_near("row 1 sums to 1", sum1, 1.0);

    // Through attention: q = 1e4, keys [-1, 0, 1].
    Inputs in(1, 1, 1, 3, 2);
    in.q = {1e4f};
    in.k_cache = {-1.0f, 0.0f, 1.0f};
    in.v_cache = {5.0f, 6.0f, 7.0f};
    const Results r = run_gpu(in);
    expect_near("prob 2", r.probs[2], 1.0);
    expect_near("output", r.output[0], 7.0);
}

// Store rows 0..2 on the device with the kernel; sentinels elsewhere.
void test_cache_append()
{
    std::cout << "cache append preserves rows\n";
    const int kv_dim = 300, seq_len = 5;   // 300 > one store block
    std::vector<float> host(static_cast<std::size_t>(kv_dim) * seq_len, -999.0f);
    DeviceBuffer d_cache(host.size());
    d_cache.upload(host.data(), "upload sentinel cache");

    std::vector<std::vector<float>> rows(3, std::vector<float>(kv_dim));
    for (int pos = 0; pos < 3; pos++) {
        for (int i = 0; i < kv_dim; i++) rows[pos][i] = static_cast<float>(pos * 1000 + i);
        DeviceBuffer d_row(kv_dim);
        d_row.upload(rows[pos].data(), "upload row");
        check_cuda(kv_cache_store_cuda(d_cache.data(), d_row.data(), kv_dim, seq_len, pos), "launch store");
        check_cuda(cudaDeviceSynchronize(), "sync");   // d_row is freed after this
    }

    const std::vector<float> got = d_cache.download("download cache");
    for (int pos = 0; pos < 3; pos++) {
        for (int i = 0; i < kv_dim; i++) {
            expect_near("row content", got[pos * kv_dim + i], rows[pos][i], 0.0);
        }
    }
    for (std::size_t i = static_cast<std::size_t>(3) * kv_dim; i < got.size(); i++) {
        expect_near("untouched sentinel", got[i], -999.0, 0.0);
    }
    expect_true("store rejects pos = seq_len",
        kv_cache_store_cuda(d_cache.data(), d_cache.data(), kv_dim, seq_len, seq_len) == cudaErrorInvalidValue);
    expect_true("store rejects negative pos",
        kv_cache_store_cuda(d_cache.data(), d_cache.data(), kv_dim, seq_len, -1) == cudaErrorInvalidValue);
}

void test_kv_head_mapping()
{
    std::cout << "query-to-KV head mapping\n";
    Inputs in(4, 2, 2, 2, 1);
    const float big = 100.0f;
    in.q = {big, 0,  0, big,  big, 0,  0, big};
    in.k_cache = { 1, 0,  0, 1,
                   0, 1,  1, 0 };
    in.v_cache = { 10, 11,  20, 21,
                   12, 13,  22, 23 };
    const Results r = run_gpu(in);
    const float expected[8] = {10, 11,  12, 13,  22, 23,  20, 21};
    for (int i = 0; i < 8; i++) expect_near("output", r.output[i], expected[i]);
    expect_near("head0 prob row0", r.probs[0 * 2 + 0], 1.0);
    expect_near("head1 prob row1", r.probs[1 * 2 + 1], 1.0);
    expect_near("head2 prob row1", r.probs[2 * 2 + 1], 1.0);
    expect_near("head3 prob row0", r.probs[3 * 2 + 0], 1.0);
    expect_matches_cpu("cpu/gpu", in, r);
}

void check_against_reference(const char* label, Inputs in)
{
    in.fill(17);
    const Results r = run_gpu(in);
    const std::vector<double> ref = reference_attention(in.q, in.k_cache, in.v_cache,
        in.n_heads, in.n_kv_heads, in.head_size, in.pos);
    for (std::size_t i = 0; i < ref.size(); i++) expect_near(label, r.output[i], ref[i]);
    for (int h = 0; h < in.n_heads; h++) {
        double sum = 0;
        for (int t = 0; t <= in.pos; t++) sum += r.probs[h * in.seq_len + t];
        expect_near("probabilities sum to 1", sum, 1.0);
    }
    expect_matches_cpu("cpu/gpu", in, r);
}

void test_awkward_shape()
{
    std::cout << "awkward head size and sequence length\n";
    // head_size 37 is prime and not a multiple of the 128-thread score
    // block; 53 active rows is an odd count.
    check_against_reference("awkward vs reference", Inputs(3, 1, 37, 53, 52));
}

void test_long_sequence()
{
    std::cout << "sequence longer than a block\n";
    // 300 active rows exceed the 256-thread softmax block, so every
    // thread handles more than one entry.
    check_against_reference("long vs reference", Inputs(2, 2, 8, 300, 299));
}

void test_future_rows_ignored()
{
    std::cout << "future rows cannot leak\n";
    Inputs clean(2, 1, 4, 8, 2);
    clean.fill(5);
    Inputs dirty = clean;
    const int kv_dim = clean.n_kv_heads * clean.head_size;
    for (std::size_t i = static_cast<std::size_t>(clean.pos + 1) * kv_dim; i < dirty.k_cache.size(); i++) {
        dirty.k_cache[i] = std::numeric_limits<float>::quiet_NaN();
        dirty.v_cache[i] = std::numeric_limits<float>::quiet_NaN();
    }
    const Results a = run_gpu(clean);
    const Results b = run_gpu(dirty);
    for (std::size_t i = 0; i < a.output.size(); i++) {
        expect_near("output unaffected", b.output[i], a.output[i], 0.0);
    }
}

void test_rejections()
{
    std::cout << "position limits and invalid configuration\n";
    DeviceBuffer d(64);
    auto run = [&](int nh, int nkv, int hs, int sl, int pos) {
        return attention_cuda(d.data(), d.data(), d.data(), d.data(), d.data(), d.data(), nh, nkv, hs, sl, pos);
    };
    expect_true("pos = seq_len rejected", run(4, 2, 4, 2, 2) == cudaErrorInvalidValue);
    expect_true("negative pos rejected", run(4, 2, 4, 2, -1) == cudaErrorInvalidValue);
    expect_true("n_heads not multiple of n_kv_heads rejected", run(3, 2, 4, 2, 0) == cudaErrorInvalidValue);
    expect_true("zero head_size rejected", run(4, 2, 0, 2, 0) == cudaErrorInvalidValue);
    expect_true("softmax active_len > stride rejected",
        softmax_rows_cuda(d.data(), d.data(), 1, 8, 9) == cudaErrorInvalidValue);
    // Nothing was launched, so no sticky error should be pending.
    expect_true("no pending error", cudaGetLastError() == cudaSuccess);
}

} // namespace

int main()
{
    try {
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
    }
    catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    if (failures == 0) {
        std::cout << "All Passed\n";
        return 0;
    }
    std::cout << failures << " failure(s)\n";
    return 1;
}
