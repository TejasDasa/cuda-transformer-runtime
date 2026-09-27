// Checkpoint-independent tests for the CUDA RoPE kernel.
//
// Each case rotates a vector on the GPU and checks it against a closed-form
// expectation, then also against the CPU rope() so the two stay in step.

#include "cpu_ops.hpp"
#include "cuda_ops.hpp"
#include "device_buffer.hpp"

#include <cmath>
#include <cstddef>
#include <exception>
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

// Runs rope_cuda on a copy of `input` and returns the rotated vector.
// cos/sin are uploaded alongside; all device memory is freed on return.
std::vector<float> rope_on_gpu(
    const std::vector<float>& input,
    int n_heads,
    int head_size,
    const std::vector<float>& cos_row,
    const std::vector<float>& sin_row
)
{
    DeviceBuffer d_vec(input.size());
    DeviceBuffer d_cos(cos_row.size());
    DeviceBuffer d_sin(sin_row.size());

    d_vec.upload(input.data(), "upload vec");
    d_cos.upload(cos_row.data(), "upload cos");
    d_sin.upload(sin_row.data(), "upload sin");

    check_cuda(
        rope_cuda(d_vec.data(), n_heads, head_size, d_cos.data(), d_sin.data()),
        "launch rope_cuda"
    );
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    return d_vec.download("download vec");
}

// GPU result must match the CPU implementation element for element.
void expect_matches_cpu(
    const char* label,
    const std::vector<float>& input,
    int n_heads,
    int head_size,
    const std::vector<float>& cos_row,
    const std::vector<float>& sin_row,
    const std::vector<float>& gpu
)
{
    std::vector<float> cpu = input;
    rope(cpu.data(), n_heads, head_size, cos_row.data(), sin_row.data());
    for (std::size_t i = 0; i < cpu.size(); i++) {
        expect_near(label, gpu[i], cpu[i]);
    }
}

void test_quarter_turn()
{
    std::cout << "quarter turn\n";
    const std::vector<float> v = {3.0f, 4.0f};
    const std::vector<float> c = {0.0f};
    const std::vector<float> s = {1.0f};
    const std::vector<float> out = rope_on_gpu(v, 1, 2, c, s);
    expect_near("a", out[0], -4.0f);
    expect_near("b", out[1], 3.0f);
}

void test_identity()
{
    std::cout << "identity\n";
    const std::vector<float> v = {0.5f, -1.25f, 2.0f, 3.75f, -0.125f, 9.0f};
    const std::vector<float> c = {1.0f, 1.0f, 1.0f};
    const std::vector<float> s = {0.0f, 0.0f, 0.0f};
    const std::vector<float> out = rope_on_gpu(v, 1, 6, c, s);
    for (std::size_t i = 0; i < v.size(); i++) {
        expect_near("element", out[i], v[i]);
    }
}

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
    std::vector<float> c(half), s(half);
    for (int p = 0; p < half; p++) {
        c[p] = std::cos(0.7f * static_cast<float>(p + 1));
        s[p] = std::sin(0.7f * static_cast<float>(p + 1));
    }

    const std::vector<float> out = rope_on_gpu(v, n_heads, head_size, c, s);

    for (int i = 0; i < n_heads * head_size; i += 2) {
        const float before = v[i] * v[i] + v[i + 1] * v[i + 1];
        const float after = out[i] * out[i] + out[i + 1] * out[i + 1];
        expect_near("pair norm", after, before);
    }
    expect_matches_cpu("cpu/gpu", v, n_heads, head_size, c, s, out);
}

void test_pattern_repeats_across_heads()
{
    std::cout << "pattern repeats across heads\n";
    constexpr int n_heads = 3;
    constexpr int head_size = 4;
    const std::vector<float> c = {0.0f, 1.0f};
    const std::vector<float> s = {1.0f, 0.0f};

    std::vector<float> v;
    for (int h = 0; h < n_heads; h++) {
        v.insert(v.end(), {1.0f, 2.0f, 5.0f, 6.0f});
    }

    const std::vector<float> out = rope_on_gpu(v, n_heads, head_size, c, s);

    for (int h = 0; h < n_heads; h++) {
        const float* head = out.data() + h * head_size;
        expect_near("pair0 a", head[0], -2.0f);
        expect_near("pair0 b", head[1], 1.0f);
        expect_near("pair1 a", head[2], 5.0f);
        expect_near("pair1 b", head[3], 6.0f);
    }
}

// Q and K are rotated by separate launches with their own head counts, so
// K's shorter length is handled without any assumption that it equals Q's.
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
    std::vector<float> k(q.begin(), q.begin() + n_kv_heads * head_size);
    k.push_back(123.0f);   // sentinel: the kernel must not write past K

    std::vector<float> c(half), s(half);
    for (int p = 0; p < half; p++) {
        c[p] = std::cos(0.3f * static_cast<float>(p + 1));
        s[p] = std::sin(0.3f * static_cast<float>(p + 1));
    }

    const std::vector<float> q_out = rope_on_gpu(q, n_heads, head_size, c, s);
    const std::vector<float> k_out = rope_on_gpu(k, n_kv_heads, head_size, c, s);

    for (int i = 0; i < n_kv_heads * head_size; i++) {
        expect_near("k vs q", k_out[i], q_out[i]);
    }
    expect_near("sentinel", k_out.back(), 123.0f);
    expect_matches_cpu("q cpu/gpu", q, n_heads, head_size, c, s, q_out);
    expect_matches_cpu("k cpu/gpu", k, n_kv_heads, head_size, c, s, k_out);
}

// 7 x 37 = 259 pairs: two blocks of 256 threads, the second one only 3
// threads full.  This exercises the bounds guard in the kernel.
void test_awkward_pair_count()
{
    std::cout << "awkward pair count (259 pairs, block size 256)\n";
    constexpr int n_heads = 7;
    constexpr int head_size = 74;
    constexpr int half = head_size / 2;

    std::vector<float> v(n_heads * head_size);
    for (std::size_t i = 0; i < v.size(); i++) {
        v[i] = 0.01f * static_cast<float>(i);
    }
    std::vector<float> c(half), s(half);
    for (int p = 0; p < half; p++) {
        c[p] = std::cos(0.05f * static_cast<float>(p));
        s[p] = std::sin(0.05f * static_cast<float>(p));
    }

    const std::vector<float> out = rope_on_gpu(v, n_heads, head_size, c, s);

    for (int h = 0; h < n_heads; h++) {
        for (int p = 0; p < half; p++) {
            const int i = h * head_size + 2 * p;
            expect_near("a", out[i], v[i] * c[p] - v[i + 1] * s[p]);
            expect_near("b", out[i + 1], v[i] * s[p] + v[i + 1] * c[p]);
        }
    }
    expect_matches_cpu("cpu/gpu", v, n_heads, head_size, c, s, out);
}

} // namespace

int main()
{
    try {
        test_quarter_turn();
        test_identity();
        test_norm_preserved();
        test_pattern_repeats_across_heads();
        test_fewer_kv_heads();
        test_awkward_pair_count();
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
