// GPU tests for the elementwise kernels and the complete single-layer
// forward path, checked against hand values, the double oracle, and the
// CPU implementation.

#include "cpu_ops.hpp"
#include "cuda_ops.hpp"
#include "device_buffer.hpp"
#include "layer_forward.hpp"
#include "layer_forward_cuda.hpp"
#include "test_layer_reference.hpp"

#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <vector>

namespace {

constexpr float kTol = 1e-5f;
int failures = 0;

void expect_near(const char* label, double actual, double expected, double tol = kTol)
{
    if (!std::isfinite(actual) || std::abs(actual - expected) > tol) {
        std::cout << "  FAIL " << label << ": expected " << expected << ", got " << actual << '\n';
        failures++;
    }
}

void expect_true(const char* label, bool ok)
{
    if (!ok) { std::cout << "  FAIL " << label << '\n'; failures++; }
}

// Runs add or gate on the GPU for a given length.  mode: 0 = separate
// output, 1 = output aliases the first input, 2 = output aliases second.
std::vector<float> run_elementwise(bool gate, const std::vector<float>& a, const std::vector<float>& b, int mode)
{
    const int n = static_cast<int>(a.size());
    DeviceBuffer d_a(n), d_b(n), d_out(n);
    d_a.upload(a.data(), "upload a");
    d_b.upload(b.data(), "upload b");
    float* out = mode == 0 ? d_out.data() : (mode == 1 ? d_a.data() : d_b.data());
    check_cuda(gate ? silu_gate_cuda(out, d_a.data(), d_b.data(), n)
                    : add_vectors_cuda(out, d_a.data(), d_b.data(), n), "launch");
    check_cuda(cudaDeviceSynchronize(), "sync");
    return (mode == 0 ? d_out : (mode == 1 ? d_a : d_b)).download("download");
}

void test_add()
{
    std::cout << "residual addition\n";
    const std::vector<float> a = {1.5f, -2.0f, 0.0f, 3.0f, -0.25f};
    const std::vector<float> b = {0.5f, -1.0f, 0.0f, -3.0f, 0.25f};
    const float expected[5] = {2.0f, -3.0f, 0.0f, 0.0f, 0.0f};
    for (int mode = 0; mode < 3; mode++) {
        const std::vector<float> out = run_elementwise(false, a, b, mode);
        for (int i = 0; i < 5; i++) expect_near("sum", out[i], expected[i]);
    }
}

void test_gate()
{
    std::cout << "SiLU gating\n";
    const std::vector<float> h1 = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 1e4f, -1e4f};
    const std::vector<float> h3 = {5.0f, 2.0f, -3.0f, 0.5f, -4.0f, -0.5f, 7.0f};
    const double expected[7] = {0.0, 0.7310585786 * 2.0, -0.2689414214 * -3.0,
                                1.7615941560 * 0.5, -0.2384058440 * -4.0, -5e3, 0.0};
    for (int mode = 0; mode < 3; mode++) {
        const std::vector<float> out = run_elementwise(true, h1, h3, mode);
        for (int i = 0; i < 7; i++) {
            expect_true("finite", std::isfinite(out[i]));
            expect_near("gated", out[i], expected[i], i == 5 ? 1e-2 : kTol);
        }
    }
}

// Lengths below, above, and not divisible by the 256-thread block.
void test_lengths()
{
    std::cout << "vector lengths around the block size\n";
    for (int n : {1, 255, 256, 257, 1000, 1030}) {
        std::vector<float> a(n), b(n);
        for (int i = 0; i < n; i++) { a[i] = 0.01f * i - 3.0f; b[i] = 0.5f - 0.002f * i; }
        const std::vector<float> sum = run_elementwise(false, a, b, 0);
        const std::vector<float> gated = run_elementwise(true, a, b, 0);
        std::vector<float> sum_cpu(n), gated_cpu(n);
        add_vectors(sum_cpu.data(), a.data(), b.data(), n);
        silu_gate(gated_cpu.data(), a.data(), b.data(), n);
        for (int i = 0; i < n; i++) {
            expect_near("sum vs hand", sum[i], static_cast<double>(a[i]) + b[i]);
            expect_near("gated vs cpu", gated[i], gated_cpu[i], kTol + kTol * std::abs(gated_cpu[i]));
        }
    }
    DeviceBuffer d(1);
    expect_true("zero length rejected", add_vectors_cuda(d.data(), d.data(), d.data(), 0) == cudaErrorInvalidValue);
}

struct GpuStep {
    std::vector<float> att_out, projected, x_att, ffn_norm, h1, h3, gated, ffn_out, x_out;
};

GpuStep download_step(DeviceLayerState& s)
{
    GpuStep g;
    g.att_out = s.att_out.download("att_out");
    g.projected = s.projected.download("projected");
    g.x_att = s.x_att.download("x_att");
    g.ffn_norm = s.ffn_norm.download("ffn_norm");
    g.h1 = s.h1.download("h1");
    g.h3 = s.h3.download("h3");
    g.gated = s.gated.download("gated");
    g.ffn_out = s.ffn_out.download("ffn_out");
    g.x_out = s.x_out.download("x_out");
    return g;
}

void test_synthetic_layer()
{
    std::cout << "synthetic layer (hidden_dim != dim, GQA) vs reference and CPU\n";
    const LayerShape s = synthetic::shape();
    const synthetic::Weights w(s);
    const synthetic::RopeTables t(s);
    const int n = 3;
    const std::vector<synthetic::RefStep> ref = synthetic::reference_sequence(s, w, t, n, 1e-5);

    DeviceLayerWeights dw(s);
    dw.upload(w.view(), s);
    DeviceBuffer d_cos(t.cos_table.size()), d_sin(t.sin_table.size()), d_x(s.dim);
    d_cos.upload(t.cos_table.data(), "cos");
    d_sin.upload(t.sin_table.data(), "sin");
    DeviceLayerState ds(s);
    CpuLayerState cs(s);

    for (int pos = 0; pos < n; pos++) {
        const std::vector<float> x = synthetic::input(s, pos);
        d_x.upload(x.data(), "x");
        check_cuda(layer_forward_cuda(ds, dw, s, d_x.data(), d_cos.data() + pos * t.half, d_sin.data() + pos * t.half, pos, 1e-5f), "forward");
        check_cuda(cudaDeviceSynchronize(), "sync");
        expect_true("cpu forward accepted", layer_forward_cpu(cs, w.view(), s, x.data(), t.cos_row(pos), t.sin_row(pos), pos, 1e-5f));
        const GpuStep g = download_step(ds);

        auto cmp = [&](const char* name, const std::vector<float>& gpu, const std::vector<double>& want, const std::vector<float>& cpu) {
            for (std::size_t i = 0; i < want.size(); i++) {
                expect_near(name, gpu[i], want[i], 1e-5 + 1e-5 * std::abs(want[i]));
                expect_near(name, gpu[i], cpu[i], 1e-5 + 1e-5 * std::abs(cpu[i]));
            }
        };
        cmp("att_out", g.att_out, ref[pos].att_out, cs.att_out);
        cmp("projected", g.projected, ref[pos].projected, cs.projected);
        cmp("x_att", g.x_att, ref[pos].x_att, cs.x_att);
        cmp("ffn_norm", g.ffn_norm, ref[pos].ffn_norm, cs.ffn_norm);
        cmp("h1", g.h1, ref[pos].h1, cs.h1);
        cmp("h3", g.h3, ref[pos].h3, cs.h3);
        cmp("gated", g.gated, ref[pos].gated, cs.gated);
        cmp("ffn_out", g.ffn_out, ref[pos].ffn_out, cs.ffn_out);
        cmp("x_out", g.x_out, ref[pos].x_out, cs.x_out);
    }

    // Guards, then chaining: x_out may be fed back in as the next input.
    expect_true("rejects repeated position",
        layer_forward_cuda(ds, dw, s, d_x.data(), d_cos.data(), d_sin.data(), 0, 1e-5f) == cudaErrorInvalidValue);
    expect_true("rejects skipped position",
        layer_forward_cuda(ds, dw, s, d_x.data(), d_cos.data(), d_sin.data(), 4, 1e-5f) == cudaErrorInvalidValue);
    expect_true("no pending launch error", cudaGetLastError() == cudaSuccess);

    ds.reset();
    cs.reset();
    const std::vector<float> x0 = synthetic::input(s, 0);
    d_x.upload(x0.data(), "x");
    check_cuda(layer_forward_cuda(ds, dw, s, d_x.data(), d_cos.data(), d_sin.data(), 0, 1e-5f), "forward after reset");
    // Position 1 takes the previous x_out as its input, on both sides.
    std::vector<float> chained = ds.x_out.download("x_out");   // diagnostic only
    check_cuda(layer_forward_cuda(ds, dw, s, ds.x_out.data(), d_cos.data() + t.half, d_sin.data() + t.half, 1, 1e-5f), "chained forward");
    check_cuda(cudaDeviceSynchronize(), "sync");
    expect_true("cpu pos 0", layer_forward_cpu(cs, w.view(), s, x0.data(), t.cos_row(0), t.sin_row(0), 0, 1e-5f));
    const std::vector<float> cpu_chain_in = cs.x_out;
    expect_true("cpu pos 1", layer_forward_cpu(cs, w.view(), s, cpu_chain_in.data(), t.cos_row(1), t.sin_row(1), 1, 1e-5f));
    const std::vector<float> gx = ds.x_out.download("x_out");
    for (int i = 0; i < s.dim; i++) {
        expect_near("chained x_out", gx[i], cs.x_out[i], 1e-5 + 1e-5 * std::abs(cs.x_out[i]));
    }
}

void test_zero_branches()
{
    std::cout << "zero branches preserve the residual stream\n";
    const LayerShape s = synthetic::shape();
    const synthetic::RopeTables t(s);
    const std::vector<float> x = synthetic::input(s, 0);
    DeviceBuffer d_cos(t.cos_table.size()), d_sin(t.sin_table.size()), d_x(s.dim);
    d_cos.upload(t.cos_table.data(), "cos");
    d_sin.upload(t.sin_table.data(), "sin");
    d_x.upload(x.data(), "x");

    synthetic::Weights w(s);
    std::fill(w.wo.begin(), w.wo.end(), 0.0f);
    std::fill(w.w2.begin(), w.w2.end(), 0.0f);
    {
        DeviceLayerWeights dw(s); dw.upload(w.view(), s);
        DeviceLayerState ds(s);
        check_cuda(layer_forward_cuda(ds, dw, s, d_x.data(), d_cos.data(), d_sin.data(), 0, 1e-5f), "forward");
        check_cuda(cudaDeviceSynchronize(), "sync");
        const std::vector<float> xa = ds.x_att.download("x_att"), xo = ds.x_out.download("x_out");
        for (int i = 0; i < s.dim; i++) {
            expect_near("x_att == x_in", xa[i], x[i], 0.0);
            expect_near("x_out == x_in", xo[i], x[i], 0.0);
        }
    }
    synthetic::Weights w2(s);
    std::fill(w2.w2.begin(), w2.w2.end(), 0.0f);
    {
        DeviceLayerWeights dw(s); dw.upload(w2.view(), s);
        DeviceLayerState ds(s);
        check_cuda(layer_forward_cuda(ds, dw, s, d_x.data(), d_cos.data(), d_sin.data(), 0, 1e-5f), "forward");
        check_cuda(cudaDeviceSynchronize(), "sync");
        const std::vector<float> xa = ds.x_att.download("x_att"), xo = ds.x_out.download("x_out");
        bool changed = false;
        for (int i = 0; i < s.dim; i++) {
            expect_near("x_out == x_att", xo[i], xa[i], 0.0);
            if (xa[i] != x[i]) changed = true;
        }
        expect_true("attention update present in x_att", changed);
    }
}

} // namespace

int main()
{
    try {
        test_add();
        test_gate();
        test_lengths();
        test_synthetic_layer();
        test_zero_branches();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
    if (failures == 0) { std::cout << "All Passed\n"; return 0; }
    std::cout << failures << " failure(s)\n";
    return 1;
}
