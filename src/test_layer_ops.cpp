// CPU tests for the residual add, SiLU gating, and the wiring of the
// complete single-layer forward path on a synthetic layer.

#include "cpu_ops.hpp"
#include "layer_forward.hpp"
#include "test_layer_reference.hpp"

#include <cmath>
#include <cstddef>
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

void test_add()
{
    std::cout << "residual addition\n";
    const std::vector<float> a = {1.5f, -2.0f, 0.0f, 3.0f, -0.25f};
    const std::vector<float> b = {0.5f, -1.0f, 0.0f, -3.0f, 0.25f};
    std::vector<float> out(5);
    add_vectors(out.data(), a.data(), b.data(), 5);
    const float expected[5] = {2.0f, -3.0f, 0.0f, 0.0f, 0.0f};
    for (int i = 0; i < 5; i++) expect_near("sum", out[i], expected[i]);

    // In place: output == a.
    std::vector<float> x = a;
    add_vectors(x.data(), x.data(), b.data(), 5);
    for (int i = 0; i < 5; i++) expect_near("in-place a", x[i], expected[i]);

    // In place: output == b.
    std::vector<float> y = b;
    add_vectors(y.data(), a.data(), y.data(), 5);
    for (int i = 0; i < 5; i++) expect_near("in-place b", y[i], expected[i]);
}

void test_silu_values()
{
    std::cout << "SiLU values\n";
    expect_near("silu(0)", silu(0.0f), 0.0);
    expect_near("silu(1)", silu(1.0f), 0.7310585786);
    expect_near("silu(-1)", silu(-1.0f), -0.2689414214);
    expect_near("silu(2)", silu(2.0f), 1.7615941560);
    expect_near("silu(-2)", silu(-2.0f), -0.2384058440);
    expect_near("sigmoid(0)", sigmoid(0.0f), 0.5);

    // Large finite inputs: naive exp(-z) or exp(z) would overflow float.
    expect_near("silu(1e4)", silu(1e4f), 1e4, 1e-2);
    expect_near("silu(-1e4)", silu(-1e4f), 0.0);
    expect_near("silu(88)", silu(88.0f), 88.0, 1e-3);
    expect_near("silu(-88)", silu(-88.0f), -88.0 * std::exp(-88.0), 1e-30);
    expect_true("sigmoid(1e4) finite", std::isfinite(sigmoid(1e4f)));
    expect_true("sigmoid(-1e4) finite", std::isfinite(sigmoid(-1e4f)));
}

void test_gate()
{
    std::cout << "SiLU gating\n";
    const std::vector<float> h1 = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 1e4f};
    const std::vector<float> h3 = {5.0f, 2.0f, -3.0f, 0.5f, -4.0f, -0.5f};
    std::vector<float> out(6);
    silu_gate(out.data(), h1.data(), h3.data(), 6);
    const double expected[6] = {0.0, 0.7310585786 * 2.0, -0.2689414214 * -3.0,
                                1.7615941560 * 0.5, -0.2384058440 * -4.0, -5e3};
    for (int i = 0; i < 6; i++) expect_near("gated", out[i], expected[i], i == 5 ? 1e-2 : kTol);

    std::vector<float> g = h1;   // in place over the gate
    silu_gate(g.data(), g.data(), h3.data(), 6);
    for (int i = 0; i < 6; i++) expect_near("in-place gate", g[i], expected[i], i == 5 ? 1e-2 : kTol);

    std::vector<float> u = h3;   // in place over the up projection
    silu_gate(u.data(), h1.data(), u.data(), 6);
    for (int i = 0; i < 6; i++) expect_near("in-place up", u[i], expected[i], i == 5 ? 1e-2 : kTol);
}

// The synthetic layer through layer_forward_cpu versus the double oracle.
void test_synthetic_layer()
{
    std::cout << "synthetic layer (hidden_dim != dim, GQA) vs double reference\n";
    const LayerShape s = synthetic::shape();
    expect_true("shape valid", s.valid());
    const synthetic::Weights w(s);
    const synthetic::RopeTables t(s);
    const int n = 3;
    const std::vector<synthetic::RefStep> ref = synthetic::reference_sequence(s, w, t, n, 1e-5);

    CpuLayerState st(s);
    for (int pos = 0; pos < n; pos++) {
        const std::vector<float> x = synthetic::input(s, pos);
        expect_true("forward accepted", layer_forward_cpu(st, w.view(), s, x.data(), t.cos_row(pos), t.sin_row(pos), pos, 1e-5f));
        auto cmp = [&](const char* name, const std::vector<float>& got, const std::vector<double>& want) {
            expect_true("size", got.size() == want.size());
            for (std::size_t i = 0; i < want.size(); i++) expect_near(name, got[i], want[i], 1e-5 + 1e-5 * std::abs(want[i]));
        };
        cmp("att_out", st.att_out, ref[pos].att_out);
        cmp("projected", st.projected, ref[pos].projected);
        cmp("x_att", st.x_att, ref[pos].x_att);
        cmp("ffn_norm", st.ffn_norm, ref[pos].ffn_norm);
        cmp("h1", st.h1, ref[pos].h1);
        cmp("h3", st.h3, ref[pos].h3);
        cmp("gated", st.gated, ref[pos].gated);
        cmp("ffn_out", st.ffn_out, ref[pos].ffn_out);
        cmp("x_out", st.x_out, ref[pos].x_out);
    }
    expect_true("rejects repeated position", !layer_forward_cpu(st, w.view(), s, synthetic::input(s, 0).data(), t.cos_row(0), t.sin_row(0), 0, 1e-5f));
    expect_true("rejects skipped position", !layer_forward_cpu(st, w.view(), s, synthetic::input(s, 0).data(), t.cos_row(4), t.sin_row(4), 4, 1e-5f));
    st.reset();
    expect_true("accepts position 0 after reset", layer_forward_cpu(st, w.view(), s, synthetic::input(s, 0).data(), t.cos_row(0), t.sin_row(0), 0, 1e-5f));
}

// Zero Wo: the attention branch contributes nothing, so x_att == x_in.
// Zero W2 as well: the FFN branch contributes nothing, so x_out == x_att.
void test_zero_branches()
{
    std::cout << "zero branches preserve the residual stream\n";
    const LayerShape s = synthetic::shape();
    const synthetic::RopeTables t(s);
    const std::vector<float> x = synthetic::input(s, 0);

    synthetic::Weights w(s);
    std::fill(w.wo.begin(), w.wo.end(), 0.0f);
    std::fill(w.w2.begin(), w.w2.end(), 0.0f);
    CpuLayerState st(s);
    expect_true("forward accepted", layer_forward_cpu(st, w.view(), s, x.data(), t.cos_row(0), t.sin_row(0), 0, 1e-5f));
    for (int i = 0; i < s.dim; i++) {
        expect_near("x_att == x_in", st.x_att[i], x[i], 0.0);
        expect_near("x_out == x_in", st.x_out[i], x[i], 0.0);
    }

    // Zero W2 only: x_out must equal x_att, which now differs from x_in.
    synthetic::Weights w2(s);
    std::fill(w2.w2.begin(), w2.w2.end(), 0.0f);
    CpuLayerState st2(s);
    expect_true("forward accepted", layer_forward_cpu(st2, w2.view(), s, x.data(), t.cos_row(0), t.sin_row(0), 0, 1e-5f));
    bool attention_changed_something = false;
    for (int i = 0; i < s.dim; i++) {
        expect_near("x_out == x_att", st2.x_out[i], st2.x_att[i], 0.0);
        if (st2.x_att[i] != x[i]) attention_changed_something = true;
    }
    expect_true("attention update present in x_att", attention_changed_something);
}

} // namespace

int main()
{
    test_add();
    test_silu_values();
    test_gate();
    test_synthetic_layer();
    test_zero_branches();
    if (failures == 0) { std::cout << "All Passed\n"; return 0; }
    std::cout << failures << " failure(s)\n";
    return 1;
}
