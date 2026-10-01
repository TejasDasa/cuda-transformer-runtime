// GPU tests for the whole-model forward pass on the synthetic fixture,
// checked against the double oracle and the CPU model.

#include "model_forward.hpp"
#include "model_forward_cuda.hpp"
#include "test_model_reference.hpp"

#include <cmath>
#include <cstddef>
#include <exception>
#include <iostream>
#include <vector>

namespace {

int failures = 0;

void expect_near(const char* label, double actual, double expected, double tol)
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

void cmp(const char* name, const std::vector<float>& got, const std::vector<double>& want)
{
    expect_true("size", got.size() == want.size());
    for (std::size_t i = 0; i < want.size() && i < got.size(); i++) {
        expect_near(name, got[i], want[i], 1e-5 + 1e-5 * std::abs(want[i]));
    }
}

void cmp_cpu(const char* name, const std::vector<float>& gpu, const std::vector<float>& cpu)
{
    expect_true("size", gpu.size() == cpu.size());
    for (std::size_t i = 0; i < cpu.size() && i < gpu.size(); i++) {
        expect_near(name, gpu[i], cpu[i], 1e-5 + 1e-5 * std::abs(cpu[i]));
    }
}

void step(GpuModel& m, int token, int pos, const char* label)
{
    check_cuda(m.forward(token, pos), label);
    check_cuda(m.synchronize(), "synchronize");
}

const std::vector<int> kTokens = {3, 0, 6, 2, 5};

void check_against_oracle(const char* label, bool tied, int n_kv_heads)
{
    std::cout << label << '\n';
    synthetic_model::Fixture f(tied, 3, n_kv_heads);
    const std::vector<synthetic_model::RefPosition> ref = synthetic_model::reference_forward(f, kTokens);

    GpuModel gpu(f.config, f.weights, /*keep_layer_outputs=*/true);
    CpuModel cpu(f.config, f.weights);
    expect_true("tied flag", gpu.classifier_is_tied() == tied);

    for (int pos = 0; pos < static_cast<int>(kTokens.size()); pos++) {
        step(gpu, kTokens[pos], pos, "forward");
        expect_true("cpu forward", cpu.forward(kTokens[pos], pos));
        for (int l = 0; l < f.config.n_layers; l++) {
            const std::vector<float> out = gpu.layer_output(l).download("layer output");
            cmp("layer output vs oracle", out, ref[pos].layer_outputs[l]);
            cmp_cpu("layer output vs cpu", out, cpu.layer_output(l));

            const std::vector<float> kc = gpu.layer_cache(l).k_cache.download("k cache");
            const std::vector<float> vc = gpu.layer_cache(l).v_cache.download("v cache");
            const std::size_t off = static_cast<std::size_t>(pos) * f.kv_dim;
            for (int i = 0; i < f.kv_dim; i++) {
                expect_near("k cache row", kc[off + i], ref[pos].k_rows[l][i], 1e-5 + 1e-5 * std::abs(ref[pos].k_rows[l][i]));
                expect_near("v cache row", vc[off + i], ref[pos].v_rows[l][i], 1e-5 + 1e-5 * std::abs(ref[pos].v_rows[l][i]));
            }
            for (std::size_t i = off + f.kv_dim; i < kc.size(); i++) {
                expect_near("future k row untouched", kc[i], 0.0, 0.0);
                expect_near("future v row untouched", vc[i], 0.0, 0.0);
            }
        }
        cmp("final norm vs oracle", gpu.final_normalized().download("final"), ref[pos].final_norm);
        const std::vector<float> logits = gpu.logits().download("logits");
        cmp("logits vs oracle", logits, ref[pos].logits);
        cmp_cpu("logits vs cpu", logits, cpu.logits());
    }
}

void test_layer_chaining()
{
    std::cout << "layer N receives layer N-1 output\n";
    synthetic_model::Fixture f(true);
    for (int l = 1; l < 3; l++) {
        std::fill(f.wo[l].begin(), f.wo[l].end(), 0.0f);
        std::fill(f.w2[l].begin(), f.w2[l].end(), 0.0f);
    }
    f.rebuild_flat();
    GpuModel m(f.config, f.weights, true);
    step(m, 4, 0, "forward");
    const std::vector<float> l0 = m.layer_output(0).download("l0"), l2 = m.layer_output(2).download("l2");
    bool differs = false;
    for (int i = 0; i < f.dim; i++) {
        if (l2[i] != l0[i]) { std::cout << "  FAIL layer2 != layer0 at " << i << '\n'; failures++; }
        if (l0[i] != f.embedding[4 * f.dim + i]) differs = true;
    }
    expect_true("layer0 differs from embedding", differs);
}

void test_final_norm_weight()
{
    std::cout << "final RMSNorm uses rms_final_weight\n";
    synthetic_model::Fixture f(true);
    std::fill(f.rms_final.begin(), f.rms_final.end(), 0.0f);
    f.rebuild_flat();
    GpuModel m(f.config, f.weights);
    step(m, 1, 0, "forward");
    for (float v : m.final_normalized().download("final")) expect_near("final norm zero", v, 0.0, 0.0);
    for (float v : m.logits().download("logits")) expect_near("logits zero", v, 0.0, 0.0);
}

void test_classifier_orientation()
{
    std::cout << "classifier orientation and untied handling\n";
    synthetic_model::Fixture f(false);
    std::fill(f.wcls_untied.begin(), f.wcls_untied.end(), 0.0f);
    for (int v = 0; v < 7; v++) f.wcls_untied[v * f.dim + (v % f.dim)] = 1.0f;
    f.rebuild_flat();
    GpuModel m(f.config, f.weights);
    expect_true("untied", !m.classifier_is_tied());
    step(m, 2, 0, "forward");
    const std::vector<float> logits = m.logits().download("logits"), fn = m.final_normalized().download("final");
    for (int v = 0; v < 7; v++) expect_near("one-hot logit", logits[v], fn[v % f.dim], 1e-6);
}

void test_rejections_and_exhaustion()
{
    std::cout << "invalid tokens, positions, and context exhaustion\n";
    synthetic_model::Fixture f(true);
    GpuModel m(f.config, f.weights);
    expect_true("negative token", m.forward(-1, 0) == cudaErrorInvalidValue);
    expect_true("token == vocab", m.forward(7, 0) == cudaErrorInvalidValue);
    expect_true("skipped position", m.forward(0, 1) == cudaErrorInvalidValue);
    expect_true("nothing changed", m.position() == 0 && !m.needs_reset());
    expect_true("no pending error", cudaGetLastError() == cudaSuccess);
    step(m, 0, 0, "pos 0");
    expect_true("repeated position", m.forward(0, 0) == cudaErrorInvalidValue);
    for (int pos = 1; pos < f.config.seq_len; pos++) step(m, pos % 7, pos, "fill");
    expect_true("context exhausted", m.forward(0, f.config.seq_len) == cudaErrorInvalidValue);
    expect_true("still clean", !m.needs_reset());
    m.reset();
    expect_true("position reset", m.position() == 0);
    step(m, 3, 0, "after reset");
}

void test_reset_replay()
{
    std::cout << "reset + replay equals fresh state (bitwise)\n";
    synthetic_model::Fixture f(true);
    const std::vector<int> seq_a = {1, 2, 3, 4};
    const std::vector<int> seq_b = {6, 5, 0};
    GpuModel reused(f.config, f.weights);
    for (int p = 0; p < 4; p++) step(reused, seq_a[p], p, "a");
    reused.reset();
    GpuModel fresh(f.config, f.weights);
    for (int p = 0; p < 3; p++) {
        step(reused, seq_b[p], p, "b reused");
        step(fresh, seq_b[p], p, "b fresh");
        const std::vector<float> a = reused.logits().download("reused"), b = fresh.logits().download("fresh");
        for (int v = 0; v < 7; v++) {
            if (a[v] != b[v]) { std::cout << "  FAIL replay logit differs\n"; failures++; }
        }
    }
}

} // namespace

int main()
{
    try {
        check_against_oracle("tied classifier, 1 KV head (GQA) vs oracle and cpu", true, 1);
        check_against_oracle("untied classifier, 1 KV head vs oracle and cpu", false, 1);
        check_against_oracle("tied classifier, 2 KV heads vs oracle and cpu", true, 2);
        test_layer_chaining();
        test_final_norm_weight();
        test_classifier_orientation();
        test_rejections_and_exhaustion();
        test_reset_replay();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
    if (failures == 0) { std::cout << "All Passed\n"; return 0; }
    std::cout << failures << " failure(s)\n";
    return 1;
}
