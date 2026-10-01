// CPU tests for the whole-model forward pass on a synthetic multi-layer
// fixture with distinct per-layer weights, hidden_dim != dim, and GQA.

#include "model_forward.hpp"
#include "test_model_reference.hpp"

#include <cmath>
#include <cstddef>
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

const std::vector<int> kTokens = {3, 0, 6, 2, 5};

void check_against_oracle(const char* label, bool tied, int n_kv_heads)
{
    std::cout << label << '\n';
    synthetic_model::Fixture f(tied, 3, n_kv_heads);
    const std::vector<synthetic_model::RefPosition> ref = synthetic_model::reference_forward(f, kTokens);

    CpuModel model(f.config, f.weights);
    expect_true("shared flag", model.shape().shared_classifier == tied);
    expect_true("vocab count", model.shape().vocab_count == 7);

    for (int pos = 0; pos < static_cast<int>(kTokens.size()); pos++) {
        expect_true("forward accepted", model.forward(kTokens[pos], pos));
        for (int l = 0; l < f.config.n_layers; l++) {
            cmp("layer output", model.layer_output(l), ref[pos].layer_outputs[l]);
            // Cache isolation: layer l's row pos holds layer l's K/V and nothing else.
            const CpuKvCache& c = model.layer_cache(l);
            const std::size_t off = static_cast<std::size_t>(pos) * f.kv_dim;
            for (int i = 0; i < f.kv_dim; i++) {
                expect_near("k cache row", c.k_cache[off + i], ref[pos].k_rows[l][i], 1e-5 + 1e-5 * std::abs(ref[pos].k_rows[l][i]));
                expect_near("v cache row", c.v_cache[off + i], ref[pos].v_rows[l][i], 1e-5 + 1e-5 * std::abs(ref[pos].v_rows[l][i]));
            }
            for (std::size_t i = off + f.kv_dim; i < c.k_cache.size(); i++) {
                expect_near("future k row untouched", c.k_cache[i], 0.0, 0.0);
            }
        }
        cmp("final norm", model.final_normalized(), ref[pos].final_norm);
        cmp("logits", model.logits(), ref[pos].logits);
    }
    expect_true("position advanced", model.position() == static_cast<int>(kTokens.size()));
}

// Layers 1 and 2 made into identities (Wo = 0, W2 = 0): the final layer
// output must equal layer 0's output, which must differ from the embedding.
void test_layer_chaining()
{
    std::cout << "layer N receives layer N-1 output\n";
    synthetic_model::Fixture f(true);
    for (int l = 1; l < 3; l++) {
        std::fill(f.wo[l].begin(), f.wo[l].end(), 0.0f);
        std::fill(f.w2[l].begin(), f.w2[l].end(), 0.0f);
    }
    f.rebuild_flat();
    CpuModel model(f.config, f.weights);
    expect_true("forward", model.forward(4, 0));
    bool differs = false;
    for (int i = 0; i < f.dim; i++) {
        if (model.layer_output(2)[i] != model.layer_output(0)[i]) {
            std::cout << "  FAIL layer2 != layer0 at " << i << '\n';
            failures++;
        }
        if (model.layer_output(0)[i] != f.embedding[4 * f.dim + i]) differs = true;
    }
    expect_true("layer0 differs from embedding", differs);
}

void test_final_norm_weight()
{
    std::cout << "final RMSNorm uses rms_final_weight\n";
    synthetic_model::Fixture f(true);
    std::fill(f.rms_final.begin(), f.rms_final.end(), 0.0f);
    f.rebuild_flat();
    CpuModel model(f.config, f.weights);
    expect_true("forward", model.forward(1, 0));
    for (float v : model.final_normalized()) expect_near("final norm zero", v, 0.0, 0.0);
    for (float v : model.logits()) expect_near("logits zero", v, 0.0, 0.0);
}

// Untied one-hot classifier: row v has a single 1 at column v % dim, so
// logits[v] must equal final_norm[v % dim].  A transposed read would not.
void test_classifier_orientation()
{
    std::cout << "classifier orientation and untied handling\n";
    synthetic_model::Fixture f(false);
    std::fill(f.wcls_untied.begin(), f.wcls_untied.end(), 0.0f);
    for (int v = 0; v < 7; v++) f.wcls_untied[v * f.dim + (v % f.dim)] = 1.0f;
    f.rebuild_flat();
    CpuModel model(f.config, f.weights);
    expect_true("untied", !model.shape().shared_classifier);
    expect_true("forward", model.forward(2, 0));
    for (int v = 0; v < 7; v++) {
        expect_near("one-hot logit", model.logits()[v], model.final_normalized()[v % f.dim], 1e-6);
    }
}

void test_rejections_and_exhaustion()
{
    std::cout << "invalid tokens, positions, and context exhaustion\n";
    synthetic_model::Fixture f(true);
    CpuModel model(f.config, f.weights);
    expect_true("negative token", !model.forward(-1, 0));
    expect_true("token == vocab", !model.forward(7, 0));
    expect_true("skipped position", !model.forward(0, 1));
    expect_true("nothing changed", model.position() == 0 && !model.needs_reset());
    expect_true("pos 0", model.forward(0, 0));
    expect_true("repeated position", !model.forward(0, 0));
    for (int pos = 1; pos < f.config.seq_len; pos++) expect_true("fill context", model.forward(pos % 7, pos));
    expect_true("context exhausted", !model.forward(0, f.config.seq_len));
    expect_true("still clean", !model.needs_reset());
    model.reset();
    expect_true("after reset", model.position() == 0 && model.forward(3, 0));
}

void test_reset_replay()
{
    std::cout << "reset + replay equals fresh state\n";
    synthetic_model::Fixture f(true);
    const std::vector<int> seq_a = {1, 2, 3, 4};
    const std::vector<int> seq_b = {6, 5, 0};
    CpuModel reused(f.config, f.weights);
    for (int p = 0; p < 4; p++) expect_true("a", reused.forward(seq_a[p], p));
    reused.reset();
    CpuModel fresh(f.config, f.weights);
    for (int p = 0; p < 3; p++) {
        expect_true("b reused", reused.forward(seq_b[p], p));
        expect_true("b fresh", fresh.forward(seq_b[p], p));
        for (int v = 0; v < 7; v++) {
            if (reused.logits()[v] != fresh.logits()[v]) { std::cout << "  FAIL replay logit differs\n"; failures++; }
        }
    }
}

} // namespace

int main()
{
    check_against_oracle("tied classifier, 1 KV head (GQA) vs oracle", true, 1);
    check_against_oracle("untied classifier, 1 KV head vs oracle", false, 1);
    check_against_oracle("tied classifier, 2 KV heads vs oracle", true, 2);
    test_layer_chaining();
    test_final_norm_weight();
    test_classifier_orientation();
    test_rejections_and_exhaustion();
    test_reset_replay();
    if (failures == 0) { std::cout << "All Passed\n"; return 0; }
    std::cout << failures << " failure(s)\n";
    return 1;
}
