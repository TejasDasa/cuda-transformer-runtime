#ifndef TEST_MODEL_REFERENCE_HPP
#define TEST_MODEL_REFERENCE_HPP

// Test-only: a tiny synthetic multi-layer model laid out exactly like the
// checkpoint (tensor-by-tensor, layers back to back inside each tensor),
// plus a naive double-precision oracle.
//
// Independence from the code under test: the fixture keeps every layer's
// weights in SEPARATE per-layer vectors and the oracle reads those
// directly.  The flat checkpoint-style tensors are built by appending the
// per-layer vectors in order, so an offset mistake shared by the CPU and
// GPU implementations would show up as a mismatch against the oracle.

#include "model_config.hpp"
#include "model_weights.hpp"
#include "test_layer_reference.hpp"   // synthetic::filler, ref_rmsnorm, ref_matvec

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace synthetic_model {

struct Fixture {
    ModelConfig config{};

    // Per-layer weights (oracle reads these).
    std::vector<std::vector<float>> rms_att, wq, wk, wv, wo, rms_ffn, w1, w2, w3;

    // Checkpoint-style flat tensors (implementation reads these).
    std::vector<float> flat_rms_att, flat_wq, flat_wk, flat_wv, flat_wo;
    std::vector<float> flat_rms_ffn, flat_w1, flat_w2, flat_w3;
    std::vector<float> embedding, rms_final, cos_table, sin_table, wcls_untied;

    ModelWeights weights{};

    int dim = 8, hidden = 12, head_size = 4, kv_dim = 0, half = 2;

    Fixture(bool tied, int n_layers = 3, int n_kv_heads = 1, unsigned seed = 0)
    {
        config.dim = dim;
        config.hidden_dim = hidden;
        config.n_layers = n_layers;
        config.n_heads = dim / head_size;   // 2
        config.n_kv_heads = n_kv_heads;
        config.vocab_size = tied ? 7 : -7;  // sign = classifier sharing
        config.seq_len = 5;
        kv_dim = head_size * n_kv_heads;

        auto fill = [&](std::vector<float>& v, std::size_t n, std::size_t salt, float scale, float offset) {
            v.resize(n);
            for (std::size_t i = 0; i < n; i++) v[i] = offset + scale * synthetic::filler(i + salt + seed);
        };

        for (int l = 0; l < n_layers; l++) {
            // Each layer gets its own salt range so weights are distinct.
            const std::size_t base = 10000u * static_cast<std::size_t>(l + 1);
            std::vector<float> t;
            fill(t, dim, base + 1, 0.2f, 1.0f);              rms_att.push_back(t);
            fill(t, static_cast<std::size_t>(dim) * dim, base + 100, 0.5f, 0.0f);      wq.push_back(t);
            fill(t, static_cast<std::size_t>(kv_dim) * dim, base + 200, 0.5f, 0.0f);   wk.push_back(t);
            fill(t, static_cast<std::size_t>(kv_dim) * dim, base + 300, 0.5f, 0.0f);   wv.push_back(t);
            fill(t, static_cast<std::size_t>(dim) * dim, base + 400, 0.5f, 0.0f);      wo.push_back(t);
            fill(t, dim, base + 500, 0.2f, 1.0f);            rms_ffn.push_back(t);
            fill(t, static_cast<std::size_t>(hidden) * dim, base + 600, 0.5f, 0.0f);   w1.push_back(t);
            fill(t, static_cast<std::size_t>(dim) * hidden, base + 700, 0.5f, 0.0f);   w2.push_back(t);
            fill(t, static_cast<std::size_t>(hidden) * dim, base + 800, 0.5f, 0.0f);   w3.push_back(t);
        }

        fill(embedding, static_cast<std::size_t>(7) * dim, 900, 1.0f, 0.0f);
        fill(rms_final, dim, 950, 0.3f, 1.0f);
        fill(wcls_untied, static_cast<std::size_t>(7) * dim, 970, 0.5f, 0.0f);

        cos_table.resize(static_cast<std::size_t>(config.seq_len) * half);
        sin_table.resize(cos_table.size());
        for (int pos = 0; pos < config.seq_len; pos++) {
            for (int p = 0; p < half; p++) {
                const double angle = pos / std::pow(10000.0, (2.0 * p) / head_size);
                cos_table[pos * half + p] = static_cast<float>(std::cos(angle));
                sin_table[pos * half + p] = static_cast<float>(std::sin(angle));
            }
        }

        rebuild_flat();
    }

    // Appends the per-layer vectors into the checkpoint layout and points
    // `weights` at them.  Call again after editing per-layer weights.
    void rebuild_flat()
    {
        auto flatten = [](const std::vector<std::vector<float>>& per_layer, std::vector<float>& flat) {
            flat.clear();
            for (const auto& v : per_layer) flat.insert(flat.end(), v.begin(), v.end());
        };
        flatten(rms_att, flat_rms_att); flatten(wq, flat_wq); flatten(wk, flat_wk);
        flatten(wv, flat_wv); flatten(wo, flat_wo); flatten(rms_ffn, flat_rms_ffn);
        flatten(w1, flat_w1); flatten(w2, flat_w2); flatten(w3, flat_w3);

        weights.token_embedding_table = embedding.data();
        weights.rms_att_weight = flat_rms_att.data();
        weights.wq = flat_wq.data(); weights.wk = flat_wk.data(); weights.wv = flat_wv.data();
        weights.wo = flat_wo.data();
        weights.rms_ffn_weight = flat_rms_ffn.data();
        weights.w1 = flat_w1.data(); weights.w2 = flat_w2.data(); weights.w3 = flat_w3.data();
        weights.rms_final_weight = rms_final.data();
        weights.freq_cis_real = cos_table.data();
        weights.freq_cis_imag = sin_table.data();
        weights.wcls = config.vocab_size > 0 ? embedding.data() : wcls_untied.data();
    }

    int vocab_count() const { return 7; }
    const std::vector<float>& classifier() const { return config.vocab_size > 0 ? embedding : wcls_untied; }
};

// Oracle output for one position.
struct RefPosition {
    std::vector<std::vector<double>> layer_outputs;   // [n_layers][dim]
    std::vector<std::vector<double>> k_rows, v_rows;  // [n_layers][kv_dim] stored this step
    std::vector<double> final_norm;                   // [dim]
    std::vector<double> logits;                       // [vocab]
};

inline std::vector<RefPosition> reference_forward(const Fixture& f, const std::vector<int>& tokens, double eps = 1e-5)
{
    const int dim = f.dim, hidden = f.hidden, hs = f.head_size, kv_dim = f.kv_dim;
    const int n_heads = f.config.n_heads, n_kv = f.config.n_kv_heads, L = f.config.n_layers;
    const int qpk = n_heads / n_kv;

    // Per-layer caches: rows appended per position.
    std::vector<std::vector<std::vector<double>>> k_cache(L), v_cache(L);
    std::vector<RefPosition> out;

    for (int pos = 0; pos < static_cast<int>(tokens.size()); pos++) {
        RefPosition rp;
        std::vector<double> x(f.embedding.begin() + static_cast<std::size_t>(tokens[pos]) * dim,
                              f.embedding.begin() + static_cast<std::size_t>(tokens[pos] + 1) * dim);

        for (int l = 0; l < L; l++) {
            std::vector<double> xn = synthetic::ref_rmsnorm(x, f.rms_att[l].data(), eps);
            std::vector<double> q = synthetic::ref_matvec(f.wq[l].data(), xn, dim, dim);
            std::vector<double> k = synthetic::ref_matvec(f.wk[l].data(), xn, kv_dim, dim);
            std::vector<double> v = synthetic::ref_matvec(f.wv[l].data(), xn, kv_dim, dim);
            auto rotate = [&](std::vector<double>& vec, int heads) {
                for (int h = 0; h < heads; h++)
                    for (int p = 0; p < hs / 2; p++) {
                        const double c = f.cos_table[pos * f.half + p], s = f.sin_table[pos * f.half + p];
                        const double a = vec[h * hs + 2 * p], b = vec[h * hs + 2 * p + 1];
                        vec[h * hs + 2 * p] = a * c - b * s;
                        vec[h * hs + 2 * p + 1] = a * s + b * c;
                    }
            };
            rotate(q, n_heads);
            rotate(k, n_kv);
            k_cache[l].push_back(k);
            v_cache[l].push_back(v);
            rp.k_rows.push_back(k);
            rp.v_rows.push_back(v);

            std::vector<double> att(dim, 0.0);
            for (int h = 0; h < n_heads; h++) {
                const int kv = h / qpk;
                std::vector<double> sc(pos + 1);
                double m = -1e300;
                for (int t = 0; t <= pos; t++) {
                    double d = 0;
                    for (int i = 0; i < hs; i++) d += q[h * hs + i] * k_cache[l][t][kv * hs + i];
                    sc[t] = d / std::sqrt(static_cast<double>(hs));
                    m = std::max(m, sc[t]);
                }
                double sum = 0;
                for (double& e : sc) { e = std::exp(e - m); sum += e; }
                for (int t = 0; t <= pos; t++)
                    for (int i = 0; i < hs; i++) att[h * hs + i] += (sc[t] / sum) * v_cache[l][t][kv * hs + i];
            }
            std::vector<double> proj = synthetic::ref_matvec(f.wo[l].data(), att, dim, dim);
            std::vector<double> x_att(dim);
            for (int i = 0; i < dim; i++) x_att[i] = x[i] + proj[i];
            std::vector<double> fn = synthetic::ref_rmsnorm(x_att, f.rms_ffn[l].data(), eps);
            std::vector<double> h1 = synthetic::ref_matvec(f.w1[l].data(), fn, hidden, dim);
            std::vector<double> h3 = synthetic::ref_matvec(f.w3[l].data(), fn, hidden, dim);
            std::vector<double> g(hidden);
            for (int i = 0; i < hidden; i++) g[i] = h1[i] / (1.0 + std::exp(-h1[i])) * h3[i];
            std::vector<double> ffn = synthetic::ref_matvec(f.w2[l].data(), g, dim, hidden);
            for (int i = 0; i < dim; i++) x[i] = x_att[i] + ffn[i];
            rp.layer_outputs.push_back(x);
        }

        rp.final_norm = synthetic::ref_rmsnorm(x, f.rms_final.data(), eps);
        rp.logits = synthetic::ref_matvec(f.classifier().data(), rp.final_norm, f.vocab_count(), dim);
        out.push_back(rp);
    }
    return out;
}

} // namespace synthetic_model

#endif
