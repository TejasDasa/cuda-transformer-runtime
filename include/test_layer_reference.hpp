#ifndef TEST_LAYER_REFERENCE_HPP
#define TEST_LAYER_REFERENCE_HPP

// Test-only: a small synthetic transformer layer with hidden_dim != dim
// and fewer KV heads than query heads, plus a naive double-precision
// forward pass written independently of cpu_ops / layer_forward.  Used by
// test_layer_ops and test_layer_ops_cuda as an oracle.

#include "layer_forward.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

namespace synthetic {

// dim 8 = 2 heads x 4; 1 KV head so kv_dim = 4; hidden 12 != dim.
inline LayerShape shape()
{
    LayerShape s;
    s.dim = 8;
    s.hidden_dim = 12;
    s.n_heads = 2;
    s.n_kv_heads = 1;
    s.head_size = 4;
    s.kv_dim = 4;
    s.seq_len = 5;
    return s;
}

// Deterministic values in [-1, 1).
inline float filler(std::size_t i)
{
    return static_cast<float>((i * 2654435761u) % 2000) / 1000.0f - 1.0f;
}

// Owns host storage for one synthetic layer and exposes LayerWeights.
struct Weights {
    std::vector<float> rms_att, wq, wk, wv, wo, rms_ffn, w1, w2, w3;

    explicit Weights(const LayerShape& s, std::size_t seed = 0)
    {
        const LayerWeightCounts c = layer_weight_counts(s);
        auto make = [&](std::vector<float>& v, std::size_t n, std::size_t salt, float scale) {
            v.resize(n);
            for (std::size_t i = 0; i < n; i++) v[i] = scale * filler(i + salt + seed);
        };
        // RMS gains near 1 so norms stay well conditioned.
        make(rms_att, c.rms, 11, 0.2f); for (float& g : rms_att) g += 1.0f;
        make(rms_ffn, c.rms, 13, 0.2f); for (float& g : rms_ffn) g += 1.0f;
        make(wq, c.wq, 101, 0.5f);
        make(wk, c.wk, 211, 0.5f);
        make(wv, c.wk, 307, 0.5f);
        make(wo, c.wo, 401, 0.5f);
        make(w1, c.w1, 503, 0.5f);
        make(w2, c.w2, 601, 0.5f);
        make(w3, c.w1, 701, 0.5f);
    }

    LayerWeights view() const
    {
        LayerWeights w;
        w.rms_att = rms_att.data(); w.wq = wq.data(); w.wk = wk.data(); w.wv = wv.data();
        w.wo = wo.data(); w.rms_ffn = rms_ffn.data();
        w.w1 = w1.data(); w.w2 = w2.data(); w.w3 = w3.data();
        return w;
    }
};

// RoPE tables [seq_len, head_size/2] with the llama2.c angle formula.
struct RopeTables {
    std::vector<float> cos_table, sin_table;
    int half = 0;

    explicit RopeTables(const LayerShape& s)
        : half(s.head_size / 2)
    {
        cos_table.resize(static_cast<std::size_t>(s.seq_len) * half);
        sin_table.resize(cos_table.size());
        for (int pos = 0; pos < s.seq_len; pos++) {
            for (int p = 0; p < half; p++) {
                const double angle = pos / std::pow(10000.0, (2.0 * p) / s.head_size);
                cos_table[pos * half + p] = static_cast<float>(std::cos(angle));
                sin_table[pos * half + p] = static_cast<float>(std::sin(angle));
            }
        }
    }
    const float* cos_row(int pos) const { return cos_table.data() + static_cast<std::size_t>(pos) * half; }
    const float* sin_row(int pos) const { return sin_table.data() + static_cast<std::size_t>(pos) * half; }
};

// Deterministic per-position layer input.
inline std::vector<float> input(const LayerShape& s, int pos)
{
    std::vector<float> x(s.dim);
    for (int i = 0; i < s.dim; i++) x[i] = filler(static_cast<std::size_t>(pos) * 97 + i + 5);
    return x;
}

// All intermediates of one step in double.
struct RefStep {
    std::vector<double> att_out, projected, x_att, ffn_norm, h1, h3, gated, ffn_out, x_out;
};

inline std::vector<double> ref_rmsnorm(const std::vector<double>& x, const float* g, double eps)
{
    double ss = 0; for (double v : x) ss += v * v;
    const double scale = 1.0 / std::sqrt(ss / x.size() + eps);
    std::vector<double> out(x.size());
    for (std::size_t i = 0; i < x.size(); i++) out[i] = g[i] * x[i] * scale;
    return out;
}

inline std::vector<double> ref_matvec(const float* m, const std::vector<double>& x, int rows, int cols)
{
    std::vector<double> out(rows, 0.0);
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++) out[r] += static_cast<double>(m[r * cols + c]) * x[c];
    return out;
}

// Runs positions 0..n-1 (each with its own input) and returns every step.
// Caches are kept here in double, independently of KvCacheState.
inline std::vector<RefStep> reference_sequence(const LayerShape& s, const Weights& w, const RopeTables& t, int n, double eps)
{
    std::vector<std::vector<double>> k_rows, v_rows;
    std::vector<RefStep> steps;
    const int qpk = s.n_heads / s.n_kv_heads;

    for (int pos = 0; pos < n; pos++) {
        const std::vector<float> xf = input(s, pos);
        std::vector<double> x(xf.begin(), xf.end());
        std::vector<double> xn = ref_rmsnorm(x, w.rms_att.data(), eps);
        std::vector<double> q = ref_matvec(w.wq.data(), xn, s.dim, s.dim);
        std::vector<double> k = ref_matvec(w.wk.data(), xn, s.kv_dim, s.dim);
        std::vector<double> v = ref_matvec(w.wv.data(), xn, s.kv_dim, s.dim);
        auto rotate = [&](std::vector<double>& vec, int heads) {
            for (int h = 0; h < heads; h++)
                for (int p = 0; p < s.head_size / 2; p++) {
                    const double c = t.cos_row(pos)[p], sn = t.sin_row(pos)[p];
                    double& a = vec[h * s.head_size + 2 * p];
                    double& b = vec[h * s.head_size + 2 * p + 1];
                    const double a0 = a, b0 = b;
                    a = a0 * c - b0 * sn;
                    b = a0 * sn + b0 * c;
                }
        };
        rotate(q, s.n_heads);
        rotate(k, s.n_kv_heads);
        k_rows.push_back(k);
        v_rows.push_back(v);

        RefStep st;
        st.att_out.assign(s.dim, 0.0);
        for (int h = 0; h < s.n_heads; h++) {
            const int kv = h / qpk;
            std::vector<double> sc(pos + 1);
            double m = -1e300;
            for (int tt = 0; tt <= pos; tt++) {
                double d = 0;
                for (int i = 0; i < s.head_size; i++) d += q[h * s.head_size + i] * k_rows[tt][kv * s.head_size + i];
                sc[tt] = d / std::sqrt(static_cast<double>(s.head_size));
                m = std::max(m, sc[tt]);
            }
            double sum = 0;
            for (double& e : sc) { e = std::exp(e - m); sum += e; }
            for (int tt = 0; tt <= pos; tt++)
                for (int i = 0; i < s.head_size; i++)
                    st.att_out[h * s.head_size + i] += (sc[tt] / sum) * v_rows[tt][kv * s.head_size + i];
        }
        st.projected = ref_matvec(w.wo.data(), st.att_out, s.dim, s.dim);
        st.x_att.resize(s.dim);
        for (int i = 0; i < s.dim; i++) st.x_att[i] = x[i] + st.projected[i];
        st.ffn_norm = ref_rmsnorm(st.x_att, w.rms_ffn.data(), eps);
        st.h1 = ref_matvec(w.w1.data(), st.ffn_norm, s.hidden_dim, s.dim);
        st.h3 = ref_matvec(w.w3.data(), st.ffn_norm, s.hidden_dim, s.dim);
        st.gated.resize(s.hidden_dim);
        for (int i = 0; i < s.hidden_dim; i++) {
            const double z = st.h1[i];
            st.gated[i] = z / (1.0 + std::exp(-z)) * st.h3[i];
        }
        st.ffn_out = ref_matvec(w.w2.data(), st.gated, s.dim, s.hidden_dim);
        st.x_out.resize(s.dim);
        for (int i = 0; i < s.dim; i++) st.x_out[i] = st.x_att[i] + st.ffn_out[i];
        steps.push_back(st);
    }
    return steps;
}

} // namespace synthetic

#endif
