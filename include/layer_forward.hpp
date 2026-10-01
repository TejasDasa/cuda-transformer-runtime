#ifndef LAYER_FORWARD_HPP
#define LAYER_FORWARD_HPP

// One transformer layer, CPU side.  The GPU counterpart lives in
// layer_forward_cuda.hpp and follows the same structure.
//
// Forward sequence for the token at position pos, input x_in [dim]:
//   xn        = RMSNorm(x_in, rms_att)
//   q, k, v   = Wq xn, Wk xn, Wv xn
//   q, k      = RoPE(q, pos), RoPE(k, pos)
//   K[pos]=k, V[pos]=v
//   att_out   = attention(q, K[0..pos], V[0..pos])      [dim]
//   projected = Wo att_out                               [dim]
//   x_att     = x_in + projected           (residual 1, uses x_in)
//   ffn_norm  = RMSNorm(x_att, rms_ffn)
//   h1, h3    = W1 ffn_norm, W3 ffn_norm                 [hidden_dim]
//   gated     = SiLU(h1) * h3                            [hidden_dim]
//   ffn_out   = W2 gated                                 [dim]
//   x_out     = x_att + ffn_out            (residual 2, uses x_att)

#include "kv_cache.hpp"
#include "model_config.hpp"
#include "model_weights.hpp"

#include <cstddef>
#include <vector>

struct LayerShape {
    int dim = 0;
    int hidden_dim = 0;
    int n_heads = 0;
    int n_kv_heads = 0;
    int head_size = 0;
    int kv_dim = 0;
    int seq_len = 0;

    static LayerShape from_config(const ModelConfig& config);

    // Positive sizes, even head_size, n_heads divisible by n_kv_heads.
    bool valid() const;
};

// Pointers to ONE layer's weight slices (row-major):
//   rms_att [dim]        wq [dim, dim]   wk [kv_dim, dim]   wv [kv_dim, dim]
//   wo      [dim, dim]
//   rms_ffn [dim]        w1 [hidden_dim, dim]   w3 [hidden_dim, dim]
//   w2      [dim, hidden_dim]
struct LayerWeights {
    const float* rms_att = nullptr;
    const float* wq = nullptr;
    const float* wk = nullptr;
    const float* wv = nullptr;
    const float* wo = nullptr;
    const float* rms_ffn = nullptr;
    const float* w1 = nullptr;
    const float* w2 = nullptr;
    const float* w3 = nullptr;
};

// Element counts of each slice, shared by host and device allocation.
struct LayerWeightCounts {
    std::size_t rms = 0;   // dim
    std::size_t wq = 0;    // dim * dim
    std::size_t wk = 0;    // kv_dim * dim   (also wv)
    std::size_t wo = 0;    // dim * dim
    std::size_t w1 = 0;    // hidden_dim * dim   (also w3)
    std::size_t w2 = 0;    // dim * hidden_dim
};
LayerWeightCounts layer_weight_counts(const LayerShape& shape);

// Selects layer `layer` from the mapped checkpoint tensors, which store
// all layers back to back.  Layer 0 is the mapped pointer itself.
LayerWeights select_layer_weights(const ModelWeights& weights, const LayerShape& shape, int layer);

// Working vectors for one token step, kept separately so tests can
// inspect every intermediate.  One scratch set can serve every layer in
// turn because nothing in it outlives a single layer step.
struct CpuLayerScratch {
    std::vector<float> xn, q, k, v, att_out, projected, x_att;
    std::vector<float> ffn_norm, h1, h3, gated, ffn_out, x_out;
    std::vector<float> scores, probs;      // [n_heads, seq_len]

    explicit CpuLayerScratch(const LayerShape& shape);
};

// One layer's KV caches, [seq_len, kv_dim] each, plus the validity
// tracker.  Every layer of a model owns one of these.
struct CpuKvCache {
    std::vector<float> k_cache, v_cache;
    KvCacheState cache;

    explicit CpuKvCache(const LayerShape& shape);

    // Forget the sequence.  Cache bytes stay but none count as valid.
    void reset();
};

// Convenience bundle for single-layer tests: scratch and cache together.
struct CpuLayerState : CpuLayerScratch, CpuKvCache {
    explicit CpuLayerState(const LayerShape& shape)
        : CpuLayerScratch(shape), CpuKvCache(shape)
    {
    }
};

// Runs one token through one layer.  cos_row / sin_row point at the
// RoPE table row for `pos`.  Returns false without computing anything
// if the shape is invalid or pos is not the next cache row.
bool layer_forward_cpu(
    CpuLayerScratch& scratch,
    CpuKvCache& cache,
    const LayerWeights& w,
    const LayerShape& shape,
    const float* x_in,
    const float* cos_row,
    const float* sin_row,
    int pos,
    float epsilon
);

inline bool layer_forward_cpu(
    CpuLayerState& state,
    const LayerWeights& w,
    const LayerShape& shape,
    const float* x_in,
    const float* cos_row,
    const float* sin_row,
    int pos,
    float epsilon
)
{
    return layer_forward_cpu(state, state, w, shape, x_in, cos_row, sin_row, pos, epsilon);
}

#endif
