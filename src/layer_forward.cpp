#include "layer_forward.hpp"

#include "attention_shape.hpp"
#include "cpu_ops.hpp"

LayerShape LayerShape::from_config(const ModelConfig& config)
{
    LayerShape s;
    s.dim = config.dim;
    s.hidden_dim = config.hidden_dim;
    s.n_heads = config.n_heads;
    s.n_kv_heads = config.n_kv_heads;
    s.head_size = (config.n_heads > 0) ? config.dim / config.n_heads : 0;
    s.kv_dim = s.head_size * config.n_kv_heads;
    s.seq_len = config.seq_len;
    return s;
}

bool LayerShape::valid() const
{
    return dim > 0 && hidden_dim > 0 && n_heads > 0 && n_kv_heads > 0 &&
           head_size > 0 && head_size % 2 == 0 &&
           head_size * n_heads == dim &&
           kv_dim == head_size * n_kv_heads &&
           seq_len > 0 &&
           attention_shape_valid(n_heads, n_kv_heads, head_size, seq_len, 0);
}

LayerWeightCounts layer_weight_counts(const LayerShape& shape)
{
    LayerWeightCounts c;
    const std::size_t dim = static_cast<std::size_t>(shape.dim);
    const std::size_t hidden = static_cast<std::size_t>(shape.hidden_dim);
    const std::size_t kv_dim = static_cast<std::size_t>(shape.kv_dim);
    c.rms = dim;
    c.wq = dim * dim;
    c.wk = kv_dim * dim;
    c.wo = dim * dim;
    c.w1 = hidden * dim;
    c.w2 = dim * hidden;
    return c;
}

LayerWeights select_layer_weights(const ModelWeights& weights, const LayerShape& shape, int layer)
{
    const LayerWeightCounts c = layer_weight_counts(shape);
    const std::size_t l = static_cast<std::size_t>(layer);
    LayerWeights w;
    w.rms_att = weights.rms_att_weight + l * c.rms;
    w.wq = weights.wq + l * c.wq;
    w.wk = weights.wk + l * c.wk;
    w.wv = weights.wv + l * c.wk;
    w.wo = weights.wo + l * c.wo;
    w.rms_ffn = weights.rms_ffn_weight + l * c.rms;
    w.w1 = weights.w1 + l * c.w1;
    w.w2 = weights.w2 + l * c.w2;
    w.w3 = weights.w3 + l * c.w1;
    return w;
}

CpuLayerState::CpuLayerState(const LayerShape& shape)
    : xn(shape.dim), q(shape.dim), k(shape.kv_dim), v(shape.kv_dim),
      att_out(shape.dim), projected(shape.dim), x_att(shape.dim),
      ffn_norm(shape.dim), h1(shape.hidden_dim), h3(shape.hidden_dim),
      gated(shape.hidden_dim), ffn_out(shape.dim), x_out(shape.dim),
      k_cache(static_cast<std::size_t>(shape.seq_len) * shape.kv_dim, 0.0f),
      v_cache(static_cast<std::size_t>(shape.seq_len) * shape.kv_dim, 0.0f),
      scores(static_cast<std::size_t>(shape.n_heads) * shape.seq_len, 0.0f),
      probs(static_cast<std::size_t>(shape.n_heads) * shape.seq_len, 0.0f),
      cache(shape.seq_len, shape.kv_dim)
{
}

void CpuLayerState::reset()
{
    cache.reset();
}

bool layer_forward_cpu(
    CpuLayerState& s,
    const LayerWeights& w,
    const LayerShape& shape,
    const float* x_in,
    const float* cos_row,
    const float* sin_row,
    int pos,
    float epsilon
)
{
    if (!shape.valid() || !s.cache.accepts(pos)) {
        return false;
    }

    const int dim = shape.dim;
    const int kv_dim = shape.kv_dim;
    const int hidden = shape.hidden_dim;

    // ----- attention block -----
    rmsnorm(s.xn.data(), x_in, w.rms_att, dim, epsilon);
    matvec(s.q.data(), w.wq, s.xn.data(), dim, dim);
    matvec(s.k.data(), w.wk, s.xn.data(), kv_dim, dim);
    matvec(s.v.data(), w.wv, s.xn.data(), kv_dim, dim);
    rope(s.q.data(), shape.n_heads, shape.head_size, cos_row, sin_row);
    rope(s.k.data(), shape.n_kv_heads, shape.head_size, cos_row, sin_row);

    if (!kv_cache_store(s.k_cache.data(), s.k.data(), kv_dim, shape.seq_len, pos) ||
        !kv_cache_store(s.v_cache.data(), s.v.data(), kv_dim, shape.seq_len, pos)) {
        return false;
    }
    s.cache.advance();

    if (!attention(s.att_out.data(), s.scores.data(), s.probs.data(),
                   s.q.data(), s.k_cache.data(), s.v_cache.data(),
                   shape.n_heads, shape.n_kv_heads, shape.head_size, shape.seq_len, pos)) {
        return false;
    }

    // Wo [dim, dim] x att_out [dim] -> projected [dim]; residual 1 adds
    // the ORIGINAL input x_in, not the normalized xn.
    matvec(s.projected.data(), w.wo, s.att_out.data(), dim, dim);
    add_vectors(s.x_att.data(), x_in, s.projected.data(), dim);

    // ----- feed-forward block (SwiGLU) -----
    rmsnorm(s.ffn_norm.data(), s.x_att.data(), w.rms_ffn, dim, epsilon);
    matvec(s.h1.data(), w.w1, s.ffn_norm.data(), hidden, dim);   // [hidden, dim] x [dim]
    matvec(s.h3.data(), w.w3, s.ffn_norm.data(), hidden, dim);
    silu_gate(s.gated.data(), s.h1.data(), s.h3.data(), hidden);
    matvec(s.ffn_out.data(), w.w2, s.gated.data(), dim, hidden);  // [dim, hidden] x [hidden]

    // Residual 2 adds x_att (post-attention stream), not ffn_norm.
    add_vectors(s.x_out.data(), s.x_att.data(), s.ffn_out.data(), dim);
    return true;
}
