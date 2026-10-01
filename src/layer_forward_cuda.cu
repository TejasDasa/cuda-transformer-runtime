#include "layer_forward_cuda.hpp"

#include "cuda_ops.hpp"

namespace {

std::size_t cache_floats(const LayerShape& shape)
{
    return static_cast<std::size_t>(shape.seq_len) * static_cast<std::size_t>(shape.kv_dim);
}

std::size_t scratch_floats(const LayerShape& shape)
{
    return static_cast<std::size_t>(shape.n_heads) * static_cast<std::size_t>(shape.seq_len);
}

} // namespace

DeviceLayerWeights::DeviceLayerWeights(const LayerShape& shape)
    : rms_att(layer_weight_counts(shape).rms),
      wq(layer_weight_counts(shape).wq),
      wk(layer_weight_counts(shape).wk),
      wv(layer_weight_counts(shape).wk),
      wo(layer_weight_counts(shape).wo),
      rms_ffn(layer_weight_counts(shape).rms),
      w1(layer_weight_counts(shape).w1),
      w2(layer_weight_counts(shape).w2),
      w3(layer_weight_counts(shape).w1)
{
}

void DeviceLayerWeights::upload(const LayerWeights& host, const LayerShape&)
{
    rms_att.upload(host.rms_att, "upload rms_att");
    wq.upload(host.wq, "upload wq");
    wk.upload(host.wk, "upload wk");
    wv.upload(host.wv, "upload wv");
    wo.upload(host.wo, "upload wo");
    rms_ffn.upload(host.rms_ffn, "upload rms_ffn");
    w1.upload(host.w1, "upload w1");
    w2.upload(host.w2, "upload w2");
    w3.upload(host.w3, "upload w3");
}

DeviceLayerScratch::DeviceLayerScratch(const LayerShape& shape)
    : xn(shape.dim), q(shape.dim), k(shape.kv_dim), v(shape.kv_dim),
      att_out(shape.dim), projected(shape.dim), x_att(shape.dim),
      ffn_norm(shape.dim), h1(shape.hidden_dim), h3(shape.hidden_dim),
      gated(shape.hidden_dim), ffn_out(shape.dim), x_out(shape.dim),
      scores(scratch_floats(shape)), probs(scratch_floats(shape))
{
    // Zero the attention scratch once so diagnostics never read
    // uninitialised device memory beyond the causal prefix.
    check_cuda(cudaMemset(scores.data(), 0, scores.bytes()), "clear scores");
    check_cuda(cudaMemset(probs.data(), 0, probs.bytes()), "clear probs");
}

DeviceKvCache::DeviceKvCache(const LayerShape& shape)
    : k_cache(cache_floats(shape)), v_cache(cache_floats(shape)),
      cache(shape.seq_len, shape.kv_dim)
{
    check_cuda(cudaMemset(k_cache.data(), 0, k_cache.bytes()), "clear k cache");
    check_cuda(cudaMemset(v_cache.data(), 0, v_cache.bytes()), "clear v cache");
}

void DeviceKvCache::reset()
{
    cache.reset();
}

// Every launch below goes to the default stream, so each kernel starts
// only after the previous one finished; that is the only ordering the
// data dependencies need.
cudaError_t layer_forward_cuda(
    DeviceLayerScratch& s,
    DeviceKvCache& c,
    const DeviceLayerWeights& w,
    const LayerShape& shape,
    const float* d_x_in,
    const float* d_cos_row,
    const float* d_sin_row,
    int pos,
    float epsilon
)
{
    if (!shape.valid() || !c.cache.accepts(pos)) {
        return cudaErrorInvalidValue;
    }

    const int dim = shape.dim;
    const int kv_dim = shape.kv_dim;
    const int hidden = shape.hidden_dim;
    cudaError_t status;

    // ----- attention block -----
    status = rmsnorm_cuda(s.xn.data(), d_x_in, w.rms_att.data(), dim, epsilon);
    if (status != cudaSuccess) return status;
    status = matvec_cuda(s.q.data(), w.wq.data(), s.xn.data(), dim, dim);
    if (status != cudaSuccess) return status;
    status = matvec_cuda(s.k.data(), w.wk.data(), s.xn.data(), kv_dim, dim);
    if (status != cudaSuccess) return status;
    status = matvec_cuda(s.v.data(), w.wv.data(), s.xn.data(), kv_dim, dim);
    if (status != cudaSuccess) return status;
    status = rope_cuda(s.q.data(), shape.n_heads, shape.head_size, d_cos_row, d_sin_row);
    if (status != cudaSuccess) return status;
    status = rope_cuda(s.k.data(), shape.n_kv_heads, shape.head_size, d_cos_row, d_sin_row);
    if (status != cudaSuccess) return status;
    status = kv_cache_store_cuda(c.k_cache.data(), s.k.data(), kv_dim, shape.seq_len, pos);
    if (status != cudaSuccess) return status;
    status = kv_cache_store_cuda(c.v_cache.data(), s.v.data(), kv_dim, shape.seq_len, pos);
    if (status != cudaSuccess) return status;
    c.cache.advance();   // row pos is queued for writing before any read of it

    status = attention_cuda(s.att_out.data(), s.scores.data(), s.probs.data(),
                            s.q.data(), c.k_cache.data(), c.v_cache.data(),
                            shape.n_heads, shape.n_kv_heads, shape.head_size, shape.seq_len, pos);
    if (status != cudaSuccess) return status;

    // Wo [dim, dim] x att_out -> projected; residual 1 uses the original
    // input d_x_in (still intact: nothing has written to it).
    status = matvec_cuda(s.projected.data(), w.wo.data(), s.att_out.data(), dim, dim);
    if (status != cudaSuccess) return status;
    status = add_vectors_cuda(s.x_att.data(), d_x_in, s.projected.data(), dim);
    if (status != cudaSuccess) return status;

    // ----- feed-forward block (SwiGLU) -----
    status = rmsnorm_cuda(s.ffn_norm.data(), s.x_att.data(), w.rms_ffn.data(), dim, epsilon);
    if (status != cudaSuccess) return status;
    status = matvec_cuda(s.h1.data(), w.w1.data(), s.ffn_norm.data(), hidden, dim);
    if (status != cudaSuccess) return status;
    status = matvec_cuda(s.h3.data(), w.w3.data(), s.ffn_norm.data(), hidden, dim);
    if (status != cudaSuccess) return status;
    status = silu_gate_cuda(s.gated.data(), s.h1.data(), s.h3.data(), hidden);
    if (status != cudaSuccess) return status;
    status = matvec_cuda(s.ffn_out.data(), w.w2.data(), s.gated.data(), dim, hidden);
    if (status != cudaSuccess) return status;

    // Residual 2 uses x_att.  x_out is written last, which is what makes
    // d_x_in == x_out.data() safe for chaining.
    return add_vectors_cuda(s.x_out.data(), s.x_att.data(), s.ffn_out.data(), dim);
}
