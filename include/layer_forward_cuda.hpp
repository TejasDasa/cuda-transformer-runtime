#ifndef LAYER_FORWARD_CUDA_HPP
#define LAYER_FORWARD_CUDA_HPP

// GPU counterpart of layer_forward.hpp.  Same forward sequence; every
// buffer here is device memory owned by a DeviceBuffer.

#include "device_buffer.hpp"
#include "kv_cache.hpp"
#include "layer_forward.hpp"

#include <cuda_runtime.h>

// One layer's weights resident on the GPU.  Allocated once from the
// shape; upload() copies the host slices in.
struct DeviceLayerWeights {
    DeviceBuffer rms_att, wq, wk, wv, wo, rms_ffn, w1, w2, w3;

    explicit DeviceLayerWeights(const LayerShape& shape);
    void upload(const LayerWeights& host, const LayerShape& shape);
};

// Per-sequence GPU state: working vectors for one token step, the KV
// caches, and the attention scratch.  Everything is allocated once and
// reused for every token.  Buffer roles and lifetimes:
//   xn, q, k, v, att_out, projected : overwritten each step
//   x_att      : holds residual 1; read by the FFN norm and residual 2
//   ffn_norm, h1, h3, gated, ffn_out : overwritten each step
//   x_out      : the layer output for this step
//   k_cache, v_cache : [seq_len, kv_dim], persist across the sequence
//   scores, probs    : [n_heads, seq_len] attention scratch
struct DeviceLayerState {
    DeviceBuffer xn, q, k, v, att_out, projected, x_att;
    DeviceBuffer ffn_norm, h1, h3, gated, ffn_out, x_out;
    DeviceBuffer k_cache, v_cache;
    DeviceBuffer scores, probs;
    KvCacheState cache;

    explicit DeviceLayerState(const LayerShape& shape);

    // Forget the sequence (the buffers keep their bytes).
    void reset();
};

// Issues every kernel for one token step on the default stream and
// returns the first launch error, or cudaErrorInvalidValue if the shape
// is bad or pos is not the next cache row.  Never synchronises: call
// cudaDeviceSynchronize afterwards to observe execution errors and to
// read state.x_out.
//
// d_x_in may be state.x_out.data() (chaining layers): x_in is only read
// by the RMSNorm and residual-1 kernels, both of which run before x_out
// is written.
cudaError_t layer_forward_cuda(
    DeviceLayerState& state,
    const DeviceLayerWeights& w,
    const LayerShape& shape,
    const float* d_x_in,
    const float* d_cos_row,
    const float* d_sin_row,
    int pos,
    float epsilon
);

#endif
