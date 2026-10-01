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

// Working device vectors for one token step.  One scratch set serves
// every layer in turn: nothing here outlives a single layer step.
//   xn, q, k, v, att_out, projected : overwritten each step
//   x_att      : holds residual 1; read by the FFN norm and residual 2
//   ffn_norm, h1, h3, gated, ffn_out : overwritten each step
//   x_out      : the layer output for this step
//   scores, probs    : [n_heads, seq_len] attention scratch
struct DeviceLayerScratch {
    DeviceBuffer xn, q, k, v, att_out, projected, x_att;
    DeviceBuffer ffn_norm, h1, h3, gated, ffn_out, x_out;
    DeviceBuffer scores, probs;

    explicit DeviceLayerScratch(const LayerShape& shape);
};

// One layer's KV caches, [seq_len, kv_dim] each, resident across the
// sequence, plus the validity tracker.
struct DeviceKvCache {
    DeviceBuffer k_cache, v_cache;
    KvCacheState cache;

    explicit DeviceKvCache(const LayerShape& shape);

    // Forget the sequence (the buffers keep their bytes).
    void reset();
};

// Convenience bundle for single-layer tests.
struct DeviceLayerState : DeviceLayerScratch, DeviceKvCache {
    explicit DeviceLayerState(const LayerShape& shape)
        : DeviceLayerScratch(shape), DeviceKvCache(shape)
    {
    }
};

// Issues every kernel for one token step on the default stream and
// returns the first launch error, or cudaErrorInvalidValue if the shape
// is bad or pos is not the next cache row.  Never synchronises: call
// cudaDeviceSynchronize afterwards to observe execution errors and to
// read scratch.x_out.
//
// d_x_in may be scratch.x_out.data() (chaining layers): x_in is only
// read by the RMSNorm and residual-1 kernels, both of which run before
// x_out is written.
cudaError_t layer_forward_cuda(
    DeviceLayerScratch& scratch,
    DeviceKvCache& cache,
    const DeviceLayerWeights& w,
    const LayerShape& shape,
    const float* d_x_in,
    const float* d_cos_row,
    const float* d_sin_row,
    int pos,
    float epsilon
);

inline cudaError_t layer_forward_cuda(
    DeviceLayerState& state,
    const DeviceLayerWeights& w,
    const LayerShape& shape,
    const float* d_x_in,
    const float* d_cos_row,
    const float* d_sin_row,
    int pos,
    float epsilon
)
{
    return layer_forward_cuda(state, state, w, shape, d_x_in, d_cos_row, d_sin_row, pos, epsilon);
}

#endif
