#ifndef MODEL_FORWARD_CUDA_HPP
#define MODEL_FORWARD_CUDA_HPP

// Whole-model single-token forward pass on the GPU.  See
// model_forward.hpp for the sequence of operations and the lifetime rule
// (the mapped Checkpoint must outlive the model; the constructor reads
// the host weights once, after which only device copies are used).

#include "device_buffer.hpp"
#include "layer_forward_cuda.hpp"
#include "model_forward.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <vector>

class GpuModel {
public:
    // Uploads every weight once.  keep_layer_outputs allocates one extra
    // [dim] buffer per layer and snapshots each layer's output into it
    // (device-to-device, stream-ordered) so tests can read them back.
    GpuModel(const ModelConfig& config, const ModelWeights& weights, bool keep_layer_outputs = false);

    const ModelShape& shape() const { return shape_; }
    int position() const { return position_; }
    bool needs_reset() const { return needs_reset_; }
    bool classifier_is_tied() const { return classifier_ == nullptr; }

    // Issues every kernel for one token on the default stream and returns
    // the first launch error.  Never synchronises.
    //   cudaErrorInvalidValue  : bad token/pos or needs_reset(); nothing changed
    //   other error            : a launch failed part-way; needs_reset() is now true
    // On success position() advances by one.  Call synchronize() before
    // reading any result buffer.
    cudaError_t forward(std::int64_t token, int pos);

    // cudaDeviceSynchronize; an execution error marks the model as
    // needing reset because the caches may hold garbage.
    cudaError_t synchronize();

    // Start a new sequence: every layer's cache is reset together.
    void reset();

    const DeviceBuffer& logits() const { return logits_; }
    const DeviceBuffer& final_normalized() const { return final_norm_; }
    const DeviceBuffer& layer_output(int layer) const;   // requires keep_layer_outputs
    const DeviceKvCache& layer_cache(int layer) const { return caches_[layer]; }

private:
    const float* d_wcls() const { return classifier_ ? classifier_->data() : embedding_.data(); }

    ModelShape shape_;
    DeviceBuffer embedding_;                    // [vocab_count, dim]
    std::unique_ptr<DeviceBuffer> classifier_;  // only when the classifier is untied
    DeviceBuffer rms_final_;
    DeviceBuffer cos_table_, sin_table_;        // [seq_len, head_size / 2], shared by all layers
    std::vector<DeviceLayerWeights> layers_;
    std::vector<DeviceKvCache> caches_;         // one per layer
    DeviceLayerScratch scratch_;                // shared by all layers
    DeviceBuffer x_;                            // embedding of the current token
    DeviceBuffer final_norm_;
    DeviceBuffer logits_;
    std::vector<DeviceBuffer> layer_outputs_;   // snapshots, only if requested
    int position_ = 0;
    bool needs_reset_ = false;
};

#endif
