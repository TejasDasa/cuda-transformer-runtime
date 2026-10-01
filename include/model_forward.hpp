#ifndef MODEL_FORWARD_HPP
#define MODEL_FORWARD_HPP

// Whole-model single-token forward pass, CPU side.  The GPU counterpart
// is in model_forward_cuda.hpp and mirrors this interface.
//
//   x = token_embedding_table[token]
//   for layer in [0, n_layers):  x = transformer_layer(x, layer, pos)
//   final_normalized = RMSNorm(x, rms_final_weight)
//   logits = wcls x                      [vocab_count]   (raw, no softmax)
//
// Lifetime: everything here holds raw pointers into the memory-mapped
// Checkpoint.  The Checkpoint must outlive any CpuModel / GpuModel built
// from its config() and weights().

#include "layer_forward.hpp"
#include "model_config.hpp"
#include "model_weights.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

constexpr float kModelRmsEpsilon = 1e-5f;

struct ModelShape {
    LayerShape layer;
    int n_layers = 0;
    std::int64_t vocab_count = 0;       // |vocab_size|
    bool shared_classifier = false;     // vocab_size > 0: wcls IS the embedding table

    static ModelShape from_config(const ModelConfig& config);

    // Valid layer shape, n_layers > 0, 0 < vocab_count <= INT_MAX.
    bool valid() const;
};

// Pointers to each layer's slices and to the model-wide tensors.
//
// Checkpoint tensors are stored tensor-by-tensor with all layers back to
// back inside each tensor, so layer l of tensor T begins at
//     T_base + l * per_layer_count(T)
// (see layer_weight_counts / select_layer_weights).  rms_final, wcls and
// the RoPE tables are model-wide and have no layer offset.
struct ModelWeightViews {
    std::vector<LayerWeights> layers;
    const float* embedding = nullptr;   // [vocab_count, dim]
    const float* rms_final = nullptr;   // [dim]
    const float* wcls = nullptr;        // [vocab_count, dim]; == embedding when tied
    const float* cos_table = nullptr;   // [seq_len, head_size / 2]
    const float* sin_table = nullptr;
};

// Builds the views.  Throws std::overflow_error if any element count
// overflows size_t and std::invalid_argument if the shape is invalid.
ModelWeightViews select_model_weights(const ModelWeights& weights, const ModelShape& shape);

// a * b with overflow detection (throws std::overflow_error).
std::size_t checked_mul(std::size_t a, std::size_t b);

class CpuModel {
public:
    CpuModel(const ModelConfig& config, const ModelWeights& weights);

    const ModelShape& shape() const { return shape_; }

    // Number of completed positions; also the only position forward()
    // will accept next.
    int position() const { return position_; }

    // True after a forward failed part-way: the per-layer caches may
    // disagree about the sequence length, so forward() refuses to run
    // until reset() is called.
    bool needs_reset() const { return needs_reset_; }

    // Runs one token.  Returns false if token or pos is invalid (nothing
    // changes), if needs_reset(), or if a layer failed (needs_reset()
    // becomes true).  On success position() advances by one.
    bool forward(std::int64_t token, int pos);

    // Start a new sequence: every layer's cache is reset together.
    void reset();

    const std::vector<float>& logits() const { return logits_; }
    const std::vector<float>& final_normalized() const { return final_norm_; }
    const std::vector<float>& layer_output(int layer) const { return layer_outputs_[layer]; }
    const CpuKvCache& layer_cache(int layer) const { return caches_[layer]; }

private:
    ModelShape shape_;
    ModelWeightViews w_;
    CpuLayerScratch scratch_;             // shared by all layers
    std::vector<CpuKvCache> caches_;      // one per layer
    std::vector<std::vector<float>> layer_outputs_;   // [n_layers][dim] snapshots
    std::vector<float> x_;                // embedding of the current token
    std::vector<float> final_norm_;
    std::vector<float> logits_;
    int position_ = 0;
    bool needs_reset_ = false;
};

#endif
