#include "model_forward.hpp"

#include "cpu_ops.hpp"

#include <algorithm>
#include <climits>
#include <stdexcept>

std::size_t checked_mul(std::size_t a, std::size_t b)
{
    if (a != 0 && b > static_cast<std::size_t>(-1) / a) {
        throw std::overflow_error("element count overflows size_t");
    }
    return a * b;
}

ModelShape ModelShape::from_config(const ModelConfig& config)
{
    ModelShape s;
    s.layer = LayerShape::from_config(config);
    s.n_layers = config.n_layers;
    // Widen before negating so INT32_MIN cannot overflow; the sign only
    // says whether the classifier shares the embedding table.
    std::int64_t vocab = config.vocab_size;
    s.shared_classifier = vocab > 0;
    s.vocab_count = vocab < 0 ? -vocab : vocab;
    return s;
}

bool ModelShape::valid() const
{
    return layer.valid() && n_layers > 0 && vocab_count > 0 && vocab_count <= INT_MAX;
}

ModelWeightViews select_model_weights(const ModelWeights& weights, const ModelShape& shape)
{
    if (!shape.valid()) {
        throw std::invalid_argument("invalid model shape");
    }

    // Make sure every whole-tensor extent we will index is representable.
    const LayerWeightCounts c = layer_weight_counts(shape.layer);
    const std::size_t n = static_cast<std::size_t>(shape.n_layers);
    checked_mul(n, c.rms);
    checked_mul(n, c.wq);
    checked_mul(n, c.wk);
    checked_mul(n, c.wo);
    checked_mul(n, c.w1);
    checked_mul(n, c.w2);
    checked_mul(static_cast<std::size_t>(shape.vocab_count), static_cast<std::size_t>(shape.layer.dim));
    checked_mul(static_cast<std::size_t>(shape.layer.seq_len), static_cast<std::size_t>(shape.layer.head_size / 2));

    ModelWeightViews v;
    v.layers.reserve(n);
    for (int l = 0; l < shape.n_layers; l++) {
        v.layers.push_back(select_layer_weights(weights, shape.layer, l));
    }
    v.embedding = weights.token_embedding_table;
    v.rms_final = weights.rms_final_weight;
    v.wcls = weights.wcls;   // the Checkpoint already pointed this at the embedding when tied
    v.cos_table = weights.freq_cis_real;
    v.sin_table = weights.freq_cis_imag;
    return v;
}

CpuModel::CpuModel(const ModelConfig& config, const ModelWeights& weights)
    : shape_(ModelShape::from_config(config)),
      w_(select_model_weights(weights, shape_)),
      scratch_(shape_.layer),
      layer_outputs_(static_cast<std::size_t>(shape_.n_layers), std::vector<float>(shape_.layer.dim)),
      x_(shape_.layer.dim),
      final_norm_(shape_.layer.dim),
      logits_(static_cast<std::size_t>(shape_.vocab_count))
{
    caches_.reserve(static_cast<std::size_t>(shape_.n_layers));
    for (int l = 0; l < shape_.n_layers; l++) {
        caches_.emplace_back(shape_.layer);
    }
}

bool CpuModel::forward(std::int64_t token, int pos)
{
    const LayerShape& ls = shape_.layer;

    // Input validation changes nothing, so it does not poison the model.
    if (needs_reset_ || token < 0 || token >= shape_.vocab_count ||
        pos != position_ || pos >= ls.seq_len) {
        return false;
    }

    const int half = ls.head_size / 2;
    const std::size_t row_offset = static_cast<std::size_t>(pos) * half;
    const float* cos_row = w_.cos_table + row_offset;
    const float* sin_row = w_.sin_table + row_offset;

    const float* embedding_row = w_.embedding + static_cast<std::size_t>(token) * ls.dim;
    std::copy(embedding_row, embedding_row + ls.dim, x_.begin());

    // Layer l reads the previous layer's output snapshot (layer 0 reads
    // the embedding) and writes scratch_.x_out, which is then snapshotted.
    const float* x_in = x_.data();
    for (int l = 0; l < shape_.n_layers; l++) {
        if (!layer_forward_cpu(scratch_, caches_[l], w_.layers[l], ls, x_in, cos_row, sin_row, pos, kModelRmsEpsilon)) {
            needs_reset_ = true;   // earlier layers advanced, this one did not
            return false;
        }
        layer_outputs_[l] = scratch_.x_out;
        x_in = layer_outputs_[l].data();
    }

    rmsnorm(final_norm_.data(), x_in, w_.rms_final, ls.dim, kModelRmsEpsilon);
    // wcls is [vocab_count, dim]: one logit per row.
    matvec(logits_.data(), w_.wcls, final_norm_.data(), static_cast<int>(shape_.vocab_count), ls.dim);

    position_++;
    return true;
}

void CpuModel::reset()
{
    for (CpuKvCache& c : caches_) {
        c.reset();
    }
    position_ = 0;
    needs_reset_ = false;
}
