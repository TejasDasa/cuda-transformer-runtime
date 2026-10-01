#include "model_forward_cuda.hpp"

#include "cuda_ops.hpp"

#include <stdexcept>

GpuModel::GpuModel(const ModelConfig& config, const ModelWeights& weights, bool keep_layer_outputs)
    : shape_(ModelShape::from_config(config)),
      embedding_(checked_mul(static_cast<std::size_t>(shape_.valid() ? shape_.vocab_count : 0),
                             static_cast<std::size_t>(shape_.layer.dim))),
      rms_final_(shape_.layer.dim),
      cos_table_(checked_mul(static_cast<std::size_t>(shape_.layer.seq_len),
                             static_cast<std::size_t>(shape_.layer.head_size / 2))),
      sin_table_(cos_table_.count()),
      scratch_(shape_.layer),
      x_(shape_.layer.dim),
      final_norm_(shape_.layer.dim),
      logits_(static_cast<std::size_t>(shape_.vocab_count))
{
    // Host views (also validates the shape and checks for overflow).
    const ModelWeightViews host = select_model_weights(weights, shape_);

    embedding_.upload(host.embedding, "upload token_embedding_table");
    rms_final_.upload(host.rms_final, "upload rms_final_weight");
    cos_table_.upload(host.cos_table, "upload freq_cis_real");
    sin_table_.upload(host.sin_table, "upload freq_cis_imag");

    // Tied classifier: wcls is the embedding table, so the device copy of
    // the embedding doubles as the classifier.  Untied: own a second copy.
    if (!shape_.shared_classifier) {
        classifier_ = std::make_unique<DeviceBuffer>(embedding_.count());
        classifier_->upload(host.wcls, "upload wcls");
    }

    layers_.reserve(static_cast<std::size_t>(shape_.n_layers));
    caches_.reserve(static_cast<std::size_t>(shape_.n_layers));
    for (int l = 0; l < shape_.n_layers; l++) {
        layers_.emplace_back(shape_.layer);
        layers_.back().upload(host.layers[l], shape_.layer);
        caches_.emplace_back(shape_.layer);
    }

    if (keep_layer_outputs) {
        layer_outputs_.reserve(static_cast<std::size_t>(shape_.n_layers));
        for (int l = 0; l < shape_.n_layers; l++) {
            layer_outputs_.emplace_back(shape_.layer.dim);
        }
    }
}

const DeviceBuffer& GpuModel::layer_output(int layer) const
{
    if (layer_outputs_.empty()) {
        throw std::logic_error("GpuModel was built without keep_layer_outputs");
    }
    return layer_outputs_[layer];
}

cudaError_t GpuModel::forward(std::int64_t token, int pos)
{
    const LayerShape& ls = shape_.layer;

    if (needs_reset_ || token < 0 || token >= shape_.vocab_count ||
        pos != position_ || pos >= ls.seq_len) {
        return cudaErrorInvalidValue;
    }

    const std::size_t row_offset = static_cast<std::size_t>(pos) * (ls.head_size / 2);
    const float* cos_row = cos_table_.data() + row_offset;
    const float* sin_row = sin_table_.data() + row_offset;

    // Everything below is stream-ordered on the default stream, so each
    // step sees the previous one's results without host synchronisation.
    cudaError_t status = embedding_lookup_cuda(x_.data(), embedding_.data(), ls.dim, shape_.vocab_count, token);
    if (status != cudaSuccess) {
        needs_reset_ = true;
        return status;
    }

    // Layer l reads x_in and writes scratch_.x_out; the next layer takes
    // scratch_.x_out as its input (safe: see layer_forward_cuda).  A
    // snapshot copy preserves each layer's output when requested, since
    // the next layer overwrites the scratch.
    const float* x_in = x_.data();
    for (int l = 0; l < shape_.n_layers; l++) {
        status = layer_forward_cuda(scratch_, caches_[l], layers_[l], ls, x_in, cos_row, sin_row, pos, kModelRmsEpsilon);
        if (status != cudaSuccess) {
            needs_reset_ = true;   // earlier layers advanced their caches, this one did not
            return status;
        }
        if (!layer_outputs_.empty()) {
            status = cudaMemcpyAsync(layer_outputs_[l].data(), scratch_.x_out.data(),
                                     scratch_.x_out.bytes(), cudaMemcpyDeviceToDevice, 0);
            if (status != cudaSuccess) {
                needs_reset_ = true;
                return status;
            }
        }
        x_in = scratch_.x_out.data();
    }

    status = rmsnorm_cuda(final_norm_.data(), x_in, rms_final_.data(), ls.dim, kModelRmsEpsilon);
    if (status != cudaSuccess) {
        needs_reset_ = true;
        return status;
    }
    // wcls [vocab_count, dim] x final_norm [dim] -> logits [vocab_count]
    status = matvec_cuda(logits_.data(), d_wcls(), final_norm_.data(),
                         static_cast<int>(shape_.vocab_count), ls.dim);
    if (status != cudaSuccess) {
        needs_reset_ = true;
        return status;
    }

    position_++;
    return cudaSuccess;
}

cudaError_t GpuModel::synchronize()
{
    const cudaError_t status = cudaDeviceSynchronize();
    if (status != cudaSuccess) {
        needs_reset_ = true;
    }
    return status;
}

void GpuModel::reset()
{
    for (DeviceKvCache& c : caches_) {
        c.reset();
    }
    position_ = 0;
    needs_reset_ = false;
}
