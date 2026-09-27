#include "checkpoint.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>

using std::uint64_t;

bool validate_config(const ModelConfig& config)
{
    auto fail = [](const char* message) {
        std::cerr << "Invalid checkpoint: " << message << '\n';
        return false;
    };

    if (config.dim <= 0) {
        return fail("model dimension must be positive");
    }

    if (config.hidden_dim <= 0) {
        return fail("hidden dimension must be positive");
    }

    if (config.n_layers <= 0) {
        return fail("number of layers must be positive");
    }

    if (config.n_heads <= 0) {
        return fail("number of attention heads must be positive");
    }

    if (config.n_kv_heads <= 0) {
        return fail("number of KV heads must be positive");
    }

    if (config.vocab_size == 0) {
        return fail("vocabulary size cannot be zero");
    }

    if (config.seq_len <= 0) {
        return fail("maximum sequence length must be positive");
    }

    if (config.n_kv_heads > config.n_heads) {
        return fail("KV head count cannot exceed attention head count");
    }

    if (config.dim % config.n_heads != 0) {
        return fail("model dimension must be divisible by attention head count");
    }

    if (config.n_heads % config.n_kv_heads != 0) {
        return fail("attention head count must be divisible by KV head count");
    }

    return true;
}



Checkpoint::Checkpoint(const std::string& path)
    : file_(path)
{
    static_assert(sizeof(ModelConfig) == 28);
    static_assert(sizeof(float) == 4);
    static_assert(sizeof(ModelConfig) % alignof(float) == 0);

    if (file_.size() < sizeof(ModelConfig)) {
        throw std::runtime_error("Checkpoint is too small");
    }

    std::memcpy(&config_, file_.data(), sizeof(config_));

    if (!validate_config(config_)) {
        throw std::runtime_error("Invalid checkpoint configuration");
    }

    const int head_size = config_.dim / config_.n_heads;
    const int kv_dim = head_size * config_.n_kv_heads;
    const bool shared_classifier = config_.vocab_size > 0;

    if (head_size % 2 != 0) {
        throw std::runtime_error("Expected head size to be even");
    }

    std::int64_t vocab_size = config_.vocab_size;

    if (vocab_size < 0) {
        vocab_size = -vocab_size;
    }

    const uint64_t vocab_count = static_cast<uint64_t>(vocab_size);

    const uint64_t embedding_count =
        vocab_count * config_.dim;

    const uint64_t rms_attention_count =
        static_cast<uint64_t>(config_.n_layers) * config_.dim;

    const uint64_t wq_count =
        static_cast<uint64_t>(config_.n_layers) *
        config_.dim * config_.dim;

    const uint64_t wk_count =
        static_cast<uint64_t>(config_.n_layers) *
        kv_dim * config_.dim;

    const uint64_t wv_count = wk_count;
    const uint64_t wo_count = wq_count;
    const uint64_t rms_ffn_count = rms_attention_count;

    const uint64_t w1_count =
        static_cast<uint64_t>(config_.n_layers) *
        config_.hidden_dim * config_.dim;

    const uint64_t w2_count = w1_count;
    const uint64_t w3_count = w1_count;

    const uint64_t rms_final_count =
        static_cast<uint64_t>(config_.dim);

    const uint64_t rope_real_count =
        static_cast<uint64_t>(config_.seq_len) * (head_size / 2);

    const uint64_t rope_imaginary_count = rope_real_count;
    const uint64_t classifier_weight_count = embedding_count;

    uint64_t learned_parameter_count =
        embedding_count + rms_attention_count +
        wq_count + wk_count + wv_count + wo_count +
        rms_ffn_count + w1_count + w2_count + w3_count +
        rms_final_count;

    if (!shared_classifier) {
        learned_parameter_count += classifier_weight_count;
    }

    const uint64_t stored_float_count =
        learned_parameter_count + rope_real_count +
        rope_imaginary_count;

    const uint64_t expected_file_bytes =
        sizeof(ModelConfig) + stored_float_count * sizeof(float);

    if (expected_file_bytes != file_.size()) {
        throw std::runtime_error(
            "Checkpoint layout does not match file size"
        );
    }

    const auto* file_begin =
        static_cast<const std::byte*>(file_.data());

    const float* cursor = reinterpret_cast<const float*>(
        file_begin + sizeof(ModelConfig)
    );

    weights_.token_embedding_table = cursor;
    cursor += embedding_count;

    weights_.rms_att_weight = cursor;
    cursor += rms_attention_count;

    weights_.wq = cursor;
    cursor += wq_count;

    weights_.wk = cursor;
    cursor += wk_count;

    weights_.wv = cursor;
    cursor += wv_count;

    weights_.wo = cursor;
    cursor += wo_count;

    weights_.rms_ffn_weight = cursor;
    cursor += rms_ffn_count;

    weights_.w1 = cursor;
    cursor += w1_count;

    weights_.w2 = cursor;
    cursor += w2_count;

    weights_.w3 = cursor;
    cursor += w3_count;

    weights_.rms_final_weight = cursor;
    cursor += rms_final_count;

    weights_.freq_cis_real = cursor;
    cursor += rope_real_count;

    weights_.freq_cis_imag = cursor;
    cursor += rope_imaginary_count;

    if (shared_classifier) {
        weights_.wcls = weights_.token_embedding_table;
    } else {
        weights_.wcls = cursor;
        cursor += classifier_weight_count;
    }

    if (reinterpret_cast<const std::byte*>(cursor) !=
        file_begin + file_.size()) {
        throw std::runtime_error(
            "Tensor cursor does not reach the file end"
        );
    }
}

const ModelConfig& Checkpoint::config() const
{
    return config_;
}

const ModelWeights& Checkpoint::weights() const
{
    return weights_;
}