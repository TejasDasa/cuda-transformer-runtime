#ifndef MODEL_CONFIG_HPP
#define MODEL_CONFIG_HPP

#include <cstdint>

using std::int32_t;

struct ModelConfig {
    int32_t dim = 0;
    int32_t hidden_dim = 0;
    int32_t n_layers = 0;
    int32_t n_heads = 0;
    int32_t n_kv_heads = 0;
    int32_t vocab_size = 0;
    int32_t seq_len = 0;
};

static_assert(sizeof(ModelConfig) == 7 * sizeof(std::int32_t));

#endif