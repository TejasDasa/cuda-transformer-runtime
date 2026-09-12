#ifndef CHECKPOINT_HPP
#define CHECKPOINT_HPP

#include "mapped_file.hpp"
#include "model_config.hpp"
#include "model_weights.hpp"
#include <string>

class Checkpoint {
public:
    explicit Checkpoint(const std::string& path);

    const ModelConfig& config() const;
    const ModelWeights& weights() const;

private:
    MappedFile file_;
    ModelConfig config_{};
    ModelWeights weights_{};
};

#endif