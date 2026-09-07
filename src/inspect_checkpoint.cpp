#include "model_config.hpp"
#include "mapped_file.hpp"
#include <iostream>
#include <cstring>

using std::uint64_t;



// Valid Header Values Check
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




int main(int argc, char *argv[])
{

    // Valid command check
    if (argc < 2) {
        std::cerr << "Provide a checkpoint path" << '\n';
        return 1;
    }

    /*
    std::ifstream file(argv[1], std::ios::binary);
    char* pconfig = reinterpret_cast<char*>(&config);
    file.read(pconfig, sizeof(config));

    if (!file) {
        std::cerr << "Error: incomplete checkpoint header\n";
        return 1;
    }
    */


    try {
        MappedFile checkpoint{argv[1]};
        
        if (checkpoint.size() < sizeof(ModelConfig)) {
            std::cerr << "Checkpoint is too small\n";
            return 1;
        }

        ModelConfig config{};

        std::memcpy(
            &config,
            checkpoint.data(),
            sizeof(config)
        );

        if (!validate_config(config)) {
            return 1;
        }

        // Extract parameters from header
        int head_size = config.dim / config.n_heads;
        int KV_dim = head_size * config.n_kv_heads;
        bool shared_classifier = config.vocab_size > 0;
        std::int64_t vocab_size = config.vocab_size;

        if (vocab_size < 0) {
            vocab_size = -vocab_size;
        }

        uint64_t vocab_count = static_cast<uint64_t>(vocab_size);

        if (head_size % 2 != 0) {
            std::cerr << "Error: expected head_size to be even\n";
            return 1;
        }

        std::cout << "Dimension:         " << config.dim << "\n";
        std::cout << "Hidden dimension:  " << config.hidden_dim << "\n";
        std::cout << "Layers:            " << config.n_layers << "\n";
        std::cout << "Attention heads:   " << config.n_heads << "\n";
        std::cout << "KV heads:          " << config.n_kv_heads << "\n";
        std::cout << "Vocabulary size:   " << vocab_count << "\n";
        std::cout << "Maximum sequence:  " << config.seq_len << "\n";
        std::cout << "Head size:         " << head_size << "\n";
        std::cout << "KV dimension:      " << KV_dim << "\n";
        std::cout << "Shared classifier: " << std::boolalpha << shared_classifier << "\n" << "\n";



        // Extract tensor counts from header params
        

        uint64_t embedding_count = static_cast<uint64_t>(vocab_count) * config.dim;
        uint64_t rms_attention_count = static_cast<uint64_t>(config.n_layers) * config.dim;
        uint64_t Wq_count = static_cast<uint64_t>(config.n_layers) * config.dim * config.dim;
        uint64_t Wk_count = static_cast<uint64_t>(config.n_layers) * KV_dim * config.dim;
        uint64_t Wv_count = static_cast<uint64_t>(config.n_layers) * KV_dim * config.dim;
        uint64_t Wo_count = static_cast<uint64_t>(config.n_layers) * config.dim * config.dim;
        uint64_t rms_ffn_count = static_cast<uint64_t>(config.n_layers) * config.dim;
        uint64_t W1_count = static_cast<uint64_t>(config.n_layers) * config.hidden_dim * config.dim;
        uint64_t W2_count = static_cast<uint64_t>(config.n_layers) * config.hidden_dim * config.dim;
        uint64_t W3_count = static_cast<uint64_t>(config.n_layers) * config.hidden_dim * config.dim;
        uint64_t rms_final_count = static_cast<uint64_t>(config.dim);
        uint64_t RoPE_real_count = static_cast<uint64_t>(config.seq_len) * (head_size / 2);
        uint64_t RoPE_imaginary_count = static_cast<uint64_t>(config.seq_len) * (head_size / 2);
        uint64_t classifier_weight_count = static_cast<uint64_t>(vocab_count) * config.dim;

        uint64_t learned_parameter_count = embedding_count + rms_attention_count + Wq_count + Wk_count + Wv_count + Wo_count + rms_ffn_count + W1_count + W2_count + W3_count + rms_final_count;

        if (!shared_classifier) {
            learned_parameter_count += classifier_weight_count;
        }



        // File size checks
        uint64_t stored_float_count = learned_parameter_count + RoPE_real_count + RoPE_imaginary_count;
        uint64_t stored_float_bytes = stored_float_count * sizeof(float);
        uint64_t expected_file_bytes = sizeof(ModelConfig) + stored_float_bytes;
        uint64_t actual_file_bytes = checkpoint.size();
        bool layout_match = expected_file_bytes == actual_file_bytes;

        std::cout << "Learned parameters:   " << learned_parameter_count << '\n';
        std::cout << "Stored floats:        " << stored_float_count << '\n';
        std::cout << "Stored float bytes:   " << stored_float_bytes << '\n';
        std::cout << "Expected files bytes: " << expected_file_bytes << '\n';
        std::cout << "Actual files bytes:   " << actual_file_bytes << '\n';
        std::cout << "Layout matches:       " << std::boolalpha << layout_match << '\n';

        if (!layout_match) {
            std::cerr << "Error: checkpoint layout doesnt match file size\n";
            return 1;
        }


    }

    catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    } 


    return 0;
}