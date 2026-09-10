#include "model_config.hpp"
#include "mapped_file.hpp"
#include "model_weights.hpp"
#include "cpu_ops.hpp"
#include <iostream>
#include <cstring>
#include <cstddef>
#include <iomanip>
#include <string>
#include <vector>

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

    /* Old code to head header into memory

    std::ifstream file(argv[1], std::ios::binary);
    char* pconfig = reinterpret_cast<char*>(&config);
    file.read(pconfig, sizeof(config));

    if (!file) {
        std::cerr << "Error: incomplete checkpoint header\n";
        return 1;
    }

    */


    try {

        // Write checkpoint into process memory using mmap
        MappedFile checkpoint{argv[1]};
        
        if (checkpoint.size() < sizeof(ModelConfig)) {
            std::cerr << "Checkpoint is too small\n";
            return 1;
        }

        ModelConfig config{};

        std::memcpy(&config, checkpoint.data(), sizeof(config));

        if (!validate_config(config)) {
            return 1;
        }

        // Extract parameters from header
        int head_size = config.dim / config.n_heads;
        int kv_dim = head_size * config.n_kv_heads;
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
        std::cout << "KV dimension:      " << kv_dim << "\n";
        std::cout << "Shared classifier: " << std::boolalpha << shared_classifier << "\n" << "\n";



        // Extract tensor counts from header params

        uint64_t embedding_count = static_cast<uint64_t>(vocab_count) * config.dim;
        uint64_t rms_attention_count = static_cast<uint64_t>(config.n_layers) * config.dim;
        uint64_t wq_count = static_cast<uint64_t>(config.n_layers) * config.dim * config.dim;
        uint64_t wk_count = static_cast<uint64_t>(config.n_layers) * kv_dim * config.dim;
        uint64_t wv_count = static_cast<uint64_t>(config.n_layers) * kv_dim * config.dim;
        uint64_t wo_count = static_cast<uint64_t>(config.n_layers) * config.dim * config.dim;
        uint64_t rms_ffn_count = static_cast<uint64_t>(config.n_layers) * config.dim;
        uint64_t w1_count = static_cast<uint64_t>(config.n_layers) * config.hidden_dim * config.dim;
        uint64_t w2_count = static_cast<uint64_t>(config.n_layers) * config.hidden_dim * config.dim;
        uint64_t w3_count = static_cast<uint64_t>(config.n_layers) * config.hidden_dim * config.dim;
        uint64_t rms_final_count = static_cast<uint64_t>(config.dim);
        uint64_t rope_real_count = static_cast<uint64_t>(config.seq_len) * (head_size / 2);
        uint64_t rope_imaginary_count = static_cast<uint64_t>(config.seq_len) * (head_size / 2);
        uint64_t classifier_weight_count = static_cast<uint64_t>(vocab_count) * config.dim;

        uint64_t learned_parameter_count = embedding_count + rms_attention_count + wq_count + wk_count + wv_count + wo_count + rms_ffn_count + w1_count + w2_count + w3_count + rms_final_count;

        if (!shared_classifier) {
            learned_parameter_count += classifier_weight_count;
        }



        // File size checks
        uint64_t stored_float_count = learned_parameter_count + rope_real_count + rope_imaginary_count;
        uint64_t stored_float_bytes = stored_float_count * sizeof(float);
        uint64_t expected_file_bytes = sizeof(ModelConfig) + stored_float_bytes;
        uint64_t actual_file_bytes = checkpoint.size();
        bool layout_match = expected_file_bytes == actual_file_bytes;

        std::cout << "Learned parameters:   " << learned_parameter_count << '\n';
        std::cout << "Stored floats:        " << stored_float_count << '\n';
        std::cout << "Stored float bytes:   " << stored_float_bytes << '\n';
        std::cout << "Expected files bytes: " << expected_file_bytes << '\n';
        std::cout << "Actual files bytes:   " << actual_file_bytes << '\n';
        std::cout << "Layout matches:       " << std::boolalpha << layout_match << '\n' << '\n';

        if (!layout_match) {
            std::cerr << "Error: checkpoint layout doesnt match file size\n";
            return 1;
        }



        // Load pointers to each tensor into the weights struct
        ModelWeights weights{};

        const std::byte* file_begin = static_cast<const std::byte*>(checkpoint.data());
        const std::byte* weights_begin = file_begin + sizeof(ModelConfig);
        const float* cursor = reinterpret_cast<const float*>(weights_begin);
        
        static_assert(sizeof(ModelConfig) % alignof(float) == 0);

        weights.token_embedding_table = cursor;
        cursor += embedding_count;

        weights.rms_att_weight = cursor;
        cursor += rms_attention_count;

        weights.wq = cursor;
        cursor += wq_count;

        weights.wk = cursor;
        cursor += wk_count;

        weights.wv = cursor;
        cursor += wv_count;

        weights.wo = cursor;
        cursor += wo_count;

        weights.rms_ffn_weight = cursor;
        cursor += rms_ffn_count;

        weights.w1 = cursor;
        cursor += w1_count;

        weights.w2 = cursor;
        cursor += w2_count;

        weights.w3 = cursor;
        cursor += w3_count;

        weights.rms_final_weight = cursor;
        cursor += rms_final_count;

        weights.freq_cis_real = cursor;
        cursor += rope_real_count;

        weights.freq_cis_imag = cursor;
        cursor += rope_imaginary_count;

        if (shared_classifier) {
            weights.wcls = weights.token_embedding_table;
        } else {
            weights.wcls = cursor;
            cursor += classifier_weight_count;
        }

        auto print_offset = [file_begin](const char* name, const float* pointer) {
            auto offset = reinterpret_cast<const std::byte*>(pointer) - file_begin;
            std::cout << std::left << std::setw(30) << (std::string(name) + ":") << offset << '\n';
        };

        print_offset("Token Embeddings offset", weights.token_embedding_table);
        print_offset("Attention RMSNorm offset", weights.rms_att_weight);
        print_offset("Wq offset", weights.wq);
        print_offset("Wk offset", weights.wk);
        print_offset("Wv offset", weights.wv);
        print_offset("Wo offset", weights.wo);
        print_offset("FFN RMSNorm offset", weights.rms_ffn_weight);
        print_offset("W1 offset", weights.w1);
        print_offset("W2 offset", weights.w2);
        print_offset("W3 offset", weights.w3);
        print_offset("Final RMSNorm offset", weights.rms_final_weight);
        print_offset("RoPE real offset", weights.freq_cis_real);
        print_offset("RoPE imaginary offset", weights.freq_cis_imag);
        print_offset("Classifier offset", weights.wcls);
        std::cout << "\n";

        auto final_offset = reinterpret_cast<const std::byte*>(cursor) - file_begin;

        bool bytes_match = final_offset >= 0 && static_cast<std::size_t>(final_offset) == checkpoint.size();

        std::cout << "Final cursor offset:          " << final_offset << '\n';
        std::cout << "Mapped file bytes:            " << checkpoint.size() << '\n';
        std::cout << "All bytes accounted for:      " << std::boolalpha << bytes_match << '\n';

        std::cout << "Classifier shares embeddings: " << (weights.wcls == weights.token_embedding_table) << '\n' << '\n';

        if (!bytes_match) {
            std::cerr << "Error: tensor cursor does not reach the file end\n";
            return 1;
        }




        // Replace with tokenization later, for now loads defined vector id with token embeddings
        int token_id = 42;

        if (static_cast<uint64_t>(token_id) > vocab_count || token_id < 0) {
            std::cerr << "Error: Token out of vocab range\n";
            return 1;
        }

        std::vector<float> x(config.dim);

        const float* row = weights.token_embedding_table + (config.dim * static_cast<std::size_t>(token_id));

        for (int i = 0; i < config.dim; i++) {
            x[i] = row[i];
        }

        std::cout << "First 8 elements of embedding vector:\n";
        for (int i = 0; i < 8; i++) {
            std::cout << ' ' << x[i] << "\n";
        }




        // RMSNorm applied to embedding vec (layer 1)
        std::vector<float> normalized(config.dim);

        rmsnorm(normalized.data(), x.data(), weights.rms_att_weight, config.dim, 1e-5f);

        std::cout << '\n' << "Input       Output\n";
        for (int i = 0; i < 8; i++) {
            std::cout << x[i] << "       " << normalized[i] << '\n';
        }



        // Layer 1 QKV computation
        std::vector<float> q(config.dim);
        std::vector<float> k(kv_dim);
        std::vector<float> v(kv_dim);

        matvec(q.data(), weights.wq, normalized.data(), config.dim, config.dim);
        matvec(k.data(), weights.wk, normalized.data(), kv_dim, config.dim);
        matvec(v.data(), weights.wv, normalized.data(), kv_dim, config.dim);

        std::cout << '\n' << "Q  K  V\n";

        for (int i = 0; i < 8; i++) {
            std::cout << q[i] << "  " << k[i] << "  " << v[i] << '\n';
        }
    }

    catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    } 


    return 0;
}