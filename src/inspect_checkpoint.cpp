#include "checkpoint.hpp"
#include "cpu_ops.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <vector>

using std::uint64_t;

int main(int argc, char *argv[])
{

    // Valid command check
    if (argc < 2) {
        std::cerr << "Provide a checkpoint path" << '\n';
        return 1;
    }

    try {

        Checkpoint checkpoint{argv[1]};

        const ModelConfig& config = checkpoint.config();
        const ModelWeights& weights = checkpoint.weights();

        
        // Extract parameters from header
        int head_size = config.dim / config.n_heads;
        int kv_dim = head_size * config.n_kv_heads;
        bool shared_classifier = config.vocab_size > 0;
        std::int64_t vocab_size = config.vocab_size;

        if (vocab_size < 0) {
            vocab_size = -vocab_size;
        }

        uint64_t vocab_count = static_cast<uint64_t>(vocab_size);

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

        // Checkpoint's constructor has already validated the header,
        // file size, and tensor layout, and assigned the weight pointers.
        std::cout << "Checkpoint layout validated\n";
        std::cout << "Classifier shares embeddings: "
                  << (weights.wcls == weights.token_embedding_table) << "\n\n";

        // Replace with tokenization later, for now loads defined vector id with token embeddings
        int token_id = 42;

        if (token_id < 0 || static_cast<uint64_t>(token_id) >= vocab_count) {
            std::cerr << "Error: Token out of vocab range\n";
            return 1;
        }

        std::vector<float> x(config.dim);

        const float* row = weights.token_embedding_table + (config.dim * static_cast<std::size_t>(token_id));

        for (int i = 0; i < config.dim; i++) {
            x[i] = row[i];
        }

        std::cout << "First 8 elements of embedding vector:\n";
        for (int i = 0; i < std::min(8, config.dim); i++) {
            std::cout << ' ' << x[i] << "\n";
        }

        // RMSNorm applied to embedding vec (layer 0)
        std::vector<float> normalized(config.dim);

        rmsnorm(normalized.data(), x.data(), weights.rms_att_weight, config.dim, 1e-5f);

        std::cout << '\n' << "Input       Output\n";
        for (int i = 0; i < std::min(8, config.dim); i++) {
            std::cout << x[i] << "       " << normalized[i] << '\n';
        }

        // Layer 0 QKV computation
        std::vector<float> q(config.dim);
        std::vector<float> k(kv_dim);
        std::vector<float> v(kv_dim);

        matvec(q.data(), weights.wq, normalized.data(), config.dim, config.dim);
        matvec(k.data(), weights.wk, normalized.data(), kv_dim, config.dim);
        matvec(v.data(), weights.wv, normalized.data(), kv_dim, config.dim);

        std::cout << '\n' << "Q  K  V\n";

        for (int i = 0; i < std::min(8, kv_dim); i++) {
            std::cout << q[i] << "  " << k[i] << "  " << v[i] << '\n';
        }
    }

    catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    } 

    return 0;
}