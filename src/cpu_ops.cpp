#include "cpu_ops.hpp"
#include "attention_shape.hpp"

#include <cmath>
#include <cstddef>

void rmsnorm(float* output, const float* input, const float* weights, int size, float epsilon)
{
    float summation = 0.0f;

    for (int i = 0; i < size; i++) {
        summation += (input[i] * input[i]);
    }

    float scale = 1.0f / std::sqrt(summation / size + epsilon);

    for (int i = 0; i < size; i++) {
        output[i] = weights[i] * input[i] * scale;
    }
}

void matvec(float* output, const float* matrix, const float* input, int rows, int cols)
{
    for (int i = 0; i < rows; i++) {
        double accum = 0;
        for (int j = 0; j < cols; j++) {
            accum += matrix[i * cols + j] * input[j];
        }
        output[i] = static_cast<float>(accum);
    }
}

void rope(float* vec, int n_heads, int head_size, const float* cos_row, const float* sin_row)
{
    const int pairs_per_head = head_size / 2;

    for (int head = 0; head < n_heads; head++) {
        float* head_vec = vec + head * head_size;

        // Pair indexing restarts at 0 for every head, so each head sees
        // the same set of rotation angles.
        for (int pair = 0; pair < pairs_per_head; pair++) {
            const float c = cos_row[pair];
            const float s = sin_row[pair];

            // Read both inputs before writing either output; b_new needs
            // the original a, which the first store would overwrite.
            const float a = head_vec[2 * pair];
            const float b = head_vec[2 * pair + 1];

            head_vec[2 * pair]     = a * c - b * s;
            head_vec[2 * pair + 1] = a * s + b * c;
        }
    }
}

bool kv_cache_store(float* cache, const float* current, int kv_dim, int seq_len, int pos)
{
    if (!cache_row_valid(kv_dim, seq_len, pos)) {
        return false;
    }

    // Row pos starts pos * kv_dim floats into the buffer.
    float* row = cache + static_cast<std::size_t>(pos) * static_cast<std::size_t>(kv_dim);
    for (int i = 0; i < kv_dim; i++) {
        row[i] = current[i];
    }
    return true;
}

void softmax(float* probs, const float* scores, int n)
{
    float max_score = scores[0];
    for (int i = 1; i < n; i++) {
        if (scores[i] > max_score) {
            max_score = scores[i];
        }
    }

    // Accumulate the sum in double so the reference is as exact as the
    // float inputs allow; the GPU sums in float and is checked against it.
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        probs[i] = std::exp(scores[i] - max_score);
        sum += probs[i];
    }

    const float inverse_sum = static_cast<float>(1.0 / sum);
    for (int i = 0; i < n; i++) {
        probs[i] *= inverse_sum;
    }
}

bool attention(
    float* output,
    float* scores,
    float* probs,
    const float* q,
    const float* k_cache,
    const float* v_cache,
    int n_heads,
    int n_kv_heads,
    int head_size,
    int seq_len,
    int pos
)
{
    if (!attention_shape_valid(n_heads, n_kv_heads, head_size, seq_len, pos)) {
        return false;
    }

    const int queries_per_kv_head = n_heads / n_kv_heads;
    const std::size_t kv_dim = static_cast<std::size_t>(n_kv_heads) * head_size;
    const int active_len = pos + 1;   // causal window: rows 0..pos inclusive
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));

    for (int h = 0; h < n_heads; h++) {
        const int kv_head = h / queries_per_kv_head;
        const float* q_head = q + static_cast<std::size_t>(h) * head_size;
        float* score_row = scores + static_cast<std::size_t>(h) * seq_len;
        float* prob_row = probs + static_cast<std::size_t>(h) * seq_len;

        // Scores against every visible cached key.
        for (int t = 0; t < active_len; t++) {
            // Row t of the cache, then head kv_head inside that row.
            const float* k_row = k_cache + t * kv_dim + static_cast<std::size_t>(kv_head) * head_size;
            double dot = 0.0;
            for (int d = 0; d < head_size; d++) {
                dot += static_cast<double>(q_head[d]) * k_row[d];
            }
            score_row[t] = static_cast<float>(dot) * scale;
        }

        softmax(prob_row, score_row, active_len);

        // Weighted sum of the visible cached values.
        float* out_head = output + static_cast<std::size_t>(h) * head_size;
        for (int d = 0; d < head_size; d++) {
            double acc = 0.0;
            for (int t = 0; t < active_len; t++) {
                const float* v_row = v_cache + t * kv_dim + static_cast<std::size_t>(kv_head) * head_size;
                acc += static_cast<double>(prob_row[t]) * v_row[d];
            }
            out_head[d] = static_cast<float>(acc);
        }
    }

    return true;
}

void add_vectors(float* output, const float* a, const float* b, int size)
{
    for (int i = 0; i < size; i++) {
        output[i] = a[i] + b[i];
    }
}

float sigmoid(float z)
{
    if (z >= 0.0f) {
        return 1.0f / (1.0f + std::exp(-z));
    }
    const float e = std::exp(z);
    return e / (1.0f + e);
}

float silu(float z)
{
    return z * sigmoid(z);
}

void silu_gate(float* output, const float* gate, const float* up, int size)
{
    for (int i = 0; i < size; i++) {
        output[i] = silu(gate[i]) * up[i];
    }
}
