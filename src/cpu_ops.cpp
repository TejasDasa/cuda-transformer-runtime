#include "cpu_ops.hpp"
#include <cmath>

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
