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