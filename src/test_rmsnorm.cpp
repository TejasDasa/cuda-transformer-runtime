#include "cpu_ops.hpp"
#include <iostream>

int main()
{
    float output[2]{};
    float input[] = {3.0f, 4.0f};
    float weights[] = {2.0f, 0.5f};
    int size = 2;
    float epsilon = 0.0f;

    rmsnorm(output, input, weights, size, epsilon);

    std::cout << output[0] << ", " << output[1] << '\n';
    return 0;
}