#include "cpu_ops.hpp"
#include <iostream>

int main()
{
    float W[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    float input[] = {1.0f, 2.0f, 3.0f};
    float output[]{};
    int rows = 2;
    int cols = 3;

    matvec(output, W, input, rows, cols);

    std::cout << output[0] << ", " << output[1] << '\n';
}