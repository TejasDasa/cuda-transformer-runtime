#ifndef CPU_OPS_HPP
#define CPU_OPS_HPP

void rmsnorm(
    float* output,
    const float* input,
    const float* weights,
    int size,
    float epsilon
);

void matvec(
    float* output,
    const float* matrix,
    const float* input,
    int rows,
    int cols
);

#endif