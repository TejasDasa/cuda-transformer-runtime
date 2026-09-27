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

// Rotary positional embedding (RoPE), applied in place.
//
// `vec` holds n_heads consecutive heads of head_size floats each
// (n_heads * head_size floats in total).  Inside every head, adjacent
// elements (0,1), (2,3), ... form pairs; pair p of every head is rotated
// by the same angle, whose cosine and sine are cos_row[p] and sin_row[p].
// Both rows therefore hold head_size / 2 entries and correspond to one
// token position.  head_size must be even.
//
// The rotation of a pair (a, b) is
//     a' = a * cos - b * sin
//     b' = a * sin + b * cos
void rope(
    float* vec,
    int n_heads,
    int head_size,
    const float* cos_row,
    const float* sin_row
);

#endif
