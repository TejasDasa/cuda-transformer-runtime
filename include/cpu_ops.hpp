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

// ---------------------------------------------------------------------
// KV cache and single-token causal attention.
// ---------------------------------------------------------------------

// Copies the kv_dim floats of `current` into row `pos` of `cache`, whose
// layout is [seq_len, kv_dim] (row-major, so row pos begins at
// pos * kv_dim).  Earlier rows are untouched.  Returns false, writing
// nothing, if pos is outside [0, seq_len) or kv_dim is not positive.
bool kv_cache_store(
    float* cache,
    const float* current,
    int kv_dim,
    int seq_len,
    int pos
);

// Numerically stable softmax of n scores:
//     probs[i] = exp(scores[i] - max) / sum_j exp(scores[j] - max)
// Subtracting the maximum first keeps every exponent <= 0, so nothing
// overflows even for very large scores, and the result is identical
// mathematically because the common factor cancels.
void softmax(float* probs, const float* scores, int n);

// Causal attention for the single token at position pos.
//
//   q       : [n_heads, head_size]            (already rotated by RoPE)
//   k_cache : [seq_len, n_kv_heads, head_size] rows [0, pos] valid (rotated)
//   v_cache : [seq_len, n_kv_heads, head_size] rows [0, pos] valid
//   output  : [n_heads, head_size]
//   scores  : [n_heads, seq_len] scratch; entries [h, 0..pos] are written
//   probs   : [n_heads, seq_len] scratch; entries [h, 0..pos] are written
//
// For query head h and its KV head kv = h / (n_heads / n_kv_heads):
//   scores[h, t] = dot(q[h], k_cache[t, kv]) / sqrt(head_size)   t <= pos
//   probs[h, :]  = softmax(scores[h, 0..pos])
//   output[h, d] = sum_t probs[h, t] * v_cache[t, kv, d]
//
// Only rows 0..pos take part (causal: a token sees itself and the past).
// Returns false, writing nothing, if the shape is invalid.
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
);

// ---------------------------------------------------------------------
// Elementwise operations used by the residual stream and the FFN.
//
// Aliasing rule for both functions: `output` may be exactly the same
// pointer as one of the inputs (true in-place update), because element i
// of the output depends only on element i of each input.  Buffers that
// overlap partially (e.g. output = a + 1) are NOT supported.
// ---------------------------------------------------------------------

// output[i] = a[i] + b[i]
void add_vectors(float* output, const float* a, const float* b, int size);

// Numerically stable logistic sigmoid.  Both branches only ever call
// exp() on a non-positive argument, so no finite input overflows.
float sigmoid(float z);

// SiLU(z) = z * sigmoid(z)
float silu(float z);

// output[i] = SiLU(gate[i]) * up[i]   (the SwiGLU gating step)
void silu_gate(float* output, const float* gate, const float* up, int size);

#endif
