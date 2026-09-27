#ifndef ATTENTION_SHAPE_HPP
#define ATTENTION_SHAPE_HPP

// Host-side validation shared by the CPU and CUDA attention entry points.
//
// Shapes involved:
//   Q            : [n_heads, head_size]
//   K / V cache  : [seq_len, n_kv_heads, head_size] = [seq_len, kv_dim]
//   output       : [n_heads, head_size]
//
// Grouped-query attention lets several query heads share one KV head:
//   queries_per_kv_head = n_heads / n_kv_heads
//   kv_head             = query_head / queries_per_kv_head
// which is only well defined when n_heads is a multiple of n_kv_heads.
inline bool attention_shape_valid(
    int n_heads,
    int n_kv_heads,
    int head_size,
    int seq_len,
    int pos
)
{
    return n_heads > 0 &&
           n_kv_heads > 0 &&
           head_size > 0 &&
           seq_len > 0 &&
           n_heads % n_kv_heads == 0 &&
           pos >= 0 &&
           pos < seq_len;
}

// A cache row is addressable only when it lies inside [0, seq_len).
inline bool cache_row_valid(int kv_dim, int seq_len, int pos)
{
    return kv_dim > 0 && seq_len > 0 && pos >= 0 && pos < seq_len;
}

#endif
