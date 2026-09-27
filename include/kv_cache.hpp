#ifndef KV_CACHE_HPP
#define KV_CACHE_HPP

#include <cstddef>

// Tracks which rows of a [seq_len, kv_dim] key or value cache hold real
// data.  The storage itself lives elsewhere (a std::vector on the host, a
// DeviceBuffer on the GPU); this object only enforces the rule that rows
// are written strictly in order 0, 1, 2, ... and that attention never
// reads a row that has not been written yet.
//
// Typical use per token step at position pos:
//     if (!state.accepts(pos)) -> error (skipped or repeated position)
//     store row pos in both caches
//     state.advance()
//     attention over rows [0, pos]   (state.covers(pos) is now true)
class KvCacheState {
public:
    KvCacheState(int seq_len, int kv_dim)
        : seq_len_(seq_len), kv_dim_(kv_dim)
    {
    }

    int seq_len() const { return seq_len_; }
    int kv_dim() const { return kv_dim_; }

    // Number of valid rows; also the next position that may be stored.
    int length() const { return length_; }

    // Total floats in one cache buffer, computed in size_t so that large
    // seq_len * kv_dim products cannot overflow an int.
    std::size_t capacity_floats() const
    {
        return static_cast<std::size_t>(seq_len_) * static_cast<std::size_t>(kv_dim_);
    }

    // Flattened index of the first float of row pos.
    std::size_t row_offset(int pos) const
    {
        return static_cast<std::size_t>(pos) * static_cast<std::size_t>(kv_dim_);
    }

    // May row pos be written now?  Only the next unwritten row qualifies.
    bool accepts(int pos) const
    {
        return pos == length_ && pos < seq_len_;
    }

    // Marks the next row as written.
    void advance()
    {
        if (length_ < seq_len_) {
            length_++;
        }
    }

    // May attention at position pos read rows [0, pos]?
    bool covers(int pos) const
    {
        return pos >= 0 && pos < length_;
    }

    // Start a new sequence: the buffers keep their old bytes but none of
    // them count as valid any more.
    void reset()
    {
        length_ = 0;
    }

private:
    int seq_len_ = 0;
    int kv_dim_ = 0;
    int length_ = 0;
};

#endif
