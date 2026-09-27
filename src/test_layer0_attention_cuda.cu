// LAYER-0 ATTENTION TEST (not full inference).
//
// Runs a sequence of tokens through the attention half of transformer
// layer 0 only, on both CPU and GPU, and compares every intermediate:
//
//   for pos = 0, 1, 2, ...:
//     x      = embedding[token[pos]]                     [dim]
//     xn     = RMSNorm(x, rms_att_weight[0])              [dim]
//     q,k,v  = Wq xn, Wk xn, Wv xn                        [dim], [kv_dim], [kv_dim]
//     q,k    = RoPE(q, pos), RoPE(k, pos)
//     K[pos] = k ; V[pos] = v          (caches: [seq_len, kv_dim])
//     out    = attention(q, K[0..pos], V[0..pos])         [dim]
//
// Wo, the residual add, the FFN and later layers are NOT applied, so
// `out` is the concatenated per-head attention output, nothing more.
//
// Usage:  test_layer0_attention_cuda <checkpoint path> [n_tokens]
//   n_tokens defaults to 8; pass seq_len (256 for Stories15M) or the word
//   "full" to fill the whole context.

#include "attention_shape.hpp"
#include "checkpoint.hpp"
#include "cpu_ops.hpp"
#include "cuda_ops.hpp"
#include "device_buffer.hpp"
#include "kv_cache.hpp"
#include "test_compare.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr float kRmsEpsilon = 1e-5f;
constexpr int kDefaultTokens = 8;
constexpr int kVerbosePositions = 16;   // per-position lines up to this many tokens
constexpr int kMaxFailedPositions = 3;  // stop after this many bad positions

// Running summary for one diagnostic across all positions.
struct DiagnosticStats {
    float max_abs_error = 0.0f;
    std::size_t mismatches = 0;
    int failed_positions = 0;

    void add(const ComparisonResult& r)
    {
        max_abs_error = std::max(max_abs_error, r.max_abs_error);
        mismatches += r.mismatches;
        if (!r.passed) {
            failed_positions++;
        }
    }
};

// The first active_len entries of each of n_rows rows of a
// [n_rows, row_stride] scratch buffer, packed together.  Used so that
// scores/probs comparisons never look at the unwritten tail of a row.
std::vector<float> active_entries(const std::vector<float>& buffer, int n_rows, int row_stride, int active_len)
{
    std::vector<float> packed;
    packed.reserve(static_cast<std::size_t>(n_rows) * active_len);
    for (int r = 0; r < n_rows; r++) {
        const float* row = buffer.data() + static_cast<std::size_t>(r) * row_stride;
        packed.insert(packed.end(), row, row + active_len);
    }
    return packed;
}

bool parse_count(const char* text, long long& out)
{
    errno = 0;
    char* end = nullptr;
    const long long value = std::strtoll(text, &end, 10);
    if (end == text || *end != '\0' || errno == ERANGE) {
        return false;
    }
    out = value;
    return true;
}

} // namespace

int main(int argc, char* argv[])
{
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: " << argv[0] << " <checkpoint path> [n_tokens | full]\n";
        return 1;
    }

    try {
        Checkpoint checkpoint{argv[1]};
        const ModelConfig& config = checkpoint.config();
        const ModelWeights& weights = checkpoint.weights();

        // ---------------------------------------------------------------
        // Dimensions, all derived from the checkpoint.
        // ---------------------------------------------------------------
        const int dim = config.dim;
        const int n_heads = config.n_heads;
        const int n_kv_heads = config.n_kv_heads;
        const int head_size = dim / n_heads;
        const int kv_dim = head_size * n_kv_heads;
        const int half = head_size / 2;
        const int seq_len = config.seq_len;

        std::int64_t vocab_size = config.vocab_size;
        if (vocab_size < 0) {
            vocab_size = -vocab_size;   // sign only marks classifier sharing
        }

        if (head_size % 2 != 0) {
            std::cerr << "Error: head_size " << head_size << " is odd\n";
            return 1;
        }
        if (!attention_shape_valid(n_heads, n_kv_heads, head_size, seq_len, 0)) {
            std::cerr << "Error: checkpoint attention shape is invalid\n";
            return 1;
        }

        long long n_tokens = kDefaultTokens;
        if (argc == 3) {
            if (std::strcmp(argv[2], "full") == 0) {
                n_tokens = seq_len;
            } else if (!parse_count(argv[2], n_tokens)) {
                std::cerr << "Error: n_tokens '" << argv[2] << "' is not an integer\n";
                return 1;
            }
        }
        if (n_tokens < 1 || n_tokens > seq_len) {
            std::cerr << "Error: n_tokens must be in [1, " << seq_len << "]\n";
            return 1;
        }

        // Deterministic token ids.  7919 is coprime with any vocabulary
        // size that is not a multiple of it, so short sequences get
        // distinct tokens; repeats in long sequences are harmless.
        std::vector<long long> tokens(n_tokens);
        for (long long i = 0; i < n_tokens; i++) {
            tokens[i] = (i * 7919 + 42) % vocab_size;
        }

        const bool verbose = n_tokens <= kVerbosePositions;

        std::cout << "LAYER-0 ATTENTION TEST (embedding -> RMSNorm -> QKV -> RoPE -> KV cache -> attention)\n"
                  << "checkpoint: " << argv[1] << '\n'
                  << "dim=" << dim << " n_heads=" << n_heads << " n_kv_heads=" << n_kv_heads
                  << " head_size=" << head_size << " kv_dim=" << kv_dim
                  << " seq_len=" << seq_len << " vocab_size=" << vocab_size
                  << " n_tokens=" << n_tokens << '\n'
                  << "tokens:";
        for (long long i = 0; i < std::min<long long>(n_tokens, 16); i++) {
            std::cout << ' ' << tokens[i];
        }
        std::cout << (n_tokens > 16 ? " ...\n" : "\n");

        // ---------------------------------------------------------------
        // Sizes (size_t so products cannot overflow).
        // ---------------------------------------------------------------
        const std::size_t wq_count = static_cast<std::size_t>(dim) * dim;
        const std::size_t wk_count = static_cast<std::size_t>(kv_dim) * dim;
        const std::size_t table_count = static_cast<std::size_t>(seq_len) * half;
        const std::size_t scratch_count = static_cast<std::size_t>(n_heads) * seq_len;

        // ---------------------------------------------------------------
        // Host state.  The CPU caches and their validity tracker.
        // ---------------------------------------------------------------
        KvCacheState cache_state(seq_len, kv_dim);
        std::vector<float> k_cache_cpu(cache_state.capacity_floats(), 0.0f);
        std::vector<float> v_cache_cpu(cache_state.capacity_floats(), 0.0f);

        std::vector<float> x(dim), xn_cpu(dim), q_cpu(dim), k_cpu(kv_dim), v_cpu(kv_dim);
        std::vector<float> out_cpu(dim), scores_cpu(scratch_count, 0.0f), probs_cpu(scratch_count, 0.0f);

        // ---------------------------------------------------------------
        // Device state, allocated once and reused for every token step.
        //   weights / tables : uploaded once
        //   d_x              : embedding row, uploaded each step
        //   d_xn, d_q, d_k, d_v, d_out : per-step working vectors
        //   d_k_cache, d_v_cache       : [seq_len, kv_dim], persist
        //   d_scores, d_probs          : [n_heads, seq_len] scratch
        // Nothing read back for diagnostics is ever uploaded again; the
        // GPU chain feeds itself entirely from device memory.
        // ---------------------------------------------------------------
        DeviceBuffer d_rms(dim);
        DeviceBuffer d_wq(wq_count);
        DeviceBuffer d_wk(wk_count);
        DeviceBuffer d_wv(wk_count);
        DeviceBuffer d_cos(table_count);
        DeviceBuffer d_sin(table_count);
        DeviceBuffer d_x(dim);
        DeviceBuffer d_xn(dim);
        DeviceBuffer d_q(dim);
        DeviceBuffer d_k(kv_dim);
        DeviceBuffer d_v(kv_dim);
        DeviceBuffer d_out(dim);
        DeviceBuffer d_k_cache(cache_state.capacity_floats());
        DeviceBuffer d_v_cache(cache_state.capacity_floats());
        DeviceBuffer d_scores(scratch_count);
        DeviceBuffer d_probs(scratch_count);

        d_rms.upload(weights.rms_att_weight, "upload rms_att_weight");
        d_wq.upload(weights.wq, "upload wq");
        d_wk.upload(weights.wk, "upload wk");
        d_wv.upload(weights.wv, "upload wv");
        d_cos.upload(weights.freq_cis_real, "upload freq_cis_real");
        d_sin.upload(weights.freq_cis_imag, "upload freq_cis_imag");

        // Start both caches from the same known bytes so a cache
        // comparison cannot pass by accident on stale memory.
        d_k_cache.upload(k_cache_cpu.data(), "clear k cache");
        d_v_cache.upload(v_cache_cpu.data(), "clear v cache");
        check_cuda(cudaMemset(d_scores.data(), 0, d_scores.bytes()), "clear scores");
        check_cuda(cudaMemset(d_probs.data(), 0, d_probs.bytes()), "clear probs");

        DiagnosticStats stat_q, stat_k_cache, stat_v_cache, stat_scores, stat_probs, stat_out;
        bool all_passed = true;
        int failed_positions = 0;

        for (int pos = 0; pos < n_tokens; pos++) {
            const long long token = tokens[pos];
            const int active_len = pos + 1;
            const std::size_t row_offset = static_cast<std::size_t>(pos) * half;
            const std::string tag = "pos " + std::to_string(pos);

            // The tracker insists rows are written in order.
            if (!cache_state.accepts(pos)) {
                std::cout << "[" << tag << "] FAIL cache refuses position\n";
                all_passed = false;
                break;
            }

            // ----- CPU reference for this step -----
            const float* embedding_row = weights.token_embedding_table + static_cast<std::size_t>(token) * dim;
            std::copy(embedding_row, embedding_row + dim, x.begin());

            rmsnorm(xn_cpu.data(), x.data(), weights.rms_att_weight, dim, kRmsEpsilon);
            matvec(q_cpu.data(), weights.wq, xn_cpu.data(), dim, dim);
            matvec(k_cpu.data(), weights.wk, xn_cpu.data(), kv_dim, dim);
            matvec(v_cpu.data(), weights.wv, xn_cpu.data(), kv_dim, dim);
            rope(q_cpu.data(), n_heads, head_size, weights.freq_cis_real + row_offset, weights.freq_cis_imag + row_offset);
            rope(k_cpu.data(), n_kv_heads, head_size, weights.freq_cis_real + row_offset, weights.freq_cis_imag + row_offset);

            if (!kv_cache_store(k_cache_cpu.data(), k_cpu.data(), kv_dim, seq_len, pos) ||
                !kv_cache_store(v_cache_cpu.data(), v_cpu.data(), kv_dim, seq_len, pos)) {
                std::cout << "[" << tag << "] FAIL cpu cache store rejected\n";
                all_passed = false;
                break;
            }

            // ----- GPU chain for this step (default stream, in order) -----
            d_x.upload(x.data(), "upload embedding");

            check_cuda(rmsnorm_cuda(d_xn.data(), d_x.data(), d_rms.data(), dim, kRmsEpsilon), "launch rmsnorm");
            check_cuda(matvec_cuda(d_q.data(), d_wq.data(), d_xn.data(), dim, dim), "launch matvec q");
            check_cuda(matvec_cuda(d_k.data(), d_wk.data(), d_xn.data(), kv_dim, dim), "launch matvec k");
            check_cuda(matvec_cuda(d_v.data(), d_wv.data(), d_xn.data(), kv_dim, dim), "launch matvec v");
            check_cuda(rope_cuda(d_q.data(), n_heads, head_size, d_cos.data() + row_offset, d_sin.data() + row_offset), "launch rope q");
            check_cuda(rope_cuda(d_k.data(), n_kv_heads, head_size, d_cos.data() + row_offset, d_sin.data() + row_offset), "launch rope k");
            check_cuda(kv_cache_store_cuda(d_k_cache.data(), d_k.data(), kv_dim, seq_len, pos), "launch store k");
            check_cuda(kv_cache_store_cuda(d_v_cache.data(), d_v.data(), kv_dim, seq_len, pos), "launch store v");

            // Both sides have written row pos; from here on attention may
            // read rows [0, pos].
            cache_state.advance();
            if (!cache_state.covers(pos)) {
                std::cout << "[" << tag << "] FAIL cache does not cover position after store\n";
                all_passed = false;
                break;
            }

            if (!attention(out_cpu.data(), scores_cpu.data(), probs_cpu.data(),
                           q_cpu.data(), k_cache_cpu.data(), v_cache_cpu.data(),
                           n_heads, n_kv_heads, head_size, seq_len, pos)) {
                std::cout << "[" << tag << "] FAIL cpu attention rejected shape\n";
                all_passed = false;
                break;
            }
            check_cuda(attention_cuda(d_out.data(), d_scores.data(), d_probs.data(),
                                      d_q.data(), d_k_cache.data(), d_v_cache.data(),
                                      n_heads, n_kv_heads, head_size, seq_len, pos),
                       "launch attention");

            // One sync per token step catches execution errors from the
            // whole chain above.
            check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

            // ----- Diagnostics (read-only; never fed back to the GPU) -----
            const std::vector<float> q_gpu = d_q.download("download q");
            const std::vector<float> k_cache_gpu = d_k_cache.download("download k cache");
            const std::vector<float> v_cache_gpu = d_v_cache.download("download v cache");
            const std::vector<float> scores_gpu = d_scores.download("download scores");
            const std::vector<float> probs_gpu = d_probs.download("download probs");
            const std::vector<float> out_gpu = d_out.download("download output");

            const std::size_t prefix = static_cast<std::size_t>(active_len) * kv_dim;
            const std::vector<float> k_prefix_gpu(k_cache_gpu.begin(), k_cache_gpu.begin() + prefix);
            const std::vector<float> k_prefix_cpu(k_cache_cpu.begin(), k_cache_cpu.begin() + prefix);
            const std::vector<float> v_prefix_gpu(v_cache_gpu.begin(), v_cache_gpu.begin() + prefix);
            const std::vector<float> v_prefix_cpu(v_cache_cpu.begin(), v_cache_cpu.begin() + prefix);

            bool step_ok = true;
            ComparisonResult r;

            r = compare_vectors("q rope " + tag, q_gpu, q_cpu, verbose);
            stat_q.add(r); step_ok &= r.passed;

            r = compare_vectors("k cache[0.." + std::to_string(pos) + "] " + tag, k_prefix_gpu, k_prefix_cpu, verbose);
            stat_k_cache.add(r); step_ok &= r.passed;

            r = compare_vectors("v cache[0.." + std::to_string(pos) + "] " + tag, v_prefix_gpu, v_prefix_cpu, verbose);
            stat_v_cache.add(r); step_ok &= r.passed;

            r = compare_vectors("scores " + tag,
                                active_entries(scores_gpu, n_heads, seq_len, active_len),
                                active_entries(scores_cpu, n_heads, seq_len, active_len), verbose);
            stat_scores.add(r); step_ok &= r.passed;

            r = compare_vectors("probs " + tag,
                                active_entries(probs_gpu, n_heads, seq_len, active_len),
                                active_entries(probs_cpu, n_heads, seq_len, active_len), verbose);
            stat_probs.add(r); step_ok &= r.passed;

            r = compare_vectors("attention out " + tag, out_gpu, out_cpu, verbose);
            stat_out.add(r); step_ok &= r.passed;

            if (!step_ok) {
                all_passed = false;
                failed_positions++;
                if (failed_positions >= kMaxFailedPositions) {
                    std::cout << "stopping after " << failed_positions << " failed positions\n";
                    break;
                }
            }
        }

        // Positions the cache must refuse now that the sequence is done.
        const bool rejects_skip = !cache_state.accepts(cache_state.length() + 1);
        const bool rejects_repeat = !cache_state.accepts(cache_state.length() - 1);
        const bool rejects_unwritten = !cache_state.covers(cache_state.length());
        std::cout << "[cache guards] " << ((rejects_skip && rejects_repeat && rejects_unwritten) ? "PASS" : "FAIL")
                  << "  skip=" << (rejects_skip ? "rejected" : "ACCEPTED")
                  << "  repeat=" << (rejects_repeat ? "rejected" : "ACCEPTED")
                  << "  unwritten=" << (rejects_unwritten ? "rejected" : "READABLE") << '\n';
        all_passed &= rejects_skip && rejects_repeat && rejects_unwritten;

        auto summarise = [](const char* name, const DiagnosticStats& s) {
            std::cout << "  " << name << ": " << (s.failed_positions == 0 ? "PASS" : "FAIL")
                      << "  max_abs_error=" << s.max_abs_error
                      << "  mismatches=" << s.mismatches
                      << "  failed_positions=" << s.failed_positions << '\n';
        };
        std::cout << "summary over " << cache_state.length() << " positions:\n";
        summarise("q after rope   ", stat_q);
        summarise("k cache prefix ", stat_k_cache);
        summarise("v cache prefix ", stat_v_cache);
        summarise("scores         ", stat_scores);
        summarise("probabilities  ", stat_probs);
        summarise("attention out  ", stat_out);

        std::cout << (all_passed ? "LAYER0 ATTENTION: ALL PASSED" : "LAYER0 ATTENTION: FAILED") << '\n';
        return all_passed ? 0 : 1;
    }
    catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
