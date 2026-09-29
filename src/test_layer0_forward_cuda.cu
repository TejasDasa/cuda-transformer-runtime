// COMPLETE LAYER-0 TEST (one transformer layer, not full inference).
//
// Runs a sequence of tokens through all of transformer layer 0 on both
// CPU and GPU and compares every intermediate:
//
//   x_in = embedding[token]  (each token starts from its OWN embedding;
//                             only the KV cache carries state between
//                             positions)
//   attention block -> att_out -> Wo -> x_att = x_in + projected
//   FFN block       -> ffn_norm, h1, h3, gated, ffn_out
//   x_out = x_att + ffn_out
//
// Later layers and the final logits are NOT applied.
//
// Usage:  test_layer0_forward_cuda <checkpoint path> [n_tokens | full]
//   n_tokens defaults to 8; "full" uses the whole context (seq_len).
//
// After the main run the sequence is replayed:
//   * the GPU state is only reset() (its cache buffers keep stale rows)
//   * a brand-new CPU state is used as the reference
//   * a different, shorter token sequence is run
// so any leak of stale cache rows into the replay would be caught.

#include "checkpoint.hpp"
#include "device_buffer.hpp"
#include "layer_forward.hpp"
#include "layer_forward_cuda.hpp"
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
constexpr int kVerbosePositions = 16;
constexpr int kMaxFailedPositions = 3;

struct DiagnosticStats {
    const char* name;
    float max_abs_error = 0.0f;
    std::size_t mismatches = 0;
    int failed_positions = 0;

    explicit DiagnosticStats(const char* n) : name(n) {}

    void add(const ComparisonResult& r)
    {
        max_abs_error = std::max(max_abs_error, r.max_abs_error);
        mismatches += r.mismatches;
        if (!r.passed) failed_positions++;
    }
};

bool parse_count(const char* text, long long& out)
{
    errno = 0;
    char* end = nullptr;
    const long long value = std::strtoll(text, &end, 10);
    if (end == text || *end != '\0' || errno == ERANGE) return false;
    out = value;
    return true;
}

// Everything needed to run and compare one sequence.
struct Harness {
    const ModelWeights& weights;
    const LayerShape& shape;
    const LayerWeights& layer;
    DeviceLayerWeights& d_weights;
    DeviceLayerState& d_state;
    const DeviceBuffer& d_cos;
    const DeviceBuffer& d_sin;
    DeviceBuffer& d_x;
    int half;

    // Runs tokens[0..n) from position 0 against a fresh CPU state.  The
    // GPU state must already be at length 0 (fresh or reset()).
    // Returns true if every comparison passed.
    bool run_sequence(const std::vector<long long>& tokens, const std::string& label, bool verbose,
                      std::vector<DiagnosticStats>& stats)
    {
        CpuLayerState cpu(shape);   // fresh: zero caches, length 0
        bool all_passed = true;
        int failed_positions = 0;
        std::vector<float> x(shape.dim);

        for (int pos = 0; pos < static_cast<int>(tokens.size()); pos++) {
            const std::string tag = label + " pos " + std::to_string(pos);
            const std::size_t row_offset = static_cast<std::size_t>(pos) * half;

            const float* embedding_row =
                weights.token_embedding_table + static_cast<std::size_t>(tokens[pos]) * shape.dim;
            std::copy(embedding_row, embedding_row + shape.dim, x.begin());

            // ----- CPU -----
            if (!layer_forward_cpu(cpu, layer, shape, x.data(),
                                   weights.freq_cis_real + row_offset, weights.freq_cis_imag + row_offset,
                                   pos, kRmsEpsilon)) {
                std::cout << "[" << tag << "] FAIL cpu forward rejected\n";
                return false;
            }

            // ----- GPU (independent: only the embedding is uploaded) -----
            d_x.upload(x.data(), "upload embedding");
            check_cuda(layer_forward_cuda(d_state, d_weights, shape, d_x.data(),
                                          d_cos.data() + row_offset, d_sin.data() + row_offset,
                                          pos, kRmsEpsilon), "layer_forward_cuda");
            check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

            // ----- diagnostics (read-only) -----
            struct Pair { int idx; const DeviceBuffer& d; const std::vector<float>& c; };
            const Pair pairs[] = {
                {0, d_state.att_out, cpu.att_out},
                {1, d_state.projected, cpu.projected},
                {2, d_state.x_att, cpu.x_att},
                {3, d_state.ffn_norm, cpu.ffn_norm},
                {4, d_state.h1, cpu.h1},
                {5, d_state.h3, cpu.h3},
                {6, d_state.gated, cpu.gated},
                {7, d_state.ffn_out, cpu.ffn_out},
                {8, d_state.x_out, cpu.x_out},
            };
            bool step_ok = true;
            for (const Pair& p : pairs) {
                const ComparisonResult r = compare_vectors(
                    std::string(stats[p.idx].name) + " " + tag, p.d.download("download"), p.c, verbose);
                stats[p.idx].add(r);
                step_ok &= r.passed;
            }
            if (!step_ok) {
                all_passed = false;
                if (++failed_positions >= kMaxFailedPositions) {
                    std::cout << "stopping " << label << " after " << failed_positions << " failed positions\n";
                    break;
                }
            }
        }

        // Cache guards on both states.
        const KvCacheState& g = d_state.cache;
        const bool guards = !g.accepts(g.length() + 1) && !g.accepts(g.length() - 1) && !g.covers(g.length()) &&
                            g.length() == cpu.cache.length();
        std::cout << "[" << label << " cache guards] " << (guards ? "PASS" : "FAIL")
                  << "  length=" << g.length() << '\n';
        return all_passed && guards;
    }
};

std::vector<DiagnosticStats> make_stats()
{
    return {DiagnosticStats("attention out"), DiagnosticStats("wo projection"), DiagnosticStats("x_att residual"),
            DiagnosticStats("ffn norm"), DiagnosticStats("h1 (W1)"), DiagnosticStats("h3 (W3)"),
            DiagnosticStats("silu gated"), DiagnosticStats("ffn out (W2)"), DiagnosticStats("x_out final")};
}

void print_stats(const char* label, int positions, const std::vector<DiagnosticStats>& stats)
{
    std::cout << label << " summary over " << positions << " positions:\n";
    for (const DiagnosticStats& s : stats) {
        std::cout << "  " << s.name;
        for (std::size_t i = std::strlen(s.name); i < 16; i++) std::cout << ' ';
        std::cout << ": " << (s.failed_positions == 0 ? "PASS" : "FAIL")
                  << "  max_abs_error=" << s.max_abs_error
                  << "  mismatches=" << s.mismatches
                  << "  failed_positions=" << s.failed_positions << '\n';
    }
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

        const LayerShape shape = LayerShape::from_config(config);
        if (!shape.valid()) {
            std::cerr << "Error: checkpoint layer shape is invalid\n";
            return 1;
        }
        const int half = shape.head_size / 2;

        std::int64_t vocab_size = config.vocab_size;
        if (vocab_size < 0) vocab_size = -vocab_size;   // sign only marks classifier sharing

        long long n_tokens = kDefaultTokens;
        if (argc == 3) {
            if (std::strcmp(argv[2], "full") == 0) {
                n_tokens = shape.seq_len;
            } else if (!parse_count(argv[2], n_tokens)) {
                std::cerr << "Error: n_tokens '" << argv[2] << "' is not an integer\n";
                return 1;
            }
        }
        if (n_tokens < 1 || n_tokens > shape.seq_len) {
            std::cerr << "Error: n_tokens must be in [1, " << shape.seq_len << "]\n";
            return 1;
        }

        std::vector<long long> tokens(n_tokens);
        for (long long i = 0; i < n_tokens; i++) tokens[i] = (i * 7919 + 42) % vocab_size;

        // Replay sequence: different tokens, at most half as long, at least 1.
        const long long n_replay = std::max<long long>(1, n_tokens / 2);
        std::vector<long long> replay(n_replay);
        for (long long i = 0; i < n_replay; i++) replay[i] = (i * 104729 + 1234) % vocab_size;

        std::cout << "COMPLETE LAYER-0 TEST (attention block + Wo + residual + SwiGLU FFN + residual)\n"
                  << "checkpoint: " << argv[1] << '\n'
                  << "dim=" << shape.dim << " hidden_dim=" << shape.hidden_dim
                  << " n_heads=" << shape.n_heads << " n_kv_heads=" << shape.n_kv_heads
                  << " head_size=" << shape.head_size << " kv_dim=" << shape.kv_dim
                  << " seq_len=" << shape.seq_len << " vocab_size=" << vocab_size
                  << " n_tokens=" << n_tokens << " replay_tokens=" << n_replay << '\n';

        // Layer 0 slices (the mapped pointers already start at layer 0).
        const LayerWeights layer = select_layer_weights(weights, shape, 0);

        // GPU allocations, all outside the token loop.
        DeviceLayerWeights d_weights(shape);
        d_weights.upload(layer, shape);
        DeviceLayerState d_state(shape);
        const std::size_t table_count = static_cast<std::size_t>(shape.seq_len) * half;
        DeviceBuffer d_cos(table_count), d_sin(table_count), d_x(shape.dim);
        d_cos.upload(weights.freq_cis_real, "upload freq_cis_real");
        d_sin.upload(weights.freq_cis_imag, "upload freq_cis_imag");

        Harness h{weights, shape, layer, d_weights, d_state, d_cos, d_sin, d_x, half};

        // Main run.
        std::vector<DiagnosticStats> main_stats = make_stats();
        const bool main_ok = h.run_sequence(tokens, "main", n_tokens <= kVerbosePositions, main_stats);
        print_stats("main", d_state.cache.length(), main_stats);

        // Reset and replay with stale GPU cache bytes still in place.
        d_state.reset();
        std::vector<DiagnosticStats> replay_stats = make_stats();
        const bool replay_ok = h.run_sequence(replay, "replay", n_replay <= kVerbosePositions, replay_stats);
        print_stats("replay", d_state.cache.length(), replay_stats);

        const bool all_passed = main_ok && replay_ok;
        std::cout << (all_passed ? "LAYER0 FORWARD: ALL PASSED" : "LAYER0 FORWARD: FAILED") << '\n';
        return all_passed ? 0 : 1;
    }
    catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
