// Integration test for the first slice of a transformer layer on the GPU:
//
//     token embedding -> layer 0 attention RMSNorm -> Q / K / V projections
//                     -> RoPE on Q and K
//
// Every GPU result is compared against the CPU reference implementation
// using the real Stories15M checkpoint weights.  Attention, the FFN and
// everything after them are not exercised here.
//
// Two independent inputs drive the test:
//   * the token id selects WHICH embedding row is fed in (what the token is)
//   * the sequence position selects WHICH RoPE rotation is applied (where
//     the token sits in the context)
// The token comes from the command line; positions are a fixed list.
//
// Usage:  test_layer0_cuda <checkpoint path> [token id]
// Exit code is 0 only if every check passes.

#include "checkpoint.hpp"
#include "cpu_ops.hpp"
#include "cuda_ops.hpp"
#include "device_buffer.hpp"
#include "test_compare.hpp"

#include <cuda_runtime.h>

#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr float kRmsEpsilon = 1e-5f;

// Exact (bit-for-bit) equality, used to prove a buffer was not touched.
bool report_unchanged(
    const std::string& name,
    const std::vector<float>& after,
    const std::vector<float>& before
)
{
    std::size_t changed = 0;
    if (after.size() != before.size()) {
        changed = before.size();
    } else {
        for (std::size_t i = 0; i < before.size(); i++) {
            if (after[i] != before[i]) {
                changed++;
            }
        }
    }
    std::cout << "[" << name << "] "
              << (changed == 0 ? "PASS" : "FAIL")
              << "  unchanged elements=" << (before.size() - changed)
              << "/" << before.size() << '\n';
    return changed == 0;
}

// Parses the optional token argument.  Rejects anything that is not a
// whole base-10 integer so "42abc" or "" cannot silently become 42 or 0.
bool parse_token_id(const char* text, long long& out)
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

// Host-side boundary for RoPE: a position may only select a row that
// exists in the [seq_len x head_size/2] tables.
bool position_in_range(long long position, int seq_len)
{
    return position >= 0 && position < seq_len;
}

// ---------------------------------------------------------------------
// RoPE table convention.
//
// The checkpoint stores two float tables, freq_cis_real and freq_cis_imag,
// each laid out row-major as [seq_len, head_size / 2]:
//
//     cos_table[position * (head_size/2) + pair]
//       = cos(position * 10000^(-2 * pair / head_size))
//     sin_table[...] = sin(same angle)
//
// i.e. the llama2.c / original Llama convention where pair p of every
// head rotates by an angle that grows linearly with position and shrinks
// geometrically with p.  This check re-derives the tables from that
// formula and fails if the checkpoint disagrees, so a checkpoint with a
// different layout (e.g. transposed) or a different RoPE base cannot be
// silently mis-read.
//
// Limitation: the tables alone cannot reveal whether the model was
// trained with adjacent-pair rotation (x[2p], x[2p+1]) or with the
// "rotate half" pairing (x[p], x[p + head_size/2]).  llama2.c, whose
// exporter wrote this file, uses adjacent pairs, and that is what rope()
// implements.  Only a full forward pass producing sensible text can
// confirm that choice end to end, which is a later stage.
//
// The exporter computed the angles in float32, so at large positions
// the stored values drift from the double-precision formula by up to
// about 1e-5; the tolerance here allows for that but nothing coarser.
// ---------------------------------------------------------------------
bool verify_rope_tables(const ModelConfig& config, const ModelWeights& weights)
{
    const int head_size = config.dim / config.n_heads;
    const int half = head_size / 2;
    constexpr double kFormulaTolerance = 1e-4;
    constexpr double kNormTolerance = 1e-5;

    double max_formula_error = 0.0;
    double max_norm_error = 0.0;
    bool row0_is_identity = true;

    for (int position = 0; position < config.seq_len; position++) {
        for (int pair = 0; pair < half; pair++) {
            const std::size_t index = static_cast<std::size_t>(position) * half + pair;
            const double c = weights.freq_cis_real[index];
            const double s = weights.freq_cis_imag[index];

            const double freq = 1.0 / std::pow(10000.0, (2.0 * pair) / head_size);
            const double angle = position * freq;

            max_formula_error = std::max(max_formula_error,
                std::max(std::abs(c - std::cos(angle)), std::abs(s - std::sin(angle))));
            max_norm_error = std::max(max_norm_error, std::abs(c * c + s * s - 1.0));

            if (position == 0 && (c != 1.0 || s != 0.0)) {
                row0_is_identity = false;
            }
        }
    }

    const bool passed =
        row0_is_identity &&
        max_formula_error <= kFormulaTolerance &&
        max_norm_error <= kNormTolerance;

    std::cout << "[rope tables] " << (passed ? "PASS" : "FAIL")
              << "  layout=[" << config.seq_len << " x " << half << "]"
              << "  row0_identity=" << (row0_is_identity ? "yes" : "no")
              << "  max_formula_error=" << max_formula_error
              << "  max_norm_error=" << max_norm_error << '\n';
    return passed;
}

} // namespace

int main(int argc, char* argv[])
{
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <checkpoint path> [token id]\n";
        return 1;
    }

    // Token to embed.  Defaults to 42 so the test is runnable with just
    // the checkpoint path.
    long long token_id = 42;
    if (argc == 3 && !parse_token_id(argv[2], token_id)) {
        std::cerr << "Error: token id '" << argv[2]
                  << "' is not an integer\n";
        return 1;
    }

    try {
        Checkpoint checkpoint{argv[1]};
        const ModelConfig& config = checkpoint.config();
        const ModelWeights& weights = checkpoint.weights();

        // ---------------------------------------------------------------
        // Dimensions.
        //
        // vocab_size is stored signed: a negative value means the output
        // classifier has its own weights instead of sharing the embedding
        // table.  That sign was already honoured while mapping the file,
        // so here only the magnitude matters.  Widen to 64 bits before
        // negating so INT32_MIN cannot overflow.
        // ---------------------------------------------------------------
        std::int64_t vocab_size = config.vocab_size;
        if (vocab_size < 0) {
            vocab_size = -vocab_size;
        }

        if (token_id < 0 || token_id >= vocab_size) {
            std::cerr << "Error: token id " << token_id
                      << " is outside the vocabulary [0, "
                      << vocab_size << ")\n";
            return 1;
        }

        const int dim = config.dim;                       // model width
        const int head_size = dim / config.n_heads;       // width per head
        const int kv_dim = head_size * config.n_kv_heads; // K and V width
        const int half = head_size / 2;                   // pairs per head
        const int seq_len = config.seq_len;               // RoPE table rows

        if (head_size % 2 != 0) {
            std::cerr << "Error: head_size " << head_size
                      << " is odd; RoPE needs whole pairs\n";
            return 1;
        }

        std::cout << "checkpoint: " << argv[1] << '\n'
                  << "dim=" << dim
                  << " n_heads=" << config.n_heads
                  << " n_kv_heads=" << config.n_kv_heads
                  << " head_size=" << head_size
                  << " kv_dim=" << kv_dim
                  << " seq_len=" << seq_len
                  << " vocab_size=" << vocab_size
                  << " token=" << token_id << '\n';

        bool all_passed = true;

        all_passed &= verify_rope_tables(config, weights);

        // ---------------------------------------------------------------
        // Host side: embedding lookup and CPU reference.
        //
        // token_embedding_table is [vocab_size x dim], row-major, so the
        // embedding for our token is one contiguous row of `dim` floats.
        // ---------------------------------------------------------------
        const float* embedding_row =
            weights.token_embedding_table +
            static_cast<std::size_t>(token_id) * dim;

        std::vector<float> x(embedding_row, embedding_row + dim);

        // Layer 0 weights.  The checkpoint pointers already start at
        // layer 0, so the first `count` floats of each tensor are exactly
        // that layer's slice:
        //   rms_att_weight : [dim]
        //   wq             : [dim    x dim]  (dim rows, dim cols)
        //   wk             : [kv_dim x dim]
        //   wv             : [kv_dim x dim]
        const std::size_t rms_count = static_cast<std::size_t>(dim);
        const std::size_t wq_count  = static_cast<std::size_t>(dim) * dim;
        const std::size_t wk_count  = static_cast<std::size_t>(kv_dim) * dim;
        const std::size_t wv_count  = wk_count;

        // RoPE tables: [seq_len x half] each, shared by every layer.
        const std::size_t table_count = static_cast<std::size_t>(seq_len) * half;

        std::vector<float> normalized_cpu(dim);
        std::vector<float> q_cpu(dim);
        std::vector<float> k_cpu(kv_dim);
        std::vector<float> v_cpu(kv_dim);

        rmsnorm(normalized_cpu.data(), x.data(), weights.rms_att_weight,
                dim, kRmsEpsilon);
        matvec(q_cpu.data(), weights.wq, normalized_cpu.data(), dim, dim);
        matvec(k_cpu.data(), weights.wk, normalized_cpu.data(), kv_dim, dim);
        matvec(v_cpu.data(), weights.wv, normalized_cpu.data(), kv_dim, dim);

        // ---------------------------------------------------------------
        // Device side.
        //
        // Each DeviceBuffer owns its allocation; they are freed in reverse
        // order when this block ends, however it ends.
        // Inputs (uploaded once):
        //   d_x      [dim]           token embedding
        //   d_rms    [dim]           layer 0 RMSNorm gain
        //   d_wq     [dim x dim]
        //   d_wk     [kv_dim x dim]
        //   d_wv     [kv_dim x dim]
        //   d_cos    [seq_len x half] RoPE cosine table
        //   d_sin    [seq_len x half] RoPE sine table
        // Outputs (written by kernels, read back afterwards):
        //   d_normalized [dim]       stays on the GPU between kernels
        //   d_q          [dim]       projected, later rotated in place
        //   d_k          [kv_dim]    projected, later rotated in place
        //   d_v          [kv_dim]    projected, never rotated
        // ---------------------------------------------------------------
        DeviceBuffer d_x(rms_count);
        DeviceBuffer d_rms(rms_count);
        DeviceBuffer d_wq(wq_count);
        DeviceBuffer d_wk(wk_count);
        DeviceBuffer d_wv(wv_count);
        DeviceBuffer d_cos(table_count);
        DeviceBuffer d_sin(table_count);
        DeviceBuffer d_normalized(rms_count);
        DeviceBuffer d_q(static_cast<std::size_t>(dim));
        DeviceBuffer d_k(static_cast<std::size_t>(kv_dim));
        DeviceBuffer d_v(static_cast<std::size_t>(kv_dim));

        d_x.upload(x.data(), "upload embedding");
        d_rms.upload(weights.rms_att_weight, "upload rms_att_weight");
        d_wq.upload(weights.wq, "upload wq");
        d_wk.upload(weights.wk, "upload wk");
        d_wv.upload(weights.wv, "upload wv");
        d_cos.upload(weights.freq_cis_real, "upload freq_cis_real");
        d_sin.upload(weights.freq_cis_imag, "upload freq_cis_imag");

        // Kernel ordering.  All launches go to the default stream, which
        // executes them in issue order, so the three matvec kernels are
        // guaranteed to see the finished d_normalized without any
        // explicit synchronisation in between.  Each wrapper returns only
        // the *launch* status (bad configuration, etc.); execution errors
        // surface at the cudaDeviceSynchronize below.
        check_cuda(
            rmsnorm_cuda(d_normalized.data(), d_x.data(), d_rms.data(),
                         dim, kRmsEpsilon),
            "launch rmsnorm_cuda"
        );
        check_cuda(
            matvec_cuda(d_q.data(), d_wq.data(), d_normalized.data(),
                        dim, dim),
            "launch matvec_cuda (q)"
        );
        check_cuda(
            matvec_cuda(d_k.data(), d_wk.data(), d_normalized.data(),
                        kv_dim, dim),
            "launch matvec_cuda (k)"
        );
        check_cuda(
            matvec_cuda(d_v.data(), d_wv.data(), d_normalized.data(),
                        kv_dim, dim),
            "launch matvec_cuda (v)"
        );

        // Wait for the whole sequence and pick up any execution error
        // (e.g. an out-of-bounds access inside a kernel).
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        // ---------------------------------------------------------------
        // Stage 1: pre-RoPE results.
        // ---------------------------------------------------------------
        const std::vector<float> normalized_gpu =
            d_normalized.download("download normalized");
        const std::vector<float> q_gpu = d_q.download("download q");
        const std::vector<float> k_gpu = d_k.download("download k");
        const std::vector<float> v_gpu = d_v.download("download v");

        all_passed &= compare_vectors("normalized", normalized_gpu,
                                      normalized_cpu).passed;
        all_passed &= compare_vectors("q", q_gpu, q_cpu).passed;
        all_passed &= compare_vectors("k", k_gpu, k_cpu).passed;
        all_passed &= compare_vectors("v", v_gpu, v_cpu).passed;

        // ---------------------------------------------------------------
        // Stage 2: RoPE on Q and K at several sequence positions.
        //
        // Why pairs within a head: RoPE encodes position by rotating the
        // query and key vectors so that the dot product q.k in attention
        // depends only on the *distance* between two positions.  A 2-D
        // rotation is the simplest operation with that property, so each
        // head's head_size numbers are treated as head_size/2 little 2-D
        // vectors (adjacent pairs) and each pair is rotated.  Pair p
        // turns by position * freq_p, so low pairs spin fast and encode
        // fine-grained position while high pairs spin slowly and encode
        // coarse position.  V is not rotated because it carries content,
        // not position.
        //
        // Selecting the rotation: position picks a row of the cos/sin
        // tables; row `position` starts at table + position * half on
        // both the host and the device, so the CPU and GPU consume the
        // same head_size/2 angles.
        //
        // Each position gets fresh, unrotated projections: on the CPU by
        // copying q_cpu/k_cpu, on the GPU by re-running the two matvecs
        // into d_q/d_k.  Q and K then stay resident on the device between
        // matvec and RoPE; only the rotated results come back.
        // ---------------------------------------------------------------
        const long long positions[] = {0, 1, 7, static_cast<long long>(seq_len) - 1};

        for (const long long position : positions) {
            if (!position_in_range(position, seq_len)) {
                std::cout << "[rope pos " << position
                          << "] FAIL  position outside [0, " << seq_len << ")\n";
                all_passed = false;
                continue;
            }

            const std::size_t row_offset = static_cast<std::size_t>(position) * half;
            const std::string tag = "pos " + std::to_string(position);

            // CPU reference: rotate copies so q_cpu / k_cpu stay pristine.
            std::vector<float> q_rot_cpu = q_cpu;
            std::vector<float> k_rot_cpu = k_cpu;
            rope(q_rot_cpu.data(), config.n_heads, head_size,
                 weights.freq_cis_real + row_offset,
                 weights.freq_cis_imag + row_offset);
            rope(k_rot_cpu.data(), config.n_kv_heads, head_size,
                 weights.freq_cis_real + row_offset,
                 weights.freq_cis_imag + row_offset);

            // GPU: fresh projections, then rotate in place.  Q and K are
            // separate launches with their own head counts.
            check_cuda(
                matvec_cuda(d_q.data(), d_wq.data(), d_normalized.data(),
                            dim, dim),
                "launch matvec_cuda (q, rope stage)"
            );
            check_cuda(
                matvec_cuda(d_k.data(), d_wk.data(), d_normalized.data(),
                            kv_dim, dim),
                "launch matvec_cuda (k, rope stage)"
            );
            check_cuda(
                rope_cuda(d_q.data(), config.n_heads, head_size,
                          d_cos.data() + row_offset, d_sin.data() + row_offset),
                "launch rope_cuda (q)"
            );
            check_cuda(
                rope_cuda(d_k.data(), config.n_kv_heads, head_size,
                          d_cos.data() + row_offset, d_sin.data() + row_offset),
                "launch rope_cuda (k)"
            );
            check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize (rope)");

            const std::vector<float> q_rot_gpu = d_q.download("download rotated q");
            const std::vector<float> k_rot_gpu = d_k.download("download rotated k");
            const std::vector<float> v_after   = d_v.download("download v after rope");

            all_passed &= compare_vectors("q rope " + tag, q_rot_gpu, q_rot_cpu).passed;
            all_passed &= compare_vectors("k rope " + tag, k_rot_gpu, k_rot_cpu).passed;
            all_passed &= report_unchanged("v after " + tag, v_after, v_gpu);
        }

        // Positions that must be rejected before any table row is read.
        const long long bad_positions[] = {-1, seq_len};
        for (const long long position : bad_positions) {
            const bool rejected = !position_in_range(position, seq_len);
            std::cout << "[reject pos " << position << "] "
                      << (rejected ? "PASS" : "FAIL") << '\n';
            all_passed &= rejected;
        }

        std::cout << (all_passed ? "LAYER0 QKV+ROPE: ALL PASSED"
                                 : "LAYER0 QKV+ROPE: FAILED")
                  << '\n';

        return all_passed ? 0 : 1;
    }
    catch (const std::exception& error) {
        // DeviceBuffer destructors have already run during unwinding.
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
