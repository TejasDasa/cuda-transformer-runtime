// FULL-MODEL SINGLE-TOKEN FORWARD TEST (all layers + final norm + logits).
//
// Runs a fixed, deterministic token sequence through the whole model on
// CPU and GPU and compares, at every position:
//   * the output of each transformer layer
//   * the final normalised vector
//   * every vocabulary logit
// Tokens are chosen up front; neither model's predictions pick the next
// token.  Text generation is NOT implemented here.
//
// Usage:  test_model_forward_cuda <checkpoint> [n_tokens | full] [logits dump path]
//   n_tokens defaults to 8; "full" uses the whole context.
//   If a dump path is given, the GPU logits of the main sequence are
//   written as text ("pos token logit0 logit1 ...") for external checks.
//
// After the main run, both models are reset and a different shorter
// sequence is replayed; the GPU replay is compared with the CPU replay
// and, bit for bit, with a freshly constructed GPU model.

#include "checkpoint.hpp"
#include "model_forward.hpp"
#include "model_forward_cuda.hpp"
#include "test_compare.hpp"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int kDefaultTokens = 8;
constexpr int kVerbosePositions = 8;
constexpr int kMaxFailedPositions = 3;

struct DiagnosticStats {
    std::string name;
    float max_abs_error = 0.0f;
    double max_rmse = 0.0;
    std::size_t mismatches = 0;
    int failed_positions = 0;

    explicit DiagnosticStats(std::string n) : name(std::move(n)) {}

    void add(const ComparisonResult& r)
    {
        max_abs_error = std::max(max_abs_error, r.max_abs_error);
        max_rmse = std::max(max_rmse, r.rmse);
        mismatches += r.mismatches;
        base_rule_mismatches += r.base_rule_mismatches;
        if (!r.passed) failed_positions++;
    }
    std::size_t base_rule_mismatches = 0;
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

std::vector<int> top_k(const std::vector<float>& v, int k)
{
    std::vector<int> idx(v.size());
    for (std::size_t i = 0; i < v.size(); i++) idx[i] = static_cast<int>(i);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) { return v[a] > v[b]; });
    idx.resize(k);
    return idx;
}

std::vector<DiagnosticStats> make_stats(int n_layers)
{
    std::vector<DiagnosticStats> s;
    for (int l = 0; l < n_layers; l++) s.emplace_back("layer " + std::to_string(l) + " out");
    s.emplace_back("final norm");
    s.emplace_back("logits");
    return s;
}

void print_stats(const char* label, int positions, const std::vector<DiagnosticStats>& stats)
{
    std::cout << label << " summary over " << positions << " positions:\n";
    for (const DiagnosticStats& s : stats) {
        std::cout << "  " << std::left << std::setw(14) << s.name << std::right
                  << ": " << (s.failed_positions == 0 ? "PASS" : "FAIL")
                  << "  max_abs_error=" << s.max_abs_error
                  << "  max_rmse=" << s.max_rmse
                  << "  mismatches=" << s.mismatches
                  << "  base_rule_mismatches=" << s.base_rule_mismatches
                  << "  failed_positions=" << s.failed_positions << '\n';
    }
}

// ---------------------------------------------------------------------
// Stage-specific tolerances for the final norm and the logits.
//
// Every layer output is held to the base rule  1e-5 + 1e-5 * |cpu|.  The
// final RMSNorm then multiplies component i of the last layer output by
// g_i * scale, where g = rms_final_weight (max |g| = 9.9 in Stories15M,
// scale up to ~1.4).  A component that is small enough to sit inside the
// base rule's 1e-5 absolute floor therefore comes out of the norm with an
// absolute difference of up to 1e-5 * |g_i| * scale, which the plain base
// rule rejects (observed: 3.6e-5 at |g*scale| ~ 14).  Investigation
// showed each side's final norm matches a double-precision norm of its
// OWN input to within 1.1e-5, so this is propagated float accumulation
// across the layers, not a defect in the norm or the classifier.
//
// Rather than loosen the constants, the base rule is propagated exactly:
//   final norm : allowed_i = 1e-5 * |out_i| + 1e-5 * |g_i| * scale
//   logits     : allowed_v = 1e-5 * |logit_v|
//                          + sqrt( sum_i (wcls[v,i] * allowed_fn_i)^2 )
// The logits term combines the per-element final-norm slack through the
// classifier row in quadrature, because independent rounding errors add
// as a root-sum-square rather than aligning.  The number of elements that
// would have failed the plain base rule is still reported.
// ---------------------------------------------------------------------
struct DerivedTolerances {
    std::vector<float> final_norm;   // [dim]
    std::vector<float> logits;       // [vocab_count]
    float max_gain_scale = 0.0f;     // max_i |g_i| * scale, for the report
};

DerivedTolerances derive_tolerances(const CpuModel& cpu, const ModelWeights& weights)
{
    const ModelShape& shape = cpu.shape();
    const int dim = shape.layer.dim;
    const int n_layers = shape.n_layers;
    const std::vector<float>& x = cpu.layer_output(n_layers - 1);

    double sum_sq = 0.0;
    for (float v : x) sum_sq += static_cast<double>(v) * v;
    const double scale = 1.0 / std::sqrt(sum_sq / dim + kModelRmsEpsilon);

    DerivedTolerances t;
    t.final_norm.resize(dim);
    for (int i = 0; i < dim; i++) {
        const double gain = std::abs(static_cast<double>(weights.rms_final_weight[i])) * scale;
        t.max_gain_scale = std::max(t.max_gain_scale, static_cast<float>(gain));
        t.final_norm[i] = static_cast<float>(kRelTolerance * std::abs(cpu.final_normalized()[i]) + kAbsTolerance * gain);
    }

    const std::size_t vocab = static_cast<std::size_t>(shape.vocab_count);
    t.logits.resize(vocab);
    for (std::size_t v = 0; v < vocab; v++) {
        const float* row = weights.wcls + v * dim;
        double q = 0.0;
        for (int i = 0; i < dim; i++) {
            const double term = static_cast<double>(row[i]) * t.final_norm[i];
            q += term * term;
        }
        t.logits[v] = static_cast<float>(kRelTolerance * std::abs(cpu.logits()[v]) + std::sqrt(q));
    }
    return t;
}

// Runs one sequence on both models (both must be at position 0) and
// compares every diagnostic.  Optionally dumps GPU logits.
bool run_sequence(CpuModel& cpu, GpuModel& gpu, const ModelWeights& weights,
                  const std::vector<long long>& tokens,
                  const std::string& label, bool verbose, std::vector<DiagnosticStats>& stats,
                  std::ofstream* dump, std::vector<std::vector<float>>* gpu_logits_out)
{
    const int n_layers = cpu.shape().n_layers;
    bool all_passed = true;
    int failed_positions = 0;
    float max_gain_scale = 0.0f;
    float max_logit_tolerance = 0.0f;

    for (int pos = 0; pos < static_cast<int>(tokens.size()); pos++) {
        const std::string tag = label + " pos " + std::to_string(pos);

        if (!cpu.forward(tokens[pos], pos)) {
            std::cout << "[" << tag << "] FAIL cpu forward rejected\n";
            return false;
        }
        check_cuda(gpu.forward(tokens[pos], pos), "gpu forward");
        check_cuda(gpu.synchronize(), "gpu synchronize");

        const DerivedTolerances tol = derive_tolerances(cpu, weights);
        max_gain_scale = std::max(max_gain_scale, tol.max_gain_scale);
        for (float a : tol.logits) max_logit_tolerance = std::max(max_logit_tolerance, a);

        bool step_ok = true;
        for (int l = 0; l < n_layers; l++) {
            const ComparisonResult r = compare_vectors(stats[l].name + " " + tag,
                gpu.layer_output(l).download("layer output"), cpu.layer_output(l), verbose);
            stats[l].add(r);
            step_ok &= r.passed;
        }
        {
            const ComparisonResult r = compare_vectors_with_tolerance("final norm " + tag,
                gpu.final_normalized().download("final norm"), cpu.final_normalized(), &tol.final_norm, verbose);
            stats[n_layers].add(r);
            step_ok &= r.passed;
        }
        const std::vector<float> gpu_logits = gpu.logits().download("logits");
        {
            const ComparisonResult r = compare_vectors_with_tolerance("logits " + tag, gpu_logits, cpu.logits(), &tol.logits, verbose);
            stats[n_layers + 1].add(r);
            step_ok &= r.passed;
        }
        if (gpu_logits_out) gpu_logits_out->push_back(gpu_logits);
        if (dump) {
            *dump << pos << ' ' << tokens[pos];
            for (float v : gpu_logits) *dump << ' ' << std::setprecision(9) << v;
            *dump << '\n';
        }

        if (!step_ok) {
            all_passed = false;
            if (++failed_positions >= kMaxFailedPositions) {
                std::cout << "stopping " << label << " after " << failed_positions << " failed positions\n";
                break;
            }
        }
    }

    // Diagnostic only: top-5 predictions at the last position, both sides.
    const std::vector<int> top_cpu = top_k(cpu.logits(), 5);
    const std::vector<int> top_gpu = top_k(gpu.logits().download("logits"), 5);
    std::cout << "[" << label << "] top-5 token ids after last position: cpu";
    for (int t : top_cpu) std::cout << ' ' << t;
    std::cout << "  gpu";
    for (int t : top_gpu) std::cout << ' ' << t;
    std::cout << '\n';

    std::cout << "[" << label << " derived tolerances] max |g|*scale=" << max_gain_scale
              << "  max final-norm allowed=" << kAbsTolerance * max_gain_scale << "+rel"
              << "  max logit allowed=" << max_logit_tolerance << '\n';

    const bool positions_agree = cpu.position() == gpu.position() &&
                                 cpu.position() == static_cast<int>(tokens.size());
    std::cout << "[" << label << " position] " << (positions_agree ? "PASS" : "FAIL")
              << "  cpu=" << cpu.position() << " gpu=" << gpu.position() << '\n';
    return all_passed && positions_agree;
}

} // namespace

int main(int argc, char* argv[])
{
    if (argc < 2 || argc > 4) {
        std::cerr << "Usage: " << argv[0] << " <checkpoint path> [n_tokens | full] [logits dump path]\n";
        return 1;
    }

    try {
        Checkpoint checkpoint{argv[1]};
        const ModelConfig& config = checkpoint.config();
        const ModelWeights& weights = checkpoint.weights();
        const ModelShape shape = ModelShape::from_config(config);
        if (!shape.valid()) {
            std::cerr << "Error: checkpoint model shape is invalid\n";
            return 1;
        }
        const int seq_len = shape.layer.seq_len;

        long long n_tokens = kDefaultTokens;
        if (argc >= 3) {
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

        std::ofstream dump;
        if (argc == 4) {
            dump.open(argv[3]);
            if (!dump) {
                std::cerr << "Error: cannot open dump path " << argv[3] << '\n';
                return 1;
            }
        }

        std::vector<long long> tokens(n_tokens);
        for (long long i = 0; i < n_tokens; i++) tokens[i] = (i * 7919 + 42) % shape.vocab_count;
        const long long n_replay = std::max<long long>(1, n_tokens / 2);
        std::vector<long long> replay(n_replay);
        for (long long i = 0; i < n_replay; i++) replay[i] = (i * 104729 + 1234) % shape.vocab_count;

        std::cout << "FULL-MODEL FORWARD TEST (" << shape.n_layers << " layers + final norm + logits; no generation)\n"
                  << "checkpoint: " << argv[1] << '\n'
                  << "dim=" << shape.layer.dim << " hidden_dim=" << shape.layer.hidden_dim
                  << " n_layers=" << shape.n_layers << " n_heads=" << shape.layer.n_heads
                  << " n_kv_heads=" << shape.layer.n_kv_heads << " seq_len=" << seq_len
                  << " vocab_count=" << shape.vocab_count
                  << " classifier=" << (shape.shared_classifier ? "tied" : "untied")
                  << " n_tokens=" << n_tokens << " replay_tokens=" << n_replay << '\n'
                  << "tokens:";
        for (long long i = 0; i < std::min<long long>(n_tokens, 16); i++) std::cout << ' ' << tokens[i];
        std::cout << (n_tokens > 16 ? " ...\n" : "\n");

        CpuModel cpu(config, weights);
        GpuModel gpu(config, weights, /*keep_layer_outputs=*/true);
        if (gpu.classifier_is_tied() != shape.shared_classifier) {
            std::cout << "[classifier] FAIL tied flag mismatch\n";
            return 1;
        }

        std::vector<DiagnosticStats> main_stats = make_stats(shape.n_layers);
        const bool main_ok = run_sequence(cpu, gpu, weights, tokens, "main", n_tokens <= kVerbosePositions, main_stats,
                                          dump.is_open() ? &dump : nullptr, nullptr);
        print_stats("main", cpu.position(), main_stats);

        // Reset + replay.  The fresh GPU model gives the bitwise reference.
        cpu.reset();
        gpu.reset();
        std::vector<DiagnosticStats> replay_stats = make_stats(shape.n_layers);
        std::vector<std::vector<float>> replay_logits;
        const bool replay_ok = run_sequence(cpu, gpu, weights, replay, "replay", n_replay <= kVerbosePositions, replay_stats,
                                            nullptr, &replay_logits);
        print_stats("replay", cpu.position(), replay_stats);

        GpuModel fresh(config, weights);
        std::size_t replay_diffs = 0;
        for (int pos = 0; pos < static_cast<int>(replay_logits.size()); pos++) {
            check_cuda(fresh.forward(replay[pos], pos), "fresh forward");
            check_cuda(fresh.synchronize(), "fresh synchronize");
            const std::vector<float> f = fresh.logits().download("fresh logits");
            for (std::size_t i = 0; i < f.size(); i++) replay_diffs += (f[i] != replay_logits[pos][i]);
        }
        const bool fresh_ok = replay_diffs == 0 && replay_logits.size() == static_cast<std::size_t>(n_replay);
        std::cout << "[replay vs fresh gpu model] " << (fresh_ok ? "PASS" : "FAIL")
                  << "  differing logits=" << replay_diffs << '\n';

        const bool all_passed = main_ok && replay_ok && fresh_ok;
        std::cout << (all_passed ? "FULL MODEL FORWARD: ALL PASSED" : "FULL MODEL FORWARD: FAILED") << '\n';
        return all_passed ? 0 : 1;
    }
    catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
