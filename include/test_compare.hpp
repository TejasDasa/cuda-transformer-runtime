#ifndef TEST_COMPARE_HPP
#define TEST_COMPARE_HPP

// Numerical comparison helper shared by the CUDA integration tests.

#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

// Tolerance used for every CPU/GPU comparison:
//     abs(gpu - cpu) <= kAbsTolerance + kRelTolerance * abs(cpu)
constexpr float kAbsTolerance = 1e-5f;
constexpr float kRelTolerance = 1e-5f;

// How many individual mismatches to print per vector before summarising.
constexpr int kMaxReportedMismatches = 10;

// Summary of one CPU-vs-GPU vector comparison.
struct ComparisonResult {
    bool passed = true;
    float max_abs_error = 0.0f;
    double rmse = 0.0;          // root mean square of (gpu - cpu) over finite pairs
    std::size_t mismatches = 0;
    std::size_t nonfinite = 0;
    // When a per-element tolerance is supplied, this counts how many
    // elements would have failed the plain base rule (informational).
    std::size_t base_rule_mismatches = 0;
};

inline float base_tolerance(float cpu_value)
{
    return kAbsTolerance + kRelTolerance * std::abs(cpu_value);
}

// Compares gpu against cpu element by element.  Mismatch details are
// always printed (bounded); the one-line PASS/FAIL summary is printed
// when print_summary is true or the comparison failed.
inline ComparisonResult compare_vectors_with_tolerance(
    const std::string& name,
    const std::vector<float>& gpu,
    const std::vector<float>& cpu,
    const std::vector<float>* allowed_per_element,
    bool print_summary
)
{
    ComparisonResult result;

    if (gpu.size() != cpu.size()) {
        std::cout << "[" << name << "] size mismatch: GPU " << gpu.size()
                  << " vs CPU " << cpu.size() << '\n';
        result.passed = false;
        return result;
    }

    int reported = 0;
    double sum_sq = 0.0;
    std::size_t finite_pairs = 0;

    for (std::size_t i = 0; i < cpu.size(); i++) {
        const float c = cpu[i];
        const float g = gpu[i];

        // A NaN or infinity on either side is always a failure; it must
        // not slip through because abs(NaN - x) compares false.
        if (!std::isfinite(c) || !std::isfinite(g)) {
            result.nonfinite++;
            result.mismatches++;
            result.passed = false;
            if (reported < kMaxReportedMismatches) {
                std::cout << "  [" << name << "] index " << i
                          << ": nonfinite value, CPU " << c
                          << ", GPU " << g << '\n';
                reported++;
            }
            continue;
        }

        const float abs_error = std::abs(g - c);
        if (abs_error > result.max_abs_error) {
            result.max_abs_error = abs_error;
        }
        sum_sq += static_cast<double>(abs_error) * abs_error;
        finite_pairs++;

        const float base_allowed = base_tolerance(c);
        const float allowed = allowed_per_element ? (*allowed_per_element)[i] : base_allowed;
        if (abs_error > base_allowed) {
            result.base_rule_mismatches++;
        }
        if (abs_error > allowed) {
            result.mismatches++;
            result.passed = false;
            if (reported < kMaxReportedMismatches) {
                std::cout << "  [" << name << "] index " << i
                          << ": CPU " << c << ", GPU " << g
                          << ", abs error " << abs_error
                          << ", allowed " << allowed << '\n';
                reported++;
            }
        }
    }

    if (finite_pairs > 0) {
        result.rmse = std::sqrt(sum_sq / static_cast<double>(finite_pairs));
    }

    if (result.mismatches > static_cast<std::size_t>(reported)) {
        std::cout << "  [" << name << "] ... "
                  << (result.mismatches - reported)
                  << " further mismatches not shown\n";
    }

    if (print_summary || !result.passed) {
        std::cout << "[" << name << "] "
                  << (result.passed ? "PASS" : "FAIL")
                  << "  elements=" << cpu.size()
                  << "  max_abs_error=" << result.max_abs_error
                  << "  rmse=" << result.rmse
                  << "  mismatches=" << result.mismatches;
        if (result.nonfinite > 0) {
            std::cout << "  nonfinite=" << result.nonfinite;
        }
        if (allowed_per_element) {
            std::cout << "  base_rule_mismatches=" << result.base_rule_mismatches;
        }
        std::cout << '\n';
    }

    return result;
}

// The standard rule: abs(gpu - cpu) <= 1e-5 + 1e-5 * abs(cpu).
inline ComparisonResult compare_vectors(
    const std::string& name,
    const std::vector<float>& gpu,
    const std::vector<float>& cpu,
    bool print_summary = true
)
{
    return compare_vectors_with_tolerance(name, gpu, cpu, nullptr, print_summary);
}

#endif
