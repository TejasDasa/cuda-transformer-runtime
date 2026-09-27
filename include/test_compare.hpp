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
    std::size_t mismatches = 0;
    std::size_t nonfinite = 0;
};

// Compares gpu against cpu element by element.  Mismatch details are
// always printed (bounded); the one-line PASS/FAIL summary is printed
// when print_summary is true or the comparison failed.
inline ComparisonResult compare_vectors(
    const std::string& name,
    const std::vector<float>& gpu,
    const std::vector<float>& cpu,
    bool print_summary = true
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

        const float allowed = kAbsTolerance + kRelTolerance * std::abs(c);
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
                  << "  mismatches=" << result.mismatches;
        if (result.nonfinite > 0) {
            std::cout << "  nonfinite=" << result.nonfinite;
        }
        std::cout << '\n';
    }

    return result;
}

#endif
