#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

// Относительный допуск для сравнения double
static constexpr double EPS = 1e-9;

static bool nearly_equal(double a, double b) {
    double diff = std::abs(a - b);
    double scale = std::max(1.0, std::max(std::abs(a), std::abs(b)));
    return diff / scale < EPS;
}

static bool check_sin(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) {
        std::cerr << "[FAIL] Cannot open " << filename << "\n";
        return false;
    }

    int count = 0, errors = 0;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        double arg, result;
        if (std::sscanf(line.c_str(), "arg=%lf result=%lf", &arg, &result) != 2) continue;
        double expected = std::sin(arg);
        if (!nearly_equal(result, expected)) {
            std::cerr << "[FAIL] sin(" << arg << "): expected=" << expected
                      << " got=" << result << "\n";
            ++errors;
        }
        ++count;
    }
    std::cout << "[SIN ] Checked " << count << " entries, errors: " << errors << "\n";
    return errors == 0 && count > 0;
}

static bool check_sqrt(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) {
        std::cerr << "[FAIL] Cannot open " << filename << "\n";
        return false;
    }

    int count = 0, errors = 0;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        double arg, result;
        if (std::sscanf(line.c_str(), "arg=%lf result=%lf", &arg, &result) != 2) continue;
        double expected = std::sqrt(arg);
        if (!nearly_equal(result, expected)) {
            std::cerr << "[FAIL] sqrt(" << arg << "): expected=" << expected
                      << " got=" << result << "\n";
            ++errors;
        }
        ++count;
    }
    std::cout << "[SQRT] Checked " << count << " entries, errors: " << errors << "\n";
    return errors == 0 && count > 0;
}

static bool check_pow(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) {
        std::cerr << "[FAIL] Cannot open " << filename << "\n";
        return false;
    }

    int count = 0, errors = 0;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        double base, exp, result;
        if (std::sscanf(line.c_str(), "base=%lf exp=%lf result=%lf", &base, &exp, &result) != 3)
            continue;
        double expected = std::pow(base, exp);
        if (!nearly_equal(result, expected)) {
            std::cerr << "[FAIL] pow(" << base << "," << exp << "): expected=" << expected
                      << " got=" << result << "\n";
            ++errors;
        }
        ++count;
    }
    std::cout << "[POW ] Checked " << count << " entries, errors: " << errors << "\n";
    return errors == 0 && count > 0;
}

int main() {
    bool ok = true;
    ok &= check_sin("results_sin.txt");
    ok &= check_sqrt("results_sqrt.txt");
    ok &= check_pow("results_pow.txt");

    if (ok)
        std::cout << "\nAll checks PASSED\n";
    else
        std::cout << "\nSome checks FAILED\n";

    return ok ? 0 : 1;
}