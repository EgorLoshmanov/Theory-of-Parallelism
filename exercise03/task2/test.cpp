#include <cmath>
#include <fstream>
#include <iostream>
#include <string>

static constexpr double EPS = 1e-9;

static bool check_sin(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) {
        std::cerr << "Cannot open " << filename << "\n";
        return false;
    }
    double arg, result;
    int count = 0, errors = 0;
    while (in >> arg >> result) {
        double expected = std::sin(arg);
        if (std::abs(result - expected) > EPS) {
            std::cerr << "sin(" << arg << ") = " << result
                      << ", expected " << expected << "\n";
            ++errors;
        }
        ++count;
    }
    std::cout << filename << ": checked " << count
              << " values, errors: " << errors << "\n";
    return errors == 0;
}

static bool check_sqrt(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) {
        std::cerr << "Cannot open " << filename << "\n";
        return false;
    }
    double arg, result;
    int count = 0, errors = 0;
    while (in >> arg >> result) {
        double expected = std::sqrt(arg);
        if (std::abs(result - expected) > EPS) {
            std::cerr << "sqrt(" << arg << ") = " << result
                      << ", expected " << expected << "\n";
            ++errors;
        }
        ++count;
    }
    std::cout << filename << ": checked " << count
              << " values, errors: " << errors << "\n";
    return errors == 0;
}

static bool check_pow(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) {
        std::cerr << "Cannot open " << filename << "\n";
        return false;
    }
    double x, y, result;
    int count = 0, errors = 0;
    while (in >> x >> y >> result) {
        double expected = std::pow(x, y);
        if (std::abs(result - expected) > EPS * std::abs(expected) + EPS) {
            std::cerr << "pow(" << x << ", " << y << ") = " << result
                      << ", expected " << expected << "\n";
            ++errors;
        }
        ++count;
    }
    std::cout << filename << ": checked " << count
              << " values, errors: " << errors << "\n";
    return errors == 0;
}

int main() {
    bool ok = true;
    ok &= check_sin("sin_results.txt");
    ok &= check_sqrt("sqrt_results.txt");
    ok &= check_pow("pow_results.txt");
    if (ok)
        std::cout << "All checks passed!\n";
    else
        std::cout << "Some checks FAILED!\n";
    return ok ? 0 : 1;
}
