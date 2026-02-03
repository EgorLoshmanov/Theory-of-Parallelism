#include <iostream>
#include <vector>
#include <numbers>
#include <cmath>

#ifdef USE_FLOAT
using FloatType = float;
#else
using FloatType = double;
#endif

int main() {
    constexpr size_t N = 10000000;
    std::vector<FloatType> vec(N);
    FloatType sum = 0;

    for (size_t i = 0; i != N; ++i) {
        vec[i] = std::sin(static_cast<FloatType>(2.0) * std::numbers::pi_v<FloatType> * static_cast<FloatType>(i) / static_cast<FloatType>(N));
    }

    for (size_t i = 0; i != N; ++i) {
        sum += vec[i];
    }

    std::cout << sum << std::endl;
    std::cout << "Using " << (sizeof(FloatType) == 4 ? "float" : "double") << "\n";

    return 0;
}