#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <algorithm>

#define N_RUNS 100

static double wtime() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

// Splits [0, total) into nthreads chunks and runs func(begin, end) per thread.
template<typename Func>
static void parallel_for(int total, int nthreads, Func func) {
    std::vector<std::jthread> threads;
    threads.reserve(nthreads);
    int chunk = (total + nthreads - 1) / nthreads;
    for (int t = 0; t < nthreads; ++t) {
        int begin = t * chunk;
        int end   = std::min(begin + chunk, total);
        if (begin >= end) break;
        threads.emplace_back([=]() { func(begin, end); });
    }
}

static void matrix_vector_product_serial(const double* a, const double* b, double* c, int m, int n) {
    for (int i = 0; i < m; ++i) {
        double sum = 0.0;
        for (int j = 0; j < n; ++j)
            sum += a[(size_t)i * (size_t)n + (size_t)j] * b[(size_t)j];
        c[(size_t)i] = sum;
    }
}

static void matrix_vector_product_threads(const double* a, const double* b, double* c,
                                          int m, int n, int nthreads) {
    parallel_for(m, nthreads, [=](int begin, int end) {
        for (int i = begin; i < end; ++i) {
            double sum = 0.0;
            for (int j = 0; j < n; ++j)
                sum += a[(size_t)i * (size_t)n + (size_t)j] * b[(size_t)j];
            c[(size_t)i] = sum;
        }
    });
}

static void run_benchmark(int m, int n, int nthreads) {
    std::vector<double> a((size_t)m * (size_t)n);
    std::vector<double> b((size_t)n);
    std::vector<double> c_serial((size_t)m);
    std::vector<double> c_parallel((size_t)m);

    // Parallel initialization of A
    parallel_for(m, nthreads, [&](int begin, int end) {
        for (int i = begin; i < end; ++i)
            for (int j = 0; j < n; ++j)
                a[(size_t)i * (size_t)n + (size_t)j] = (double)(i + j);
    });

    // Parallel initialization of b
    parallel_for(n, nthreads, [&](int begin, int end) {
        for (int j = begin; j < end; ++j)
            b[(size_t)j] = (double)j;
    });

    double total_t1 = 0.0, total_tp = 0.0;

    for (int k = 0; k < N_RUNS; ++k) {
        double t = wtime();
        matrix_vector_product_serial(a.data(), b.data(), c_serial.data(), m, n);
        total_t1 += wtime() - t;
    }

    for (int k = 0; k < N_RUNS; ++k) {
        double t = wtime();
        matrix_vector_product_threads(a.data(), b.data(), c_parallel.data(), m, n, nthreads);
        total_tp += wtime() - t;
    }

    double avg_t1 = total_t1 / N_RUNS;
    double avg_tp = total_tp / N_RUNS;

    std::cout << "Runs: " << N_RUNS << "\n";
    std::cout << "Average serial time (T1):   " << avg_t1 << " sec\n";
    std::cout << "Average parallel time (Tp): " << avg_tp << " sec\n";
    std::cout << "Speedup S = T1/Tp:          " << avg_t1 / avg_tp << "\n";
    std::cout << "Threads used (p):           " << nthreads << "\n";
}

int main(int argc, char** argv) {
    int m = 20000;
    int n = 20000;
    int nthreads = (int)std::thread::hardware_concurrency();

    if (argc >= 3) {
        m = std::atoi(argv[1]);
        n = std::atoi(argv[2]);
        if (m <= 0 || n <= 0) {
            std::cerr << "Error: m and n must be positive integers\n";
            return 1;
        }
    }
    if (argc >= 4) {
        nthreads = std::atoi(argv[3]);
        if (nthreads <= 0) nthreads = (int)std::thread::hardware_concurrency();
    }

    std::cout << "Matrix-vector product (c[m] = a[m,n] * b[n]; m=" << m << ", n=" << n << ")\n";
    std::uint64_t mem_bytes = ((std::uint64_t)m * n + n + 2 * m) * sizeof(double);
    std::cout << "Memory used: " << (mem_bytes >> 20) << " MiB\n";
    run_benchmark(m, n, nthreads);

    return 0;
}