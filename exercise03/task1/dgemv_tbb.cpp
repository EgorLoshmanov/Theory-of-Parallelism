#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <memory>
#include <tbb/tbb.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/global_control.h>

#define N_RUNS 100

static double wtime() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

// Serial version 
static void matrix_vector_product_serial(const double* a, const double* b, double* c, int m, int n) {
    for (int i = 0; i < m; ++i) {
        double sum = 0.0;
        for (int j = 0; j < n; ++j) {
            sum += a[(size_t)i * (size_t)n + (size_t)j] * b[(size_t)j];
        }
        c[(size_t)i] = sum;
    }
}

// Parallel version using TBB parallel_for
static void matrix_vector_product_tbb(const double* a, const double* b, double* c, int m, int n) {
    tbb::parallel_for(tbb::blocked_range<int>(0, m),
        [=](const tbb::blocked_range<int>& r) {
            for (int i = r.begin(); i < r.end(); ++i) {
                double sum = 0.0;
                for (int j = 0; j < n; ++j) {
                    sum += a[(size_t)i * (size_t)n + (size_t)j] * b[(size_t)j];
                }
                c[(size_t)i] = sum;
            }
        }
    );
}

static void run_benchmark(int m, int n) {
    std::vector<double> a((size_t)m * (size_t)n);
    std::vector<double> b((size_t)n);
    std::vector<double> c_serial((size_t)m);
    std::vector<double> c_parallel((size_t)m);

    // Parallel initialization of A using TBB
    tbb::parallel_for(tbb::blocked_range<int>(0, m),
        [&](const tbb::blocked_range<int>& r) {
            for (int i = r.begin(); i < r.end(); ++i) {
                for (int j = 0; j < n; ++j) {
                    a[(size_t)i * (size_t)n + (size_t)j] = (double)(i + j);
                }
            }
        }
    );

    // Parallel initialization of b using TBB
    tbb::parallel_for(tbb::blocked_range<int>(0, n),
        [&](const tbb::blocked_range<int>& r) {
            for (int j = r.begin(); j < r.end(); ++j) {
                b[(size_t)j] = (double)j;
            }
        }
    );

    double total_t1 = 0.0;
    double total_tp = 0.0;

    // Serial runs
    for (int k = 0; k < N_RUNS; ++k) {
        double t = wtime();
        matrix_vector_product_serial(a.data(), b.data(), c_serial.data(), m, n);
        total_t1 += wtime() - t;
    }

    // Parallel runs (TBB)
    for (int k = 0; k < N_RUNS; ++k) {
        double t = wtime();
        matrix_vector_product_tbb(a.data(), b.data(), c_parallel.data(), m, n);
        total_tp += wtime() - t;
    }

    double avg_t1 = total_t1 / N_RUNS;
    double avg_tp = total_tp / N_RUNS;
    double speedup = avg_t1 / avg_tp;
    size_t p = tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism);

    std::cout << "Runs: " << N_RUNS << "\n";
    std::cout << "Average serial time (T1):   " << avg_t1 << " sec\n";
    std::cout << "Average parallel time (Tp): " << avg_tp << " sec\n";
    std::cout << "Speedup S = T1/Tp:          " << speedup << "\n";
    std::cout << "Threads used (p):           " << p << "\n";
}

int main(int argc, char** argv) {
    int m = 20000;
    int n = 20000;
    int threads = 0; 

    if (argc >= 3) {
        m = std::atoi(argv[1]);
        n = std::atoi(argv[2]);
        if (m <= 0 || n <= 0) {
            std::cerr << "Error: m and n must be positive integers\n";
            return 1;
        }
    }
    if (argc >= 4) {
        threads = std::atoi(argv[3]);
    }

    std::unique_ptr<tbb::global_control> gc;
    if (threads > 0) {
        gc = std::make_unique<tbb::global_control>(
            tbb::global_control::max_allowed_parallelism, threads);
    }

    std::cout << "Matrix-vector product (c[m] = a[m,n] * b[n]; m=" << m << ", n=" << n << ")\n";
    std::uint64_t mem_bytes = ((std::uint64_t)m * n + n + 2 * m) * sizeof(double);
    std::cout << "Memory used: " << (mem_bytes >> 20) << " MiB\n";
    run_benchmark(m, n);

    return 0;
}