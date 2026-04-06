#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <omp.h>

#define N_RUNS 100

// seconds timer
static double wtime() {
    return omp_get_wtime();
}

// matrix_vector_product_serial
static void matrix_vector_product_serial(const double* a, const double* b, double* c, int m, int n) {
    for (int i = 0; i < m; ++i) {
        double sum = 0.0;
        for (int j = 0; j < n; ++j) {
            sum += a[(size_t)i * (size_t)n + (size_t)j] * b[(size_t)j];
        }
        c[(size_t)i] = sum;
    }
}

// matrix_vector_product_omp
static void matrix_vector_product_omp(const double* a, const double* b, double* c, int m, int n) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; ++i) {
        double sum = 0.0;
        for (int j = 0; j < n; ++j) {
            sum += a[(size_t)i * (size_t)n + (size_t)j] * b[(size_t)j];
        }
        c[(size_t)i] = sum;
    }
}

static void run_benchmark(int m, int n) {
    std::vector<double> a((size_t)m * (size_t)n);
    std::vector<double> b((size_t)n);
    std::vector<double> c_serial((size_t)m);
    std::vector<double> c_parallel((size_t)m);

// parallel initialization of A
#pragma omp parallel for schedule(static)
for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
        a[(size_t)i * (size_t)n + (size_t)j] = (double)(i + j);
    }
}

    // initialization of b
#pragma omp parallel for schedule(static)
    for (int j = 0; j < n; ++j)
        b[(size_t)j] = (double)j;

    double total_t1 = 0.0;
    double total_tp = 0.0;

    // Serial runs
    for (int k = 0; k < N_RUNS; ++k) {
        double t = wtime();
        matrix_vector_product_serial(a.data(), b.data(), c_serial.data(), m, n);
        t = wtime() - t;
        total_t1 += t;
    }

    // Parallel runs
    for (int k = 0; k < N_RUNS; ++k) {
        double t = wtime();
        matrix_vector_product_omp(a.data(), b.data(), c_parallel.data(), m, n);
        t = wtime() - t;
        total_tp += t;
    }

    double avg_t1 = total_t1 / N_RUNS;
    double avg_tp = total_tp / N_RUNS;
    double speedup = avg_t1 / avg_tp;

int p = omp_get_max_threads(); 

    std::cout << "Runs: " << N_RUNS << "\n";
    std::cout << "Average serial time (T1):   " << avg_t1 << " sec\n";
    std::cout << "Average parallel time (Tp): " << avg_tp << " sec\n";
    std::cout << "Speedup S = T1/Tp:          " << speedup << "\n";
    std::cout << "Threads used (p):           " << p << "\n";
}

int main(int argc, char** argv) {

    int m = 20000;
    int n = 20000;

    if (argc >= 3) {
        m = std::atoi(argv[1]);
        n = std::atoi(argv[2]);
        if (m <= 0 || n <= 0) {
            std::cerr << "Error: m and n must be positive integers\n";
            return 1;
        }
    }

    std::cout << "Matrix-vector product (c[m] = a[m,n] * b[n]; m=" << m << ", n=" << n << ")\n";
    std::uint64_t mem_bytes = ((std::uint64_t)m * n + n + 2 * m) * sizeof(double);
    std::cout << "Memory used: " << (mem_bytes >> 20) << " MiB\n";
    run_benchmark(m, n);

    return 0;
}