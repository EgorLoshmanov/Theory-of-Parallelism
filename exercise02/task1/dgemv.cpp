#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>  
#include <omp.h>

// seconds timer
static double wtime() {
    return omp_get_wtime();
}

/*
 * matrix_vector_product_omp:
 * c[i] = sum_j a[i*n + j] * b[j]
 */
static void matrix_vector_product_omp(const double* a, const double* b, double* c, int m, int n) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; ++i) {
        double sum = 0.0;
        for (int j = 0; j < n; ++j) {
            sum += a[i * n + j] * b[j];
        }
        c[i] = sum;
    }
}

/*
 * run_parallel:
 * 1) memory allocation
 * 2) PARALLEL initialization of a and c (row-wise)
 * 3) initialization of b
 * 4) parallel matrix-vector multiplication
 */
static void run_parallel(int m, int n) {
    std::vector<double> a(static_cast<size_t>(m) * static_cast<size_t>(n));
    std::vector<double> b(static_cast<size_t>(n));
    std::vector<double> c(static_cast<size_t>(m));

    // Parallel initialization of A and C (distributing matrix rows among threads)
    #pragma omp parallel
    {
        int nth = omp_get_num_threads();
        int tid = omp_get_thread_num();

        int items_per_thread = m / nth;
        int lb = tid * items_per_thread;
        int ub = (tid == nth - 1) ? (m - 1) : (lb + items_per_thread - 1);

        for (int i = lb; i <= ub; ++i) {
            for (int j = 0; j < n; ++j) {
                a[static_cast<size_t>(i) * static_cast<size_t>(n) + static_cast<size_t>(j)] =
                    static_cast<double>(i + j);
            }
            c[static_cast<size_t>(i)] = 0.0;
        }
    }

    #pragma omp parallel for schedule(static)
    for (int j = 0; j < n; ++j) {
        b[static_cast<size_t>(j)] = static_cast<double>(j);
    }

    double t = wtime();
    matrix_vector_product_omp(a.data(), b.data(), c.data(), m, n);
    t = wtime() - t;

    std::cout << "Elapsed time (parallel dgemv): " << t << " sec.\n";
}

int main(int argc, char** argv) {
    int m = 20000;
    int n = 2000;

    if (argc >= 3) {
        m = std::atoi(argv[1]);
        n = std::atoi(argv[2]);
    }

    std::cout << "Matrix-vector product (c[m] = a[m,n] * b[n]; m=" << m << ", n=" << n << ")\n";

    std::uint64_t mem_bytes =
        (static_cast<std::uint64_t>(m) * static_cast<std::uint64_t>(n) +
         static_cast<std::uint64_t>(m) +
         static_cast<std::uint64_t>(n)) * sizeof(double);

    std::cout << "Memory used: " << (mem_bytes >> 20) << " MiB\n";

    run_parallel(m, n);
    return 0;
}