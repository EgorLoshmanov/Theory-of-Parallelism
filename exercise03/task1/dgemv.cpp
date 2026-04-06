#include <iostream>
#include <vector>
#include <thread>
#include <cstdint>
#include <cstdlib>
#include <chrono>

#define N_RUNS 100

static double wtime() {
    using clock = std::chrono::steady_clock;
    static const auto start = clock::now();
    return std::chrono::duration<double>(clock::now() - start).count();
}

static std::pair<int, int> split_range(int total, int num_threads, int tid) {
    int chunk = total / num_threads;
    int rem   = total % num_threads;
    int begin = tid * chunk + std::min(tid, rem);
    int end   = begin + chunk + (tid < rem ? 1 : 0);
    return {begin, end};
}

static void matrix_vector_product_serial(const double* a, const double* b, double* c, int m, int n) {
    for (int i = 0; i < m; ++i) {
        double sum = 0.0;
        for (int j = 0; j < n; ++j)
            sum += a[(size_t)i * n + j] * b[j];
        c[i] = sum;
    }
}

static void matrix_vector_product_threads(const double* a, const double* b, double* c,
                                          int m, int n, int num_threads) {
    std::vector<std::jthread> threads;
    threads.reserve(num_threads);
    for (int tid = 0; tid < num_threads; ++tid) {
        threads.emplace_back([=] {
            auto [begin, end] = split_range(m, num_threads, tid);
            for (int i = begin; i < end; ++i) {
                double sum = 0.0;
                for (int j = 0; j < n; ++j)
                    sum += a[(size_t)i * n + j] * b[j];
                c[i] = sum;
            }
        });
    }
    // jthread присоединяется автоматически при выходе из scope
}

int main(int argc, char** argv) {
    int m = 20000;
    int n = 20000;
    int init_threads = (int)std::thread::hardware_concurrency();

    if (argc >= 3) {
        m = std::atoi(argv[1]);
        n = std::atoi(argv[2]);
        if (m <= 0 || n <= 0) {
            std::cerr << "Error: m and n must be positive integers\n";
            return 1;
        }
    }
    if (argc >= 4) {
        init_threads = std::atoi(argv[3]);
        if (init_threads <= 0) {
            std::cerr << "Error: init_threads must be a positive integer\n";
            return 1;
        }
    }

    std::cout << "Matrix-vector product (c[m] = a[m,n] * b[n]; m=" << m << ", n=" << n << ")\n";
    std::uint64_t mem_bytes = ((std::uint64_t)m * n + n + 2 * m) * sizeof(double);
    std::cout << "Memory used: " << (mem_bytes >> 20) << " MiB\n";
    std::cout << "Init threads: " << init_threads << "\n";
    std::cout << "Runs: " << N_RUNS << "\n";

    std::vector<double> a((size_t)m * n);
    std::vector<double> b((size_t)n);
    std::vector<double> c((size_t)m);

    // Параллельная инициализация — фиксированное число потоков для всех запусков
    // Раскладка страниц памяти (NUMA first-touch) одинакова для T1 и всех Tp
    {
        std::vector<std::jthread> threads;
        threads.reserve(init_threads);
        for (int tid = 0; tid < init_threads; ++tid) {
            threads.emplace_back([&, tid] {
                auto [begin, end] = split_range(m, init_threads, tid);
                for (int i = begin; i < end; ++i)
                    for (int j = 0; j < n; ++j)
                        a[(size_t)i * n + j] = (double)(i + j);
            });
        }
    }
    {
        std::vector<std::jthread> threads;
        threads.reserve(init_threads);
        for (int tid = 0; tid < init_threads; ++tid) {
            threads.emplace_back([&, tid] {
                auto [begin, end] = split_range(n, init_threads, tid);
                for (int j = begin; j < end; ++j)
                    b[j] = (double)j;
            });
        }
    }

    // T1 измеряется один раз — данные уже инициализированы, раскладка фиксирована
    std::cout << "\nMeasuring serial time (T1)...\n";
    double total_t1 = 0.0;
    for (int k = 0; k < N_RUNS; ++k) {
        double t = wtime();
        matrix_vector_product_serial(a.data(), b.data(), c.data(), m, n);
        total_t1 += wtime() - t;
    }
    double avg_t1 = total_t1 / N_RUNS;
    std::cout << "T1 = " << avg_t1 << " sec\n";

    // Параллельные запуски для разных чисел потоков — T1 один и тот же
    std::vector<int> thread_counts = {1, 2, 4, 7, 8, 16, 20, 40};

    std::cout << "\n";
    std::cout << "Threads |     Tp (sec)  |  Speedup S\n";
    std::cout << "--------|---------------|----------\n";

    for (int p : thread_counts) {
        double total_tp = 0.0;
        for (int k = 0; k < N_RUNS; ++k) {
            double t = wtime();
            matrix_vector_product_threads(a.data(), b.data(), c.data(), m, n, p);
            total_tp += wtime() - t;
        }
        double avg_tp = total_tp / N_RUNS;
        std::cout << "  " << p
                  << "\t| " << avg_tp
                  << "\t| " << avg_t1 / avg_tp << "\n";
    }

    return 0;
}