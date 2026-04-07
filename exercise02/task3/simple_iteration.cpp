#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <omp.h>

static constexpr double EPS = 1e-5;
static constexpr int MAX_ITER = 1000000;
static constexpr int N_RUNS = 100; 

static double norm2(const double* v, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) s += v[i] * v[i];
    return std::sqrt(s);
}

// Инициализация матрицы и векторов.
// Параллельная инициализация с фиксированным числом потоков (max доступных)
// обеспечивает одинаковую NUMA-раскладку страниц для всех прогонов.
static void init_data(double* a, double* b, double* x, int n) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
            a[(size_t)i * n + j] = (i == j) ? 2.0 : 1.0;
        b[i] = (double)(n + 1);
        x[i] = 0.0;
    }
}

// Последовательная версия
static int solve_serial(const double* a, const double* b, double* x,
                        int n, double tau, double& elapsed) {
    std::vector<double> r(n);
    double b_norm = norm2(b, n);
    int iter = 0;

    double t0 = omp_get_wtime();
    while (iter < MAX_ITER) {
        for (int i = 0; i < n; ++i) {
            double sum = 0.0;
            for (int j = 0; j < n; ++j)
                sum += a[(size_t)i * n + j] * x[j];
            r[i] = sum - b[i];
        }
        if (norm2(r.data(), n) / b_norm < EPS) break;
        for (int i = 0; i < n; ++i)
            x[i] -= tau * r[i];
        ++iter;
    }
    elapsed = omp_get_wtime() - t0;
    return iter;
}

// Вариант 1: отдельный #pragma omp parallel for на каждый цикл
static int solve_v1(const double* a, const double* b, double* x,
                    int n, double tau, int nthreads, double& elapsed) {
    std::vector<double> r(n);
    double b_norm = norm2(b, n);
    int iter = 0;

    omp_set_num_threads(nthreads);
    double t0 = omp_get_wtime();
    while (iter < MAX_ITER) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            double sum = 0.0;
            for (int j = 0; j < n; ++j)
                sum += a[(size_t)i * n + j] * x[j];
            r[i] = sum - b[i];
        }
        if (norm2(r.data(), n) / b_norm < EPS) break;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i)
            x[i] -= tau * r[i];
        ++iter;
    }
    elapsed = omp_get_wtime() - t0;
    return iter;
}

// Вариант 2: одна параллельная секция #pragma omp parallel 

static int solve_v2(const double* a, const double* b, double* x,
                    int n, double tau, int nthreads, double& elapsed) {
    std::vector<double> r(n);
    double b_norm = norm2(b, n);
    int iter = 0;
    bool converged = false;

    omp_set_num_threads(nthreads);
    double t0 = omp_get_wtime();

    #pragma omp parallel shared(converged, iter, r)
    {
        while (true) {
            // Умножение матрицы на вектор: r = A*x - b
            #pragma omp for schedule(static)
            for (int i = 0; i < n; ++i) {
                double sum = 0.0;
                for (int j = 0; j < n; ++j)
                    sum += a[(size_t)i * n + j] * x[j];
                r[i] = sum - b[i];
            }
            // implicit barrier

            // Проверка сходимости и обновление счётчика (один поток)
            #pragma omp single
            {
                ++iter;
                converged = (norm2(r.data(), n) / b_norm < EPS) || (iter >= MAX_ITER);
            }
            // implicit barrier - все потоки видят обновлённый converged

            if (converged) break;

            // Обновление x: x = x - tau * r
            #pragma omp for schedule(static)
            for (int i = 0; i < n; ++i)
                x[i] -= tau * r[i];
            // implicit barrier
        }
    }

    elapsed = omp_get_wtime() - t0;
    return iter;
}

// Запускает вариант 1 с разными schedule и chunk_size при фиксированных n и nthreads

struct ScheduleResult {
    const char* name;
    int         chunk;
    double      elapsed;
    int         iter;
};

static ScheduleResult bench_schedule(const char* sched_name, int schedule_kind, int chunk,
                                     const double* a, const double* b, double* x,
                                     int n, double tau, int nthreads) {
    // Сбросить x
    for (int i = 0; i < n; ++i) x[i] = 0.0;

    std::vector<double> r(n);
    double b_norm = norm2(b, n);
    int iter = 0;
    double t0 = omp_get_wtime();

    // schedule_kind: 1=static, 2=dynamic, 3=guided
    while (iter < MAX_ITER) {
        if (schedule_kind == 1) {
            #pragma omp parallel for schedule(static) num_threads(nthreads)
            for (int i = 0; i < n; ++i) {
                double sum = 0.0;
                for (int j = 0; j < n; ++j)
                    sum += a[(size_t)i * n + j] * x[j];
                r[i] = sum - b[i];
            }
        } else if (schedule_kind == 2) {
            #pragma omp parallel for schedule(dynamic) num_threads(nthreads)
            for (int i = 0; i < n; ++i) {
                double sum = 0.0;
                for (int j = 0; j < n; ++j)
                    sum += a[(size_t)i * n + j] * x[j];
                r[i] = sum - b[i];
            }
        } else {
            #pragma omp parallel for schedule(guided) num_threads(nthreads)
            for (int i = 0; i < n; ++i) {
                double sum = 0.0;
                for (int j = 0; j < n; ++j)
                    sum += a[(size_t)i * n + j] * x[j];
                r[i] = sum - b[i];
            }
        }
        if (norm2(r.data(), n) / b_norm < EPS) break;
        for (int i = 0; i < n; ++i)
            x[i] -= tau * r[i];
        ++iter;
    }

    return {sched_name, chunk, omp_get_wtime() - t0, iter};
}

int main(int argc, char** argv) {
    int n = 2000; 

    if (argc >= 2) {
        n = std::atoi(argv[1]);
        if (n <= 0) { std::cerr << "Error: N must be positive\n"; return 1; }
    }

    // tau_opt = 2 / (lambda_min + lambda_max) = 2 / (1 + N+1) = 2 / (N+2)
    // lambda_min = 1 (N-1 кратное), lambda_max = N+1 (однократное)
    double tau = 2.0 / (double)(n + 2);

    std::uint64_t mem_bytes = (std::uint64_t)n * n * sizeof(double);

    std::cout << "=== Метод простых итераций (Ax=b) ===\n";
    std::cout << "N = " << n << "  (матрица " << n << "x" << n << ")\n";
    std::cout << "Память: " << (mem_bytes >> 20) << " МиБ\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "tau  = " << tau << "  (оптимальный: 2/(N+2))\n";
    std::cout << "eps  = " << EPS << "\n\n";

    // Выделяем матрицу и векторы
    std::vector<double> a((size_t)n * n);
    std::vector<double> b(n), x(n), x_ref(n);

    // Инициализация (параллельная, фиксированный number of threads = max)
    omp_set_num_threads(omp_get_max_threads());
    init_data(a.data(), b.data(), x.data(), n);

    // Серийный прогон (усреднение по N_RUNS) 
    double t_serial = 0.0;
    int iters_serial = 0;
    for (int run = 0; run < N_RUNS; ++run) {
        x_ref.assign(n, 0.0);
        double t;
        iters_serial = solve_serial(a.data(), b.data(), x_ref.data(), n, tau, t);
        t_serial += t;
    }
    t_serial /= N_RUNS;
    std::cout << "Серийно (avg " << N_RUNS << " runs):  t = "
              << t_serial << " сек,  итераций = " << iters_serial << "\n\n";

    // Вывод таблицы 
    std::vector<int> thread_counts = {1, 2, 4, 8, 16, 20, 40};

    std::cout << std::setw(8)  << "Потоков"
              << std::setw(14) << "V1 t (сек)"
              << std::setw(12) << "V1 Speedup"
              << std::setw(14) << "V2 t (сек)"
              << std::setw(12) << "V2 Speedup"
              << "\n";
    std::cout << std::string(60, '-') << "\n";

    for (int p : thread_counts) {
        double t1 = 0.0, t2 = 0.0;
        int i1 = 0, i2 = 0;

        for (int run = 0; run < N_RUNS; ++run) {
            double t;
            x.assign(n, 0.0);
            i1 = solve_v1(a.data(), b.data(), x.data(), n, tau, p, t);
            t1 += t;
        }
        for (int run = 0; run < N_RUNS; ++run) {
            double t;
            x.assign(n, 0.0);
            i2 = solve_v2(a.data(), b.data(), x.data(), n, tau, p, t);
            t2 += t;
        }
        t1 /= N_RUNS;
        t2 /= N_RUNS;

        std::cout << std::setw(8)  << p
                  << std::setw(14) << t1
                  << std::setw(12) << t_serial / t1
                  << std::setw(14) << t2
                  << std::setw(12) << t_serial / t2
                  << "   (iters: " << i1 << " / " << i2 << ")\n";
    }

    // Анализ schedule 
    int sched_threads = 8; // фиксированное число потоков для анализа schedule
    if (argc >= 3) sched_threads = std::atoi(argv[2]);

    std::cout << "\n=== Анализ schedule (p=" << sched_threads << ", Вариант 1) ===\n";
    std::cout << std::setw(12) << "Schedule"
              << std::setw(14) << "t (сек)"
              << std::setw(12) << "Speedup"
              << "\n";
    std::cout << std::string(38, '-') << "\n";

    struct SchedEntry { const char* name; int kind; int chunk; };
    std::vector<SchedEntry> schedules = {
        {"static",   1, 0},
        {"dynamic",  2, 0},
        {"guided",   3, 0},
    };

    for (auto& s : schedules) {
        double total = 0.0;
        int iters = 0;
        for (int run = 0; run < N_RUNS; ++run) {
            x.assign(n, 0.0);
            auto res = bench_schedule(s.name, s.kind, s.chunk,
                                      a.data(), b.data(), x.data(),
                                      n, tau, sched_threads);
            total += res.elapsed;
            iters  = res.iter;
        }
        double avg = total / N_RUNS;
        std::cout << std::setw(12) << s.name
                  << std::setw(14) << avg
                  << std::setw(12) << t_serial / avg
                  << "\n";
    }

    return 0;
}