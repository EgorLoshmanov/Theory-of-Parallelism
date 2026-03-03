#include <iostream>
#include <cmath>
#include <iomanip>
#include <omp.h>

double func(double x)
{
    return std::exp(-x * x);
}

double integrate(double (*f)(double), double a, double b, int n)
{
    double h = (b - a) / n;
    double sum = 0.0;
    for (int i = 0; i < n; ++i)
        sum += f(a + h * (i + 0.5));

    return sum * h;
}

double integrate_omp(double (*f)(double), double a, double b, int n)
{
    double h = (b - a) / n;
    double sum = 0.0;

#pragma omp parallel
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();

        int items_per_thread = n / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (n - 1) : (lb + items_per_thread - 1);

        double sumloc = 0.0;
        for (int i = lb; i <= ub; ++i)
            sumloc += f(a + h * (i + 0.5));

#pragma omp atomic
        sum += sumloc;
    }

    return sum * h;
}

constexpr double PI = 3.14159265358979323846;
constexpr double A = -4.0;
constexpr double B = 4.0;
constexpr int NSTEPS = 40000000;

double run_serial()
{
    double t = omp_get_wtime();
    double res = integrate(func, A, B, NSTEPS);
    t = omp_get_wtime() - t;

    std::cout << std::fixed << std::setprecision(12);
    std::cout << "Result (serial):   " << res << "; error " << std::fabs(res - std::sqrt(PI)) << "\n";

    return t;
}

double run_parallel()
{
    double t = omp_get_wtime();
    double res = integrate_omp(func, A, B, NSTEPS);
    t = omp_get_wtime() - t;

    std::cout << std::fixed << std::setprecision(12);
    std::cout << "Result (parallel): " << res << "; error " << std::fabs(res - std::sqrt(PI)) << "\n";

    return t;
}

int main()
{
    std::cout << "Integration f(x) on [" << A << ", " << B << "], nsteps = " << NSTEPS << "\n";

    double tserial = run_serial();
    double tparallel = run_parallel();
    
    std::cout << std::setprecision(6);
    std::cout << "Execution time (serial):   " << tserial << "\n";
    std::cout << "Execution time (parallel): " << tparallel << "\n";
    std::cout << "Speedup: " << tserial / tparallel << "\n";

    return 0;
}