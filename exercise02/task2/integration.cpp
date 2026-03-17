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

    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (int i = 0; i < n; ++i) {
        sum += f(a + h * (i + 0.5));
    }

    return sum * h;
}

constexpr double PI = 3.14159265358979323846;
constexpr double A = -4.0;
constexpr double B = 4.0;
constexpr int NSTEPS = 40000000;
constexpr int N_RUNS = 50;

int main()
{
    std::cout << "Integration f(x) on [" << A << ", " << B << "], nsteps = " << NSTEPS << "\n";

    double serial_time_sum = 0.0;
    double parallel_time_sum = 0.0;

    double res_serial = 0.0;
    double res_parallel = 0.0;

    for (int i = 0; i < N_RUNS; ++i)
    {
        double t = omp_get_wtime();
        res_serial = integrate(func, A, B, NSTEPS);
        serial_time_sum += omp_get_wtime() - t;

        t = omp_get_wtime();
        res_parallel = integrate_omp(func, A, B, NSTEPS);
        parallel_time_sum += omp_get_wtime() - t;
    }

    double tserial = serial_time_sum / N_RUNS;
    double tparallel = parallel_time_sum / N_RUNS;

    std::cout << std::fixed << std::setprecision(12);
    std::cout << "Result (serial):   " << res_serial << "; error " << std::fabs(res_serial - std::sqrt(PI)) << "\n";
    std::cout << "Result (parallel): " << res_parallel << "; error " << std::fabs(res_parallel - std::sqrt(PI)) << "\n";
    std::cout << std::setprecision(6);
    std::cout << "Average time (serial):   " << tserial << "\n";
    std::cout << "Average time (parallel): " << tparallel << "\n";
    std::cout << "Speedup: " << tserial / tparallel << "\n";

    return 0;
}