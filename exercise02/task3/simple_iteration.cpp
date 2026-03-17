#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

#define N_RUNS 50

double norm_serial(const std::vector<double>& v)
{
    double sum = 0.0;
    for (int i = 0; i < (int)v.size(); ++i)
        sum += v[i] * v[i];
    return std::sqrt(sum);
}

double norm_parallel(const std::vector<double>& v)
{
    double sum = 0.0;

#pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < (int)v.size(); ++i)
        sum += v[i] * v[i];

    return std::sqrt(sum);
}

double solve_serial(int N, double tau, double eps)
{
    std::vector<double> b(N, N + 1);
    std::vector<double> x(N, 0.0);
    std::vector<double> x_new(N);
    std::vector<double> r(N);

    double norm_b = norm_serial(b);
    int max_iter = 1000;
    int iter = 0;

    double t1 = omp_get_wtime();

    while (iter < max_iter)
    {
        double sum_x = 0.0;
        for (int i = 0; i < N; ++i)
            sum_x += x[i];

        for (int i = 0; i < N; ++i)
            r[i] = (x[i] + sum_x) - b[i];

        double res = norm_serial(r) / norm_b;
        if (res < eps)
            break;

        for (int i = 0; i < N; ++i)
            x_new[i] = x[i] - tau * r[i];

        x = x_new;
        ++iter;
    }

    double t2 = omp_get_wtime();
    return t2 - t1;
}

double solve_parallel(int N, double tau, double eps)
{
    std::vector<double> b(N, N + 1);
    std::vector<double> x(N, 0.0);
    std::vector<double> x_new(N);
    std::vector<double> r(N);

    double norm_b = norm_parallel(b);
    int max_iter = 1000;
    int iter = 0;

    double t1 = omp_get_wtime();

    while (iter < max_iter)
    {
        double sum_x = 0.0;

#pragma omp parallel for reduction(+:sum_x)
        for (int i = 0; i < N; ++i)
            sum_x += x[i];

#pragma omp parallel for
        for (int i = 0; i < N; ++i)
            r[i] = (x[i] + sum_x) - b[i];

        double res = norm_parallel(r) / norm_b;
        if (res < eps)
            break;

#pragma omp parallel for
        for (int i = 0; i < N; ++i)
            x_new[i] = x[i] - tau * r[i];

        x = x_new;
        ++iter;
    }

    double t2 = omp_get_wtime();
    return t2 - t1;
}

int main()
{
    int N = 1000000;
    double tau = 1.0 / (2.0 * (N + 1));
    double eps = 1e-5;

    double sum_T1 = 0.0;
    double sum_Tp = 0.0;

    for (int i = 0; i < N_RUNS; ++i)
    {
        sum_T1 += solve_serial(N, tau, eps);
        sum_Tp += solve_parallel(N, tau, eps);
    }

    double T1_avg = sum_T1 / N_RUNS;
    double Tp_avg = sum_Tp / N_RUNS;

    std::cout << "Average sequential time T1: " << T1_avg << " sec\n";
    std::cout << "Average parallel time Tp:   " << Tp_avg << " sec\n";
    std::cout << "Speedup S = T1 / Tp = " << T1_avg / Tp_avg << std::endl;

    return 0;
}