#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <thread>
#include <tuple>
#include <vector>

#include "server.h"

static constexpr int N = 1000; // задач на клиента (5 < N < 10000)

template<typename T> T fun_sin(T arg)       { return std::sin(arg); }
template<typename T> T fun_sqrt(T arg)      { return std::sqrt(arg); }
template<typename T> T fun_pow(T x, T y)   { return std::pow(x, y); }

// Клиент 1: вычисление синуса
void client_sin(Server<double>& server) {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    std::vector<std::pair<double, size_t>> tasks;
    tasks.reserve(N);
    for (int i = 0; i < N; ++i) {
        double arg = dist(rng);
        size_t id  = server.add_task([arg] { return fun_sin(arg); });
        tasks.push_back({arg, id});
    }

    std::ofstream out("results_sin.txt");
    out << std::fixed << std::setprecision(15);
    for (auto& [arg, id] : tasks) {
        double result = server.request_result(id);
        out << "arg=" << arg << " result=" << result << "\n";
    }
    std::cout << "Client sin done, wrote " << N << " results\n";
}

// Клиент 2: вычисление квадратного корня
void client_sqrt(Server<double>& server) {
    std::mt19937_64 rng(123);
    std::uniform_real_distribution<double> dist(0.0, 1000.0);

    std::vector<std::pair<double, size_t>> tasks;
    tasks.reserve(N);
    for (int i = 0; i < N; ++i) {
        double arg = dist(rng);
        size_t id  = server.add_task([arg] { return fun_sqrt(arg); });
        tasks.push_back({arg, id});
    }

    std::ofstream out("results_sqrt.txt");
    out << std::fixed << std::setprecision(15);
    for (auto& [arg, id] : tasks) {
        double result = server.request_result(id);
        out << "arg=" << arg << " result=" << result << "\n";
    }
    std::cout << "Client sqrt done, wrote " << N << " results\n";
}

// Клиент 3: возведение в степень
void client_pow(Server<double>& server) {
    std::mt19937_64 rng(777);
    std::uniform_real_distribution<double> base_dist(0.1, 10.0);
    std::uniform_real_distribution<double> exp_dist(-3.0, 3.0);

    std::vector<std::tuple<double, double, size_t>> tasks;
    tasks.reserve(N);
    for (int i = 0; i < N; ++i) {
        double x  = base_dist(rng);
        double y  = exp_dist(rng);
        size_t id = server.add_task([x, y] { return fun_pow(x, y); });
        tasks.emplace_back(x, y, id);
    }

    std::ofstream out("results_pow.txt");
    out << std::fixed << std::setprecision(15);
    for (auto& [x, y, id] : tasks) {
        double result = server.request_result(id);
        out << "base=" << x << " exp=" << y << " result=" << result << "\n";
    }
    std::cout << "Client pow done, wrote " << N << " results\n";
}

int main() {
    Server<double> server;
    server.start();

    std::jthread t1([&server] { client_sin(server); });
    std::jthread t2([&server] { client_sqrt(server); });
    std::jthread t3([&server] { client_pow(server); });

    // jthread автоматически joinится при выходе из scope
    t1.join();
    t2.join();
    t3.join();

    server.stop();

    std::cout << "Done. Results saved to results_sin.txt, results_sqrt.txt, results_pow.txt\n";
    return 0;
}