#include <cmath>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <thread>
#include <vector>
#include "server.h"

static constexpr int N = 10000; 

template<typename T>
T fun_sin(T arg) { return std::sin(arg); }

template<typename T>
T fun_sqrt(T arg) { return std::sqrt(arg); }

template<typename T>
T fun_pow(T x, T y) { return std::pow(x, y); }

// Клиент 1: вычисление синуса
void client_sin(Server<double>& server) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-M_PI, M_PI);

    std::vector<std::pair<size_t, double>> tasks; // {id, arg}
    tasks.reserve(N);

    // отправка задач
    for (int i = 0; i < N; ++i) {
        double arg = dist(rng);  // генерим случайное число от -pi до pi
        auto task = std::packaged_task<double()>([arg]{ return fun_sin<double>(arg); });  // создаём задачу, packaged_task позволит забрат рез-тат позже по id
        size_t id = server.add_task(std::move(task));  // кладем в очередь сервера
        tasks.push_back({id, arg});                    // запоминаем {id, arg} для второго цикла
    }

    std::ofstream out("sin_results.txt");
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (auto& [id, arg] : tasks) {
        double result = server.request_result(id);  // ждем пока сервер выполнит эту задачу
        out << arg << " " << result << "\n";
    }
    std::cout << "Client sin: done, results -> sin_results.txt\n";
}

// Клиент 2: вычисление квадратного корня
void client_sqrt(Server<double>& server) {
    std::mt19937 rng(123);
    std::uniform_real_distribution<double> dist(0.0, 1000.0);

    std::vector<std::pair<size_t, double>> tasks;
    tasks.reserve(N);

    for (int i = 0; i < N; ++i) {
        double arg = dist(rng);
        auto task = std::packaged_task<double()>([arg]{ return fun_sqrt<double>(arg); });
        size_t id = server.add_task(std::move(task));
        tasks.push_back({id, arg});
    }

    std::ofstream out("sqrt_results.txt");
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (auto& [id, arg] : tasks) {
        double result = server.request_result(id);
        out << arg << " " << result << "\n";
    }
    std::cout << "Client sqrt: done, results -> sqrt_results.txt\n";
}

// Клиент 3: возведение в степень
void client_pow(Server<double>& server) {
    std::mt19937 rng(999);
    std::uniform_real_distribution<double> base_dist(1.0, 10.0);
    std::uniform_real_distribution<double> exp_dist(0.0, 5.0);

    std::vector<std::tuple<size_t, double, double>> tasks; // {id, x, y}
    tasks.reserve(N);

    for (int i = 0; i < N; ++i) {
        double x = base_dist(rng);
        double y = exp_dist(rng);
        auto task = std::packaged_task<double()>([x, y]{ return fun_pow<double>(x, y); });
        size_t id = server.add_task(std::move(task));
        tasks.emplace_back(id, x, y);
    }

    std::ofstream out("pow_results.txt");
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (auto& [id, x, y] : tasks) {
        double result = server.request_result(id);
        out << x << " " << y << " " << result << "\n";
    }
    std::cout << "Client pow: done, results -> pow_results.txt\n";
}

int main() {
    Server<double> server;
    server.start();

    std::thread t1(client_sin,  std::ref(server));
    std::thread t2(client_sqrt, std::ref(server));
    std::thread t3(client_pow,  std::ref(server));

    t1.join();
    t2.join();
    t3.join();

    server.stop();
    return 0;
}
