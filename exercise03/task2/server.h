#pragma once
#include <atomic>
#include <condition_variable>
#include <future>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <tbb/concurrent_queue.h>

template<typename T>
class Server {
    using Task = std::pair<size_t, std::packaged_task<T()>>;  // <id задачи, ф-ция для выполнения>

public:
    void start() {
        running_ = true;
        worker_ = std::thread([this]{ server_loop(); });   // запуск нового потока, который вызывает server_loop()
    }

    void stop() {
        running_ = false;
        cv_.notify_all();  // будим немедленно
        if (worker_.joinable())
            worker_.join();
    }

    // Добавляет задачу в очередь, возвращает id
    size_t add_task(std::packaged_task<T()> task) {
        size_t id = next_id_++;
        {
            std::lock_guard lk(futures_mutex_);
            futures_.emplace(id, task.get_future());  // получаем обещание рез-та от задачи
        }
        queue_.push({id, std::move(task)});           // кладём задачу в очередь
        cv_.notify_one();                             // будим воркера, если он спит (очередь была пуста)
        return id;
    }

    // Блокирующий запрос результата по id
    T request_result(size_t id) {
        std::unique_lock lk(futures_mutex_);
        auto it = futures_.find(id);
        auto fut = std::move(it->second);
        futures_.erase(it);
        lk.unlock();      // вручную отпускаем мьютекс, чтобы избежать дедлок
        return fut.get(); // блокирует до завершения задачи
    }

    ~Server() {
        if (running_)
            stop();
    }

private:
    void server_loop() {
        while (running_) {
            Task task;
            if (queue_.try_pop(task)) {  // пытаемся вытащить задачу из очереди
                task.second();           // выполняем задачу, устанавливаем значение future
            } else {
                std::unique_lock lk(cv_mutex_);
                cv_.wait_for(lk, std::chrono::milliseconds(5),
                    [this]{ return !queue_.empty() || !running_; });
            }
        }
        // дочищаем оставшиеся задачи
        Task task;
        while (queue_.try_pop(task))
            task.second();
    }

    std::atomic<size_t> next_id_{1};                      // счётчик для выдачи уникальных id
    tbb::concurrent_queue<Task> queue_;                   // очередь задач

    std::unordered_map<size_t, std::future<T>> futures_;  // словарь id -> future
    std::mutex futures_mutex_;                            // мьютекс для защиты словаря

    std::mutex cv_mutex_;
    std::condition_variable cv_;                          // сон/пробуждение воркера

    std::atomic<bool> running_{false};                    // флаг работы сервера
    std::thread worker_;                                  // поток-воркер
};