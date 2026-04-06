#pragma once

#include <condition_variable>
#include <cstddef>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <utility>

// Сервер обрабатывает задачи в отдельном потоке и хранит результаты в unordered_map
// (O(1) вставка, удаление, поиск по id)
template<typename T>
class Server {
public:
    Server() = default;

    ~Server() { stop(); }

    void start() {
        worker_ = std::jthread([this](std::stop_token st) { run(st); });
    }

    void stop() {
        if (worker_.joinable()) {
            worker_.request_stop();
            task_cv_.notify_all();
            worker_.join();
        }
    }

    // Добавляет задачу в очередь, возвращает её id
    size_t add_task(std::function<T()> func) {
        size_t id;
        {
            std::lock_guard lock(task_mutex_);
            id = next_id_++;
            task_queue_.push({id, std::move(func)});
        }
        task_cv_.notify_one();
        return id;
    }

    // Блокирующее получение результата по id
    T request_result(size_t id) {
        std::unique_lock lock(result_mutex_);
        result_cv_.wait(lock, [this, id] { return results_.count(id) > 0; });
        T val = std::move(results_[id]);
        results_.erase(id);
        return val;
    }

private:
    void run(std::stop_token st) {
        while (true) {
            std::pair<size_t, std::function<T()>> item;
            {
                std::unique_lock lock(task_mutex_);
                task_cv_.wait(lock, [this, &st] {
                    return !task_queue_.empty() || st.stop_requested();
                });
                if (task_queue_.empty()) break; // stop запрошен и очередь пуста
                item = std::move(task_queue_.front());
                task_queue_.pop();
            }
            T result = item.second(); // выполняем задачу вне мьютекса
            {
                std::lock_guard lock(result_mutex_);
                results_[item.first] = std::move(result);
            }
            result_cv_.notify_all();
        }
    }

    std::queue<std::pair<size_t, std::function<T()>>> task_queue_;
    std::unordered_map<size_t, T> results_;

    std::mutex task_mutex_;
    std::mutex result_mutex_;
    std::condition_variable task_cv_;
    std::condition_variable result_cv_;

    std::jthread worker_;
    size_t next_id_ = 0;
};