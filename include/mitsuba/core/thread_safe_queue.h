#pragma once

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>

namespace mitsuba {

template <typename T> class ThreadSafeQueue {
public:
    ThreadSafeQueue() = default;

    ThreadSafeQueue(ThreadSafeQueue const &)            = delete;
    ThreadSafeQueue &operator=(ThreadSafeQueue const &) = delete;

    ThreadSafeQueue(ThreadSafeQueue &&)            = default;
    ThreadSafeQueue &operator=(ThreadSafeQueue &&) = default;

    void push(T &&value) {
        {
            std::scoped_lock lock(mutex_);
            queue_.push(std::move(value));
        }
        cond_.notify_one();
    }

    std::optional<T> wait_and_pop() {
        std::unique_lock lock(mutex_);
        cond_.wait(lock, [this] { return !queue_.empty() || done_; });
        if (queue_.empty()) {
            return {};
        }

        auto front = std::move(queue_.front());
        queue_.pop();

        return front;
    }

    void done() {
        done_ = true;
        cond_.notify_all();
    }

private:
    mutable std::mutex mutex_;
    std::queue<T> queue_;
    std::condition_variable cond_;
    std::atomic<bool> done_{ false };
};

} // namespace mitsuba