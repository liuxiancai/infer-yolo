#ifndef __CPM_HPP__
#define __CPM_HPP__

// Comsumer Producer Model  主要是实现了一种多线程任务处理的框架
// 其中包含了一个生产者线程和一个消费者线程，生产者线程负责将任务放入任务队列，消费者线程负责从任务队列中取出任务并执行

#include <algorithm>
#include <condition_variable>
#include <future>
#include <memory>
#include <queue>
#include <thread>

namespace cpm {

template <typename Result, typename Input, typename Model>  //定义了一个模板类 Instance 分别表示 结果 输入 模型
class Instance {
 protected:
  struct Item {
    Input input;
    std::shared_ptr<std::promise<Result>> pro;
  };  //用于存储任务输入和一个promise对象，以便在线程间传递任务结果

  std::condition_variable cond_; //条件变量，用于线程同步
  std::queue<Item> input_queue_;//任务队列，存储待处理的任务
  std::mutex queue_lock_;//互斥锁，保护任务队列的访问
  std::shared_ptr<std::thread> worker_;//工作线程，用于处理任务
  volatile bool run_ = false;//表示工作线程是否应该继续运行
  volatile int max_items_processed_ = 0;//一次处理的最大任务数
  void *stream_ = nullptr;// 额外的数据流，如CUDA流

 public:
  virtual ~Instance() { stop(); } //虚析构函数，确保在对象销毁时停止工作线程并清理资源

  void stop() { // 停止工作线程并清理任务队列
    run_ = false;
    cond_.notify_one();
    {
      std::unique_lock<std::mutex> l(queue_lock_);
      while (!input_queue_.empty()) {
        auto &item = input_queue_.front();
        if (item.pro) item.pro->set_value(Result());
        input_queue_.pop();
      }
    };

    if (worker_) {
      worker_->join();
      worker_.reset();
    }
  }

  virtual std::shared_future<Result> commit(const Input &input) {
    Item item;
    item.input = input;
    item.pro.reset(new std::promise<Result>());
    {
      std::unique_lock<std::mutex> __lock_(queue_lock_);
      input_queue_.push(item);   //创建一个独占锁 __lock_，将item添加到输入队列“input_queue_”中
    }
    cond_.notify_one();
    return item.pro->get_future();
  }

  virtual std::vector<std::shared_future<Result>> commits(const std::vector<Input> &inputs) {
    std::vector<std::shared_future<Result>> output;
    {
      std::unique_lock<std::mutex> __lock_(queue_lock_);
      for (int i = 0; i < (int)inputs.size(); ++i) {
        Item item;
        item.input = inputs[i];
        item.pro.reset(new std::promise<Result>());
        output.emplace_back(item.pro->get_future());
        input_queue_.push(item);
      }
    }
    cond_.notify_one();
    return output;
  }

  template <typename LoadMethod>
  bool start(const LoadMethod &loadmethod, int max_items_processed = 1, void *stream = nullptr) {
    stop();

    this->stream_ = stream;
    this->max_items_processed_ = max_items_processed;
    std::promise<bool> status;
    worker_ = std::make_shared<std::thread>(&Instance::worker<LoadMethod>, this,
                                            std::ref(loadmethod), std::ref(status));
    return status.get_future().get();
  }

 private:
  template <typename LoadMethod>
  void worker(const LoadMethod &loadmethod, std::promise<bool> &status) {  //在独立线程中循环处理输入任务，将任务提交给模型进行处理，并将结果返回给相应的 promise 对象。
                                                                               // 通过使用 promise 和 future，主线程和工作线程可以进行同步和通信，确保线程安全和高效运行。
    std::shared_ptr<Model> model = loadmethod();
    if (model == nullptr) {
      status.set_value(false);
      return;
    }

    run_ = true;
    status.set_value(true);

    std::vector<Item> fetch_items;
    std::vector<Input> inputs;
    while (get_items_and_wait(fetch_items, max_items_processed_)) {
      inputs.resize(fetch_items.size());
      std::transform(fetch_items.begin(), fetch_items.end(), inputs.begin(),
                     [](Item &item) { return item.input; });

      auto ret = model->forwards(inputs, stream_);
      for (int i = 0; i < (int)fetch_items.size(); ++i) {
        if (i < (int)ret.size()) {
          fetch_items[i].pro->set_value(ret[i]);
        } else {
          fetch_items[i].pro->set_value(Result());
        }
      }
      inputs.clear();
      fetch_items.clear();
    }
    model.reset();
    run_ = false;
  }

  virtual bool get_items_and_wait(std::vector<Item> &fetch_items, int max_size) {
    std::unique_lock<std::mutex> l(queue_lock_);
    cond_.wait(l, [&]() { return !run_ || !input_queue_.empty(); });

    if (!run_) return false;

    fetch_items.clear();
    for (int i = 0; i < max_size && !input_queue_.empty(); ++i) {
      fetch_items.emplace_back(std::move(input_queue_.front()));
      input_queue_.pop();
    }
    return true;
  }

  virtual bool get_item_and_wait(Item &fetch_item) {
    std::unique_lock<std::mutex> l(queue_lock_);
    cond_.wait(l, [&]() { return !run_ || !input_queue_.empty(); });

    if (!run_) return false;

    fetch_item = std::move(input_queue_.front());
    input_queue_.pop();
    return true;
  }
};
};  // namespace cpm

#endif  // __CPM_HPP__