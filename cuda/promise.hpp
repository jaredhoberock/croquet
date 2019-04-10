#pragma once

#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <memory>
#include <utility>
#include "future.hpp"


namespace cuda
{
namespace detail
{


void promise_host_callback(void* user_data)
{
  std::condition_variable& cv = **reinterpret_cast<std::unique_ptr<std::condition_variable>*>(user_data);
  
  std::mutex mtx;
  std::unique_lock<std::mutex> lock(mtx);
  // XXX need to guard against spurious wakeup
  cv.wait(lock);
}


} // end detail


template<class T>
class promise
{
  public:
    promise()
      : synchronization_channel_(std::make_unique<synchronization_channel>()),
        result_ptr_(nullptr),
        future_(launch_host_function(result_ptr_))
    {}

    promise(promise&& other)
      : synchronization_channel_(std::move(other.synchronization_channel_)),
        result_ptr_(nullptr),
        future_(std::move(other.future_))
    {
      std::swap(result_ptr_, other.result_ptr_);
    }

    future<T> get_future()
    {
      return std::move(future_);
    }

    void set_value(const T& value)
    {
      // construct the result
      allocator_type alloc;
      agency::detail::allocator_traits<allocator_type>::construct(alloc, result_ptr_, value);

      // clear the flag
      synchronization_channel_->flag.clear();

      // wake the host thread
      synchronization_channel_->cv.notify_all();
    }

  private:
    using allocator_type = typename agency::cuda::detail::asynchronous_state<T>::allocator_type;
    using pointer = typename std::allocator_traits<allocator_type>::pointer;

    static void callback(void* user_data)
    {
      synchronization_channel& channel = **reinterpret_cast<std::unique_ptr<synchronization_channel>*>(user_data);
      
      std::mutex mtx;
      std::unique_lock<std::mutex> lock(mtx);
      channel.cv.wait(lock, [&]{ return !channel.flag.test_and_set(); });
    }

    future<T> launch_host_function(pointer& result_ptr)
    {
      // create the asynchronous state to store the result
      agency::cuda::detail::asynchronous_state<T> state(agency::detail::construct_not_ready);

      // get a pointer to the result
      result_ptr = state.data();

      // create a stream on which to launch the callback
      cudaStream_t stream{};
      if(cudaError_t error = cudaStreamCreate(&stream))
      {
        throw std::runtime_error("cuda::promise<T>::launch_host_function: CUDA error after launch_host_function: " + std::string(cudaGetErrorString(error)));
      }

      // launch the callback
      if(cudaError_t error = cudaLaunchHostFunc(stream, callback, &synchronization_channel_))
      {
        throw std::runtime_error("cuda::promise<T>::launch_host_function: CUDA error after cudaLaunchHostFunc: " + std::string(cudaGetErrorString(error)));
      }

      // create an event corresponding to the completion of the callback
      cudaEvent_t event{};
      if(cudaError_t error = cudaEventCreateWithFlags(&event, cudaEventDisableTiming))
      {
        throw std::runtime_error("cuda::promise<T>::launch_host_function: CUDA error after cudaEventCreateWithFlags: " + std::string(cudaGetErrorString(error)));
      }

      // create a future
      future<T> result = agency::cuda::experimental::detail::make_async_future(event, std::move(state));

      // destroy the event
      if(cudaError_t error = cudaEventDestroy(event))
      {
        throw std::runtime_error("cuda::promise<T>::launch_host_function: CUDA error after cudaEventDestroy: " + std::string(cudaGetErrorString(error)));
      }

      // destroy the stream
      if(cudaError_t error = cudaStreamDestroy(stream))
      {
        throw std::runtime_error("cuda::promise<T>::launch_host_function: CUDA error after cudaStreamDestroy: " + std::string(cudaGetErrorString(error)));
      }

      return result;
    }

    struct synchronization_channel
    {
      std::condition_variable cv;
      std::atomic_flag flag = ATOMIC_FLAG_INIT;
    };

    std::unique_ptr<synchronization_channel> synchronization_channel_;
    pointer result_ptr_;
    future<T> future_;
};


// XXX can we avoid a specialization?
template<>
class promise<void>
{
  public:
    promise()
      : cv_(std::make_unique<std::condition_variable>()),
        future_(launch_host_function())
    {}

    future<void> get_future()
    {
      return std::move(future_);
    }

    void set_value()
    {
      cv_->notify_all();
    }

  private:
    future<void> launch_host_function()
    {
      cudaStream_t stream{};
      if(cudaError_t error = cudaStreamCreate(&stream))
      {
        throw std::runtime_error("cuda::promise<void>::launch_host_function: CUDA error after launch_host_function: " + std::string(cudaGetErrorString(error)));
      }

      if(cudaError_t error = cudaLaunchHostFunc(stream, detail::promise_host_callback, &cv_))
      {
        throw std::runtime_error("cuda::promise<void>::launch_host_function: CUDA error after cudaLaunchHostFunc: " + std::string(cudaGetErrorString(error)));
      }

      cudaEvent_t event{};
      if(cudaError_t error = cudaEventCreateWithFlags(&event, cudaEventDisableTiming))
      {
        throw std::runtime_error("cuda::promise<void>::launch_host_function: CUDA error after cudaEventCreateWithFlags: " + std::string(cudaGetErrorString(error)));
      }

      future<void> result = agency::cuda::experimental::make_async_future(event);

      if(cudaError_t error = cudaEventDestroy(event))
      {
        throw std::runtime_error("cuda::promise<void>::launch_host_function: CUDA error after cudaEventDestroy: " + std::string(cudaGetErrorString(error)));
      }

      if(cudaError_t error = cudaStreamDestroy(stream))
      {
        throw std::runtime_error("cuda::promise<void>::launch_host_function: CUDA error after cudaStreamDestroy: " + std::string(cudaGetErrorString(error)));
      }

      return result;
    }

    std::unique_ptr<std::condition_variable> cv_;
    future<void> future_;
};


} // end cuda

