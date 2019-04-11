#pragma once

#include <agency/cuda/execution/executor/grid_executor.hpp>
#include <agency/async.hpp>
#include <agency/execution/executor/detail/execution_functions/then_execute.hpp>
#include <agency/detail/type_traits.hpp>
#include "../just.hpp"
#include "future.hpp"
#include "promise.hpp"


namespace cuda
{
namespace detail
{


template<class Function, class Receiver>
struct receive_invoke_result
{
  template<class... Args,
           __AGENCY_REQUIRES(
             !std::is_void<
               agency::detail::result_of_t<
                 Function(Args&&...)
               >
             >::value
           )>
  __host__ __device__
  void operator()(Args&&... args) const
  {
    receiver_.set_value(function_(std::forward<Args>(args)...));

    // XXX do we need to set_done?
  }

  template<class... Args,
           __AGENCY_REQUIRES(
             std::is_void<
               agency::detail::result_of_t<
                 Function(Args&&...)
               >
             >::value
          )>
  __host__ __device__
  void operator()(Args&&... args) const
  {
    function_(std::forward<Args>(args)...);;
    receiver_.set_value();

    // XXX do we need to set_done?
  }

  mutable Function function_;
  mutable Receiver receiver_;
};


} // end detail


class single_executor
{
  public:
    __host__ __device__
    just<single_executor> schedule() const
    {
      return {*this};
    }

    template<class T>
    cuda::promise<T> make_promise() const
    {
      return cuda::promise<T>();
    }

    template<class Function>
    __host__ __device__
    void execute(Function f) const
    {
      agency::cuda::grid_executor ex;
      agency::async(ex, f);
    }

  private:
    template<class T, class Function>
    struct future_sender
    {
      using sender_concept = sender_tag;

      template<template<class...> class Variant, template<class...> class Tuple>
      using value_types = Variant<Tuple<std::result_of_t<Function(T)>>>;

      using error_type = void;

      template<class Receiver>
      __host__ __device__
      void submit(Receiver r)
      {
        // create a function that calls the continuation and passes its result to the receiver
        detail::receive_invoke_result<Function, Receiver> execute_me{std::move(continuation_), std::move(r)};

        // execute the function after the future becomes ready
        agency::cuda::grid_executor ex;
        agency::detail::then_execute(ex, std::move(execute_me), future_);
      }

      // XXX privatize these
      cuda::future<T> future_;
      Function continuation_;
    };

  public:
    template<class T, class Function>
    __host__ __device__
    future_sender<T,Function> make_value_task(cuda::future<T> predecessor, Function f) const
    {
      return {std::move(predecessor), std::move(f)};
    }
};


} // cuda

