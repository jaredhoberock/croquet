#pragma once

#include <utility>
#include <type_traits>
#include <cassert>
#include "detail/invoke_and_set_value_receiver.hpp"
#include "make_promise.hpp"
#include "submit.hpp"
#include "traits.hpp"


// this sender does eager enqueue upon submit before the Predecessor has completed
// XXX maybe name this eager_chained_sender and we could also implement lazy_chained_sender
template<class Executor, class Function, class Predecessor>
class chained_sender
{
  private:
    using predecessor_result_type = sender_value_type_t<Predecessor>;

  public:
    using sender_concept = sender_tag;

    template<template<class...> class Variant, template<class...> class Tuple>
    using value_types = Variant<Tuple<
      std::result_of_t<
        Function(predecessor_result_type)
      >
    >>;

    using error_type = void;

    __host__ __device__
    chained_sender(const Executor& executor, Function function, Predecessor predecessor)
      : executor_(executor),
        function_(function),
        predecessor_(std::move(predecessor))
    {}

    template<class Receiver>
    void submit(Receiver r) const
    {
      // create a promise/future pair
      auto promise = op::make_promise<predecessor_result_type>(executor_);
      auto successor = promise.get_future();

      // submit the predecessor
      op::submit(predecessor_, std::move(promise));

      // fuse the function and receiver
      detail::invoke_and_set_value_receiver<Function,Receiver> fused_receiver{function_, std::move(r)};

      // submit the successor
      op::submit(std::move(successor), std::move(fused_receiver));
    }

    // XXX eliminate this
    __host__ __device__
    Function function() const
    {
      return function_;
    }
    
  private:
    Executor executor_;
    Function function_;
    Predecessor predecessor_;
};


template<class Executor, class Function, class Predecessor>
__host__ __device__
chained_sender<Executor, Function, Predecessor>
  make_chained_sender(const Executor& executor, Function f, Predecessor predecessor)
{
  return {executor, f, std::move(predecessor)};
}

