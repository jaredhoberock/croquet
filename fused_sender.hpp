#pragma once

#include <utility>
#include <type_traits>
#include <cassert>
#include "detail/set_value_functor.hpp"
#include "traits.hpp"


template<class Function, class Executor>
class fused_sender
{
  public:
    using sender_concept = sender_tag;

    template<template<class...> class Variant, template<class...> class Tuple> 
    using value_types = Variant<Tuple<std::result_of_t<Function()>>>;

    using error_type = void;

    __host__ __device__
    fused_sender(Function function, const Executor& executor)
      : function_(function),
        executor_(executor)
    {}

    template<class Receiver>
    __host__ __device__
    auto submit(Receiver r) const
    {
      detail::set_value_functor<Function, Receiver> f{function_, std::move(r)};
      return executor_.execute(std::move(f));
    }

    // XXX eliminate this
    __host__ __device__
    Function function() const
    {
      return function_;
    }

    __host__ __device__
    Executor executor() const
    {
      return executor_;
    }
    
  private:
    Function function_;
    Executor executor_;
};


template<class Function, class Executor>
__host__ __device__
fused_sender<std::decay_t<Function>,Executor>
  make_fused_sender(Function&& f, const Executor& executor)
{
  return {std::forward<Function>(f), executor};
}

