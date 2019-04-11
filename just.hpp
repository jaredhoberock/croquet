#pragma once

#include <utility>
#include "traits.hpp"
#include "inline_executor.hpp"
#include "detail/set_value_functor.hpp"


namespace detail
{


template<class T>
struct return_value
{
  #pragma nv_exec_check_disable
  __host__ __device__
  T operator()() const
  {
    return value;
  }

  T value;
};


} // end detail


template<class T>
class just
{
  public:
    using sender_concept = sender_tag;

    template<template<class...> class Variant, template<class...> class Tuple>
    using value_types = Variant<Tuple<T>>;

    using error_type = void;

    __host__ __device__
    just(just&& other)
      : value_(std::move(other.value_))
    {}

    template<class OtherT, 
             class = std::enable_if_t<
               std::is_constructible<T,OtherT&&>::value
             >>
    __host__ __device__
    just(OtherT&& value)
      : value_(std::forward<OtherT>(value))
    {}

    // XXX shouldn't this be an rvalue member function?
    #pragma nv_exec_check_disable
    template<class Receiver>
    __host__ __device__
    void submit(Receiver&& r) const &
    {
      detail::set_value_functor<detail::return_value<T>, std::decay_t<Receiver>> f{function(), std::forward<Receiver>(r)};

      executor_.execute(std::move(f));
    }

    __host__ __device__
    T&& value() &&
    {
      return std::move(value_);
    }

    // XXX eliminate this
    __host__ __device__
    detail::return_value<T> function() const
    {
      return {value_};
    }

    __host__ __device__
    inline_executor executor() const
    {
      return executor_;
    }

  private:
    T value_;
    inline_executor executor_;
};

