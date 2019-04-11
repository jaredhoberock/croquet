#pragma once

#include <utility>
#include <agency/cuda.hpp>
#include "../traits.hpp"

namespace cuda
{
namespace detail
{


template<class Receiver>
struct set_value_with_argument
{
  template<class Arg>
  __host__ __device__
  void operator()(Arg&& arg)
  {
    receiver.set_value(std::forward<Arg>(arg));
  }

  Receiver receiver;
};


} // end detail


template<class T>
class future
{
  public:
    using sender_concept = sender_tag;

    template<template<class...> class Variant, template<class...> class Tuple>
    using value_types = Variant<Tuple<T>>;

    using error_type = void;

    future() = default;
    future(future&&) = default;

    future(agency::cuda::async_future<T>&& other)
      : impl_(std::move(other))
    {}

    template<class Receiver>
    __host__ __device__
    void submit(Receiver&& r) &&
    {
      impl_.then(detail::set_value_with_argument<std::decay_t<Receiver>>{std::move(r)});
    }

  private:
    agency::cuda::async_future<T> impl_;
};


} // end cuda

