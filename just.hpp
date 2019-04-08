#pragma once

#include <utility>
#include "detail/type_list.hpp"
#include "traits.hpp"


namespace detail
{


template<class T>
struct return_value
{
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
    using value_types = detail::type_list<T>;
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

    #pragma nv_exec_check_disable
    template<class Receiver>
    __host__ __device__
    void submit(Receiver&& r) &&
    {
      // XXX when CPOs are available:
      // set_value(std::forward<Receiver>(r), std::move(value_));

      std::forward<Receiver>(r).set_value(std::move(value_));
    }

    #pragma nv_exec_check_disable
    template<class Receiver>
    __host__ __device__
    void submit(Receiver&& r) const &
    {
      // XXX when CPOs are available:
      // set_value(std::forward<Receiver>(r), value_);

      std::forward<Receiver>(r).set_value(value_);
    }

    __host__ __device__
    T&& value() &&
    {
      return std::move(value_);
    }

    __host__ __device__
    detail::return_value<T> function() const
    {
      return {value_};
    }

  private:
    T value_;
};

