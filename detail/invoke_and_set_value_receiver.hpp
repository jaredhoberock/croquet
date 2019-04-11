#pragma once

#include <type_traits>
#include <utility>
#include <agency/detail/type_traits.hpp>
#include <agency/detail/requires.hpp>

namespace detail
{


template<class Function, class Receiver>
struct invoke_and_set_value_receiver
{
  template<class... Args,
           __AGENCY_REQUIRES(
             !std::is_void<
               agency::detail::result_of_t<Function(Args&&...)>
             >::value
           )>
  __host__ __device__
  void set_value(Args&&... args)
  {
    receiver.set_value(function(std::forward<Args>(args)...));
  }

  template<class... Args,
           __AGENCY_REQUIRES(
             std::is_void<
               agency::detail::result_of_t<Function(Args&&...)>
             >::value
           )>
  __host__ __device__
  void set_value(Args&&... args)
  {
    function(std::forward<Args>(args)...);
    receiver.set_value();
  }

  Function function;
  Receiver receiver;
};


} // end detail

