#pragma once

#include <utility>
#include <type_traits>
#include <agency/detail/requires.hpp>

namespace detail
{


template<class Function, class Receiver>
struct receive_invoke_result_functor
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

