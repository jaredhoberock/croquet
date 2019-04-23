#pragma once

#include <utility>

namespace detail
{


template<class Receiver>
struct receive_arguments_functor
{
  template<class... Args>
  __host__ __device__
  void operator()(Args&&... args) const
  {
    receiver_.set_value(std::forward<Args>(args)...);

    // XXX do we need to set_done?
  }

  mutable Receiver receiver_;
};


} // end detail

