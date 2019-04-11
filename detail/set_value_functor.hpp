#pragma once

namespace detail
{


template<class Function, class Receiver>
struct set_value_functor
{
  mutable Function f;
  mutable Receiver r;

  #pragma nv_exec_check_disable
  __host__ __device__
  void operator()() const
  {
    r.set_value(f());
  }

  // XXX overload operator()() for the f() returns void case
};


} // end detail

