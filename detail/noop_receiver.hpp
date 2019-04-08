#pragma once

namespace detail
{


struct noop_receiver
{
  template<class T>
  __host__ __device__
  void set_value(T&&) const {}
}; // end noop_receiver


} // end detail

