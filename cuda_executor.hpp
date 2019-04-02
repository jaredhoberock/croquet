#pragma once

#include "just.hpp"
#include "cuda_task.hpp"

namespace detail
{


template<class F, class G>
struct compose
{
  // returns f of g
  #pragma nv_exec_check_disable
  __host__ __device__
  auto operator()()
  {
    return f_(g_());
  }

  F f_;
  G g_;
};


template<class F, class Arg>
struct bind
{
  // returns f(arg)
  #pragma nv_exec_check_disable
  __host__ __device__
  auto operator()()
  {
    return f_(arg_);
  }

  F f_;
  Arg arg_;
};


} // end detail

class cuda_executor
{
  public:
    __host__ __device__
    just<cuda_executor> schedule() const
    {
      return {*this};
    }

    // XXX is there an easy way to have a single, generic
    // make_value_task rather than these two overloads?

    template<class G, class F>
    __host__ __device__
    cuda_task<detail::compose<F, G>> make_value_task(cuda_task<G> predecessor, F f)
    {
      return {f, predecessor.function()};
    }

    template<class T, class F>
    __host__ __device__
    cuda_task<detail::bind<F, T>> make_value_task(just<T> predecessor, F f)
    {
      detail::bind<F,T> g{f, std::move(predecessor).value()};
      return {g};
    }
};

