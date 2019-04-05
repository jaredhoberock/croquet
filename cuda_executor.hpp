#pragma once

#include "just.hpp"
#include "single_sender.hpp"

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


template<class Function>
__global__ void single_kernel(Function f)
{
  f();
}


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

    template<class G, class F, class OtherExecutor>
    __host__ __device__
    single_sender<detail::compose<F, G>, cuda_executor> make_value_task(single_sender<G,OtherExecutor> predecessor, F f)
    {
      // XXX what happens to the OtherExecutor?
      detail::compose<F,G> g{f, predecessor.function()};
      return {g, *this};
    }

    template<class T, class F>
    __host__ __device__
    single_sender<detail::bind<F,T>, cuda_executor> make_value_task(just<T> predecessor, F f)
    {
      detail::bind<F,T> g{f, std::move(predecessor).value()};
      return {g, *this};
    }

    template<class Function>
    __host__ __device__
    void execute(Function f) const
    {
      auto* kernel_ptr = &detail::single_kernel<Function>;
      silence_unused_variable_warning(kernel_ptr);

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 350)
      kernel_ptr<<<1,1>>>(f);
#else
      printf("cuda_executor::execute: Unimplemented.\n");
      assert(0);
#endif
    }

  private:
    template<class T>
    __host__ __device__
    inline static void silence_unused_variable_warning(T&&) {}
};

