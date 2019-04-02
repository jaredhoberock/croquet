#pragma once

#include <utility>
#include <type_traits>
#include <cassert>
#include "detail/type_list.hpp"
#include "sender_traits.hpp"

namespace detail
{

template<class Function, class Receiver>
__global__ void cuda_task_kernel(Function f, Receiver r)
{
  r.set_value(f());
}

} // end detail


template<class Function>
class cuda_task
{
  public:
    using sender_concept = sender_tag;
    using value_types = detail::type_list<std::result_of_t<Function()>>;
    using error_type = void;

    template<class OtherFunction,
             class = std::enable_if_t<
               std::is_constructible<Function,OtherFunction&&>::value
             >>
    __host__ __device__
    cuda_task(OtherFunction&& function)
      : function_(std::forward<OtherFunction>(function))
    {}

    template<class Receiver>
    __host__ __device__
    void submit(Receiver r) &&
    {
      auto* kernel_ptr = &detail::cuda_task_kernel<Function,Receiver>;
      silence_unused_variable_warning(kernel_ptr);

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 350)
      kernel_ptr<<<1,1>>>(function_, r);
#else
      printf("cuda_task::submit: Unimplemented.\n");
      assert(0);
#endif
    }

    __host__ __device__
    Function function() const
    {
      return function_;
    }
    
  private:
    template<class T>
    __host__ __device__
    inline void silence_unused_variable_warning(T&&) {}

    Function function_;
};

