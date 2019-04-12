// $ nvcc --expt-extended-lambda -std=c++14 -Iagency fusing_executor_demo.cu
#include <iostream>
#include <typeinfo>
#include "fusing_executor.hpp"
#include "cuda/single_executor.hpp"
#include "submit.hpp"
#include "share.hpp"


struct my_receiver
{
  template<class T>
  __host__ __device__
  void set_value(T arg)
  {
#ifndef __CUDA_ARCH__
    std::cout << "my_receiver::set_value: received " << typeid(arg).name() << std::endl;
#else
    printf("Hello world from my_receiver\n");
#endif
  }
};

int main()
{
  fusing_executor<cuda::single_executor> ex = make_fusing_executor(cuda::single_executor());

  just<fusing_executor<cuda::single_executor>> s1 = ex.schedule();

  {
    // test submission on a trivial sender
    op::submit(s1, my_receiver());
  }

  {
    // test submission on a value task
    auto s2 = ex.make_value_task(std::move(s1), [] __host__ __device__ (fusing_executor<cuda::single_executor> ex)
    {
      printf("Hello world from value task\n");
      return 0;
    });

    op::submit(std::move(s2), my_receiver());

    if(cudaError_t error = cudaDeviceSynchronize())
    {
      throw std::runtime_error("CUDA error after cudaDeviceSynchronize: " + std::string(cudaGetErrorString(error)));
    }
  }

  //{
  //  // XXX this is currently not compiling
  //
  //  // test share on a value task
  //  auto s2 = ex.make_value_task(std::move(s1), [] __host__ __device__ (fusing_executor<cuda::single_executor> ex)
  //  {
  //    printf("Hello world from value task\n");
  //    return 13;
  //  });

  //  auto fut = op::share(std::move(s2));

  //  assert(fut.get() == 13);
  //}

  return 0; 
}

