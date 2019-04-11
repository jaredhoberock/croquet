// $ nvcc --expt-extended-lambda -std=c++14 -I. chaining_executor_demo.cu
#include <iostream>
#include <typeinfo>
#include <cassert>
#include "chaining_executor.hpp"
#include "submit.hpp"
#include "cuda/single_executor.hpp"


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
  chaining_executor<cuda::single_executor> ex = make_chaining_executor(cuda::single_executor());

  just<chaining_executor<cuda::single_executor>> s1 = ex.schedule();

  op::submit(s1, my_receiver());

  auto s2 = ex.make_value_task(std::move(s1), [] __host__ __device__ (chaining_executor<cuda::single_executor>)
  {
    printf("Hello world from value task\n");
    return 0;
  });

  op::submit(std::move(s2), my_receiver());

  if(cudaError_t error = cudaDeviceSynchronize())
  {
    throw std::runtime_error("CUDA error after cudaDeviceSynchronize: " + std::string(cudaGetErrorString(error)));
  }

  return 0; 
}

