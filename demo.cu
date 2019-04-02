// $ nvcc --expt-extended-lambda -std=c++14 demo.cu
#include "cuda_executor.hpp"
#include <iostream>
#include <typeinfo>

struct my_receiver
{
  template<class T>
  __host__ __device__
  void set_value(T arg)
  {
#ifndef __CUDA_ARCH__
    std::cout << "receive_executor::set_value: received " << typeid(arg).name() << std::endl;
#else
    printf("Hello world from my_receiver\n");
#endif
  }
};

int main()
{
  cuda_executor ex;

  just<cuda_executor> s1 = ex.schedule();

  s1.submit(my_receiver());

  auto s2 = ex.make_value_task(std::move(s1), [] __device__ (cuda_executor)
  {
    printf("Hello world from value task\n");
    return 0;
  });

  std::move(s2).submit(my_receiver());

  cudaDeviceSynchronize();

  return 0; 
}

