// $ nvcc --expt-extended-lambda -std=c++14 -I../agency-tot fusing_executor_demo.cu
#include <iostream>
#include <typeinfo>
#include "fusing_executor.hpp"

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
  fusing_executor ex;

  just<fusing_executor> s1 = ex.schedule();

  s1.submit(my_receiver());

  auto s2 = ex.make_value_task(std::move(s1), [] __host__ __device__ (fusing_executor)
  {
    printf("Hello world from value task\n");
    return 0;
  });

  std::move(s2).submit(my_receiver());

  if(cudaError_t error = cudaDeviceSynchronize())
  {
    throw std::runtime_error("CUDA error after cudaDeviceSynchronize: " + std::string(cudaGetErrorString(error)));
  }

  return 0; 
}

