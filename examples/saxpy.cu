// $ nvcc -std=c++14 --expt-extended-lambda -I../ -I../agency saxpy.cu
#include <thrust/device_vector.h>
#include "../cuda/bulk_executor.hpp"
#include "../share.hpp"

struct ignored {};

// functions containing extended __device__ lambdas cannot have
// deduced return types so define this factory as a function object
struct ignored_factory
{
  __host__ __device__
  ignored operator()() const
  {
    return ignored{};
  }
};

struct my_receiver
{
  template<class T>
  __host__ __device__
  void set_value(T arg)
  {
#ifndef __CUDA_ARCH__
    std::cout << "my_receiver::set_value: Received " << typeid(arg).name() << std::endl;
#else
    printf("my_receiver::set_value: Hello world from my_receiver\n");
#endif
  }
};


// functions containing extended __device__ lambdas cannot have
// deduced return types so define the kernel as a function object
struct saxpy_kernel
{
  int n;
  int block_size;
  float a;
  const float* x;
  float* y;

  __device__
  void operator()(cuda::bulk_executor::index_type idx, ignored& predecessor, ignored& result, ignored& outer_shared, ignored& inner_shared)
  {
    int block_idx = idx[0];
    int thread_idx = idx[1];
    int linear_idx = block_idx * block_size + thread_idx;

    if(linear_idx < n)
    {
      y[linear_idx] = a * x[linear_idx] + y[linear_idx];
    }
  }
};


auto make_saxpy_task(const cuda::bulk_executor& ex, int n, float a, const float* x, float* y)
{
  // choose a shape
  int block_size = 256;
  int num_blocks = (n + block_size - 1) / block_size;

  // create the task
  return ex.make_bulk_value_task(
    just<ignored>(),                  // predecessor
    saxpy_kernel{n,block_size,a,x,y}, // function
    {num_blocks, block_size},         // shape
    ignored_factory{},                // result factory
    ignored_factory{},                // outer factory
    ignored_factory{}                 // inner factory
  );
}


void test(size_t n)
{
  thrust::device_vector<float> x(n, 1);
  thrust::device_vector<float> y(n, 2);
  float a = 2;

  cuda::bulk_executor ex;
  auto saxpy_task = make_saxpy_task(ex, n, a, x.data().get(), y.data().get());

  {
    // test submit

    thrust::fill(x.begin(), x.end(), 1);
    thrust::fill(y.begin(), y.end(), 2);

    op::submit(saxpy_task, my_receiver{});

    if(cudaError_t error = cudaDeviceSynchronize())
    {
      throw std::runtime_error("CUDA error after cudaDeviceSynchronize: " + std::string(cudaGetErrorString(error)));
    }

    thrust::device_vector<float> reference(n, 4);
    assert(reference == y);
  }

  {
    // test share

    thrust::fill(x.begin(), x.end(), 1);
    thrust::fill(y.begin(), y.end(), 2);

    auto future = op::share(saxpy_task);

    future.wait();

    thrust::device_vector<float> reference(n, 4);
    assert(reference == y);
  }
}


int main(int argc, char** argv)
{
  size_t n = 1 << 25;
  if(argc > 1)
  {
    n = std::atoi(argv[1]);
  }

  test(n);

  std::cout << "OK" << std::endl;

  return 0;
}

