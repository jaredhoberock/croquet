#pragma once

#include "../just.hpp"
#include <agency/cuda/execution/executor/grid_executor.hpp>
#include <agency/async.hpp>

namespace cuda
{


class single_executor
{
  public:
    __host__ __device__
    just<single_executor> schedule() const
    {
      return {*this};
    }

    template<class Function>
    __host__ __device__
    void execute(Function f) const
    {
      agency::cuda::grid_executor ex;
      agency::async(ex, f);
    }
};


} // cuda

