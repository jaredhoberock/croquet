#pragma once

#include <utility>

class inline_executor
{
  public:
    // XXX unimplemented due to circular dependency between just & inline_executor
    //__host__ __device__
    //just<inline_executor> schedule() const
    //{
    //  return {*this};
    //}

    #pragma nv_exec_check_disable
    template<class Function>
    __host__ __device__
    void execute(Function&& f) const noexcept
    {
      std::forward<Function>(f)();
    }
};

