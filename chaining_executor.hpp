#pragma once

#include <type_traits>
#include <agency/cuda/execution/executor/grid_executor.hpp>
#include <agency/execution/executor/detail/execution_functions/then_execute.hpp>
#include <agency/async.hpp>
#include <agency/detail/type_traits.hpp>
#include "just.hpp"
#include "chained_sender.hpp"


template<class SingleExecutor>
class chaining_executor
{
  public:
    chaining_executor() = default;
    chaining_executor(const chaining_executor&) = default;

    __host__ __device__
    chaining_executor(const SingleExecutor& executor)
      : executor_(executor)
    {}

    __host__ __device__
    just<chaining_executor> schedule() const
    {
      // XXX how should an adaptor implement schedule?
      return {*this};
    }

    template<class Function>
    __host__ __device__
    void execute(Function f) const
    {
      executor_.execute(std::move(f));
    }

    template<class Sender, class Function>
    __host__ __device__
    auto make_value_task(Sender predecessor, Function f) const
    {
      return make_chained_sender(*this, f, std::move(predecessor));
    }

    // XXX need a general way to forward to an adapted type's customizations
    template<class T>
    auto make_promise() const
    {
      return op::make_promise<T>(executor_);
    }

  private:
    SingleExecutor executor_;
};

template<class SingleExecutor>
__host__ __device__
chaining_executor<SingleExecutor> make_chaining_executor(const SingleExecutor& executor)
{
  return {executor};
}

