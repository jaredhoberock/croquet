#pragma once

#include <type_traits>
#include <agency/cuda/execution/executor/grid_executor.hpp>
#include <agency/async.hpp>
#include "just.hpp"
#include "single_sender.hpp"
#include "single_twoway_sender.hpp"


namespace detail
{


template<class F, class G>
struct composition
{
  // returns f of g
  #pragma nv_exec_check_disable
  __host__ __device__
  auto operator()() const
  {
    return f_(g_());
  }

  mutable F f_;
  mutable G g_;
};


template<class F, class G>
__host__ __device__
composition<std::decay_t<F>, std::decay_t<G>> compose(F&& f, G&& g)
{
  return {std::forward<F>(f), std::forward<G>(g)};
}


} // end detail


class single_oneway_executor
{
  public:
    __host__ __device__
    just<single_oneway_executor> schedule() const
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

    template<class Sender, class Function>
    __host__ __device__
    auto make_value_task(Sender predecessor, Function f) const
    {
      auto g = detail::compose(f, std::move(predecessor).function());
      return make_single_sender(std::move(g), *this);
    }
};


class single_twoway_executor
{
  public:
    __host__ __device__
    just<single_twoway_executor> schedule() const
    {
      return {*this};
    }

    template<class Function>
    struct twoway_function
    {
      Function f;
    };

    template<class Function>
    __host__ __device__
    static twoway_function<Function> make_executable(Function f)
    {
      return {f};
    }

    template<class Function, class T>
    struct dependent_function
    {
      Function f;
      agency::cuda::async_future<T> predecessor;
    };

    template<class Function, class T>
    __host__ __device__
    static dependent_function<Function,T>
      make_dependent_function(Function f, agency::cuda::async_future<T>&& predecessor)
    {
      return {f, std::move(predecessor)};
    }

    template<class Function>
    __host__ __device__
    agency::cuda::async_future<std::result_of_t<Function()>>
      execute(twoway_function<Function> f) const
    {
      agency::cuda::grid_executor ex;
      return agency::async(ex, f.f);
    }

    template<class Sender, class Function>
    __host__ __device__
    auto make_value_task(Sender predecessor, twoway_function<Function> f) const
    {
      return make_twoway_single_sender(*this, f.f, std::move(predecessor));
    }
};

