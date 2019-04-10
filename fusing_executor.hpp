#pragma once

#include <utility>
#include <type_traits>
#include <agency/cuda.hpp>
#include "just.hpp"
#include "fused_sender.hpp"

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


class fusing_executor
{
  public:
    __host__ __device__
    just<fusing_executor> schedule() const
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

    // XXX presumably something like this would be the default implementation of make_value_task
    template<class Sender, class Function>
    __host__ __device__
    auto make_value_task(Sender predecessor, Function f) const
    {
      // XXX need to do this composition with a receiver instead of by grabbing the function
      // XXX what happens to predecessor's executor?
      auto g = detail::compose(f, std::move(predecessor).function());
      return make_fused_sender(std::move(g), *this);
    }
};

