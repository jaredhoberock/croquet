#pragma once

#include <utility>
#include <type_traits>
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


template<class SingleExecutor>
class fusing_executor
{
  public:
    fusing_executor() = default;
    fusing_executor(const fusing_executor&) = default;

    __host__ __device__
    fusing_executor(const SingleExecutor& executor)
      : executor_(executor)
    {}

    __host__ __device__
    just<fusing_executor> schedule() const
    {
      // XXX how should an adaptor implement schedule?
      return {*this};
    }

    template<class Function>
    __host__ __device__
    void execute(Function f) const
    {
      return executor_.execute(std::move(f));
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

  private:
    SingleExecutor executor_;
};

template<class SingleExecutor>
__host__ __device__
fusing_executor<SingleExecutor> make_fusing_executor(const SingleExecutor& executor)
{
  return {executor};
}

