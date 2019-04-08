#pragma once

#include <utility>
#include <type_traits>
#include <cassert>
#include "detail/type_list.hpp"
#include "detail/set_value_functor.hpp"
#include "traits.hpp"
#include "executable.hpp"


// XXX call this fusing_sender or something
template<class Function, class Executor>
class single_sender
{
  public:
    using sender_concept = sender_tag;
    using value_types = detail::type_list<std::result_of_t<Function()>>;
    using error_type = void;

    __host__ __device__
    single_sender(Function function, const Executor& executor)
      : function_(function),
        executor_(executor)
    {}

    template<class Receiver>
    __host__ __device__
    auto submit(Receiver r) const
    {
      detail::set_value_functor<Function, Receiver> f{function_, r};
      auto executable = op::make_executable(executor_, std::move(f));
      return executor_.execute(std::move(executable));
    }

    __host__ __device__
    Function function() const
    {
      return function_;
    }
    
  private:
    Function function_;
    Executor executor_;
};

template<class Function, class Executor>
__host__ __device__
single_sender<std::decay_t<Function>,Executor>
  make_single_sender(Function&& f, const Executor& executor)
{
  return {std::forward<Function>(f), executor};
}

