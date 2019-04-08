#pragma once

#include <utility>
#include <type_traits>
#include <cassert>
#include "detail/type_list.hpp"
#include "detail/set_value_functor.hpp"
#include "detail/noop_receiver.hpp"
#include "traits.hpp"
#include "executable.hpp"


// XXX call this chaining_sender or something
template<class Executor, class Function, class Predecessor>
class twoway_single_sender
{
  private:
    // XXX use sender_value_type_t
    using predecessor_arg_type = detail::first_in_type_list_t<typename Predecessor::value_types>;

  public:
    using sender_concept = sender_tag;
    using value_types = detail::type_list<
      std::result_of_t<Function(predecessor_arg_type)>
    >;
    using error_type = void;

    __host__ __device__
    twoway_single_sender(const Executor& executor, Function function, Predecessor predecessor)
      : executor_(executor),
        function_(function),
        predecessor_(std::move(predecessor))
    {}

    template<class Receiver>
    __host__ __device__
    auto submit(Receiver r) const
    {
      // need some sort of cuda promise
      // would need to specialize submit(sender, cuda_promise)

      // XXX need a promise/future pair for executor_
      //     the receiver would set_value
      auto dependency = predecessor_.submit(detail::noop_receiver());
      detail::set_value_functor<Function, Receiver> f{function_, r};
      auto executable = op::make_dependent_executable(executor_, std::move(f), std::move(dependency));
      return executor_.execute(std::move(executable));
    }

    __host__ __device__
    Function function() const
    {
      return function_;
    }
    
  private:
    Executor executor_;
    Function function_;
    Predecessor predecessor_;
};


template<class Executor, class Function, class Predecessor>
__host__ __device__
twoway_single_sender<Executor, Function, Predecessor>
  make_twoway_single_sender(const Executor& executor, Function f, Predecessor predecessor)
{
  return {executor, f, std::move(predecessor)};
}

