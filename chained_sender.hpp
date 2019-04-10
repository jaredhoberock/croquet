#pragma once

#include <utility>
#include <type_traits>
#include <cassert>
#include "detail/type_list.hpp"
#include "detail/set_value_functor.hpp"
#include "detail/noop_receiver.hpp"
#include "make_promise.hpp"
#include "traits.hpp"
#include "executable.hpp"

// this sender does eager enqueue upon submit before the Predecessor has completed
// XXX maybe name this eager_chained_sender and we could also implement lazy_chained_sender
template<class Executor, class Function, class Predecessor>
class chained_sender
{
  private:
    // XXX use sender_value_type_t
    using predecessor_result_type = detail::first_in_type_list_t<typename Predecessor::value_types>;

  public:
    using sender_concept = sender_tag;
    using value_types = detail::type_list<
      std::result_of_t<
        Function(predecessor_result_type)
      >
    >;
    using error_type = void;

    __host__ __device__
    chained_sender(const Executor& executor, Function function, Predecessor predecessor)
      : executor_(executor),
        function_(function),
        predecessor_(std::move(predecessor))
    {}

    template<class Receiver>
    void submit(Receiver r) const
    {
      // submit the predecessor and get a future
      auto promise = op::make_promise<predecessor_result_type>(executor_);
      auto dependency = promise.get_future();

      // XXX do this through a CPO
      // XXX how to ensure this recursion terminates? 
      predecessor_.submit(std::move(promise));

      // create a sender for the function
      // XXX do this through a CPO
      auto submit_me = executor_.make_value_task(std::move(dependency), function_);

      // submit the sender
      // XXX do this through a CPO
      // XXX how to ensure this recursion terminates? 
      submit_me.submit(r);
    }

    // XXX eliminate this
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
chained_sender<Executor, Function, Predecessor>
  make_chained_sender(const Executor& executor, Function f, Predecessor predecessor)
{
  return {executor, f, std::move(predecessor)};
}

