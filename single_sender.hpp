#pragma once

#include <utility>
#include <type_traits>
#include <cassert>
#include "detail/type_list.hpp"
#include "traits.hpp"

namespace detail
{


template<class Function, class Receiver>
struct set_value_functor
{
  mutable Function f;
  mutable Receiver r;

  __host__ __device__
  void operator()() const
  {
    r.set_value(f());
  }
};


} // end detail


template<class Function, class Executor>
class single_sender
{
  public:
    using sender_concept = sender_tag;
    using value_types = detail::type_list<std::result_of_t<Function()>>;
    using error_type = void;

    template<class OtherFunction,
             class = std::enable_if_t<
               std::is_constructible<Function,OtherFunction&&>::value
             >>
    __host__ __device__
    single_sender(OtherFunction&& function, const Executor& executor)
      : function_(std::forward<OtherFunction>(function)),
        executor_(executor)
    {}

    template<class Receiver>
    __host__ __device__
    void submit(Receiver r) const
    {
      detail::set_value_functor<Function, Receiver> f{function_, r};
      executor_.execute(f);
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

