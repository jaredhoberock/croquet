#pragma once

#include <utility>
#include <future>
#include "detail/static_const.hpp"


namespace op
{
namespace detail
{


template<class E, class T>
using make_promise_member_t = decltype(std::declval<E&>().template make_promise<T>());


template<class E, class T>
struct has_make_promise_member_impl
{
  template<class E_, class T_,
           class = make_promise_member_t<E_,T_>
          >
  static std::true_type test(int);

  template<class, class>
  static std::false_type test(...);

  using type = decltype(test<E,T>(0));
};


template<class E, class T>
using has_make_promise_member = typename has_make_promise_member_impl<E,T>::type;


// XXX it might be a useful to make the executor on which .set_value() will be called a second parameter
//     OTOH, that other executor wouldn't always be available
template<class T>
struct make_promise_customization_point
{
  template<class Executor,
           class = std::enable_if_t<
             has_make_promise_member<const Executor&,T>::value
           >>
  auto operator()(const Executor& executor) const ->
    decltype(executor.template make_promise<T>())
  {
    return executor.template make_promise<T>();
  }

  // XXX free function overload would go here as well

  // by default, return a std::promise<T>
  // XXX probably need to constrain Executor to have mapping.thread or some sort of strong fwd progress
  template<class Executor,
           class = std::enable_if_t<
             !detail::has_make_promise_member<const Executor&,T>::vaslue
           >>
  std::promise<T> operator()(const Executor&) const
  {
    return std::promise<T>();
  }
};


} // end detail


namespace
{


// define the CPO
#ifndef __CUDA_ARCH__
template<class T>
constexpr auto const& make_promise = ::detail::static_const<detail::make_promise_customization_point<T>>::value;
#else
// CUDA __device__ functions cannot access global variables so make the CPO a __device__ variable in __device__ code
template<class T>
const __device__ detail::make_promise_customization_point<T> make_promise;
#endif


} // end anonymous namespace


} // end op

