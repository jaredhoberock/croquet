#pragma once

#include <utility>
#include <type_traits>
#include "detail/static_const.hpp"


namespace op
{
namespace detail
{


template<class S, class R>
using submit_member_t = decltype(std::declval<S>().submit(std::declval<R>()));

template<class S, class R>
struct has_submit_member_impl
{
  template<class S_, class R_,
           class = submit_member_t<S_,R_>
          >
  static std::true_type test(int);

  template<class,class>
  static std::false_type test(...);

  using type = decltype(test<S,R>(0));
};

template<class S, class R>
using has_submit_member = typename has_submit_member_impl<S,R>::type;


template<class S, class R>
using submit_free_function_t = decltype(submit(std::declval<S>(), std::declval<R>()));

template<class S, class R>
struct has_submit_free_function_impl
{
  template<class S_, class R_,
           class = submit_free_function_t<S_,R_>
          >
  static std::true_type test(int);

  template<class,class>
  static std::false_type test(...);

  using type = decltype(test<S,R>(0));
};

template<class S, class R>
using has_submit_free_function = typename has_submit_free_function_impl<S,R>::type;


struct submit_customization_point
{
  template<class Sender, class Receiver,
           class = std::enable_if_t<
             has_submit_member<Sender&&,Receiver&&>::value
          >>
  auto operator()(Sender&& s, Receiver&& r) const ->
    decltype(std::forward<Sender>(s).submit(std::forward<Receiver>(r)))
  {
    return std::forward<Sender>(s).submit(std::forward<Receiver>(r));
  }

  template<class Sender, class Receiver,
           class = std::enable_if_t<
             !has_submit_member<Sender&&,Receiver&&>::value
           >,
           class = std::enable_if_t<
             has_submit_free_function<Sender&&,Receiver&&>::value
           >>
  auto operator()(Sender&& s, Receiver&& r) const ->
    decltype(submit(std::forward<Sender>(s), std::forward<Receiver>(r)))
  {
    return submit(std::forward<Sender>(s), std::forward<Receiver>(r));
  }

  // XXX is there a default?
};


} // end detail


namespace
{


// define the CPO
#ifndef __CUDA_ARCH__
constexpr auto const& submit = ::detail::static_const<detail::submit_customization_point>::value;
#else
// CUDA __device__ functions cannot access global variables so make the CPO a __device__ variable in __device__ code
const __device__ detail::submit_customization_point submit;
#endif

} // end anonymous namespace


} // end op

