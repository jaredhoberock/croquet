#pragma once

#include <utility>
#include <type_traits>
#include "detail/static_const.hpp"


namespace op
{
namespace detail
{


template<class S>
using executor_member_t = decltype(std::declval<S>().executor());

template<class S>
struct has_executor_member_impl
{
  template<class S_,
           class = executor_member_t<S_>
          >
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<S>(0));
};

template<class S>
using has_executor_member = typename has_executor_member_impl<S>::type;


template<class S>
using executor_free_function_t = decltype(executor(std::declval<S>()));

template<class S>
struct has_executor_free_function_impl
{
  template<class S_,
           class = executor_free_function_t<S_>
          >
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<S>(0));
};

template<class S>
using has_executor_free_function = typename has_executor_free_function_impl<S>::type;


struct get_executor_customization_point
{
  template<class Sender,
           class = std::enable_if_t<
             has_executor_member<Sender&&>::value
          >>
  auto operator()(Sender&& s) const ->
    decltype(std::forward<Sender>(s).executor())
  {
    return std::forward<Sender>(s).executor();
  }

  template<class Sender,
           class = std::enable_if_t<
             !has_executor_member<Sender&&>::value
           >,
           class = std::enable_if_t<
             has_executor_free_function<Sender&&>::value
           >>
  auto operator()(Sender&& s) const ->
    decltype(executor(std::forward<Sender>(s)))
  {
    return executor(std::forward<Sender>(s));
  }
};


} // end detail


namespace
{


// define the CPO
#ifndef __CUDA_ARCH__
constexpr auto const& get_executor = ::detail::static_const<detail::get_executor_customization_point>::value;
#else
// CUDA __device__ functions cannot access global variables so make the CPO a __device__ variable in __device__ code
const __device__ detail::get_executor_customization_point get_executor;
#endif


} // end anonymous namespace


} // end namespace op

