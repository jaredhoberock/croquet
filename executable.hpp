#pragma once

#include <utility>
#include <type_traits>


namespace op
{
namespace detail
{


template<class T>
struct static_const
{
  static constexpr T value{};
};

// provide the definition of static_const<T>::value
template<class T>
constexpr T static_const<T>::value;


template<class E, class F>
using make_executable_member_t = decltype(std::declval<E&>().make_executable(std::declval<F&>()));


template<class E, class F>
struct has_make_executable_member_impl
{
  template<class E_, class F_,
           class = make_executable_member_t<E_,F_>
          >
  static std::true_type test(int);

  template<class, class>
  static std::false_type test(...);

  using type = decltype(test<E,F>(0));
};

template<class E, class F>
using has_make_executable_member = typename has_make_executable_member_impl<E,F>::type;


struct make_executable_customization_point
{
  template<class Executor, class Function,
           class = std::enable_if_t<
             detail::has_make_executable_member<Executor&&,Function&&>::value
           >>
  __host__ __device__
  auto operator()(Executor&& ex, Function&& f) const ->
    decltype(std::forward<Executor>(ex).make_executable(std::forward<Function>(f)))
  {
    return std::forward<Executor>(ex).make_executable(std::forward<Function>(f));
  }

  // XXX free function overload would go here as well


  // default: just return the Function
  template<class Executor, class Function,
           class = std::enable_if_t<
             !detail::has_make_executable_member<Executor&&,Function&&>::value
           >>
  __host__ __device__
  auto operator()(Executor&&, Function&& f) const ->
    decltype(std::forward<Function>(f))
  {
    return std::forward<Function>(f);
  }
};


template<class E, class F, class P>
using make_dependent_executable_member_t = decltype(std::declval<E&>().make_dependent_executable(std::declval<F&>(), std::declval<P&>()));


template<class E, class F, class P>
struct has_make_dependent_executable_member_impl
{
  template<class E_, class F_, class P_,
           class = make_dependent_executable_member_t<E_,F_,P_>
          >
  static std::true_type test(int);

  template<class, class>
  static std::false_type test(...);

  using type = decltype(test<E,F,P>(0));
};

template<class E, class F, class P>
using has_make_dependent_executable_member = typename has_make_dependent_executable_member_impl<E,F,P>::type;


struct make_dependent_executable_customization_point
{
  template<class Executor, class Function, class Predecessor,
           class = std::enable_if_t<
             detail::has_make_dependent_executable_member<Executor&&,Function&&,Predecessor&&>::value
           >>
  __host__ __device__
  auto operator()(Executor&& ex, Function&& f, Predecessor&& p) const ->
    decltype(std::forward<Executor>(ex).make_executable(std::forward<Function>(f), std::forward<Predecessor>(p)))
  {
    return std::forward<Executor>(ex).make_executable(std::forward<Function>(f), std::forward<Predecessor>(p));
  }

  // XXX free function overload would go here as well


  // XXX not sure what could be the default
};


} // end detail


namespace
{


// define the CPOs
#ifndef __CUDA_ARCH__
constexpr auto const& make_executable = detail::static_const<detail::make_executable_customization_point>::value;
constexpr auto const& make_dependent_executable = detail::static_const<detail::make_dependent_executable_customization_point>::value;
#else
// CUDA __device__ functions cannot access global variables so make the CPOs __device__ variable in __device__ code
const __device__ detail::make_executable_customization_point make_executable;
const __device__ detail::make_dependent_executable_customization_point make_dependent_executable;
#endif


} // end anonymous namespace


} // end op

