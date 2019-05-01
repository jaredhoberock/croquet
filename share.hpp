#pragma once

#include <utility>
#include <type_traits>
#include "detail/static_const.hpp"
#include "traits.hpp"
#include "make_promise.hpp"
#include "get_executor.hpp"


namespace op
{
namespace detail
{


template<class S>
using share_member_t = decltype(std::declval<S>().share());

template<class S>
struct has_share_member_impl
{
  template<class S_,
           class = share_member_t<S_>
          >
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<S>(0));
};

template<class S>
using has_share_member = typename has_share_member_impl<S>::type;


template<class S>
using share_free_function_t = decltype(share(std::declval<S>()));

template<class S>
struct has_share_free_function_impl
{
  template<class S_,
           class = share_free_function_t<S_>
          >
  static std::true_type test(int);

  template<class>
  static std::false_type test(...);

  using type = decltype(test<S>(0));
};

template<class S>
using has_share_free_function = typename has_share_free_function_impl<S>::type;


struct share_customization_point
{
  template<class Sender,
           class = std::enable_if_t<
             has_share_member<Sender&&>::value
          >>
  auto operator()(Sender&& s) const ->
    decltype(std::forward<Sender>(s).share())
  {
    return std::forward<Sender>(s).share();
  }

  template<class Sender,
           class = std::enable_if_t<
             !has_share_member<Sender&&>::value
           >,
           class = std::enable_if_t<
             has_share_free_function<Sender&&>::value
           >>
  auto operator()(Sender&& s) const ->
    decltype(share(std::forward<Sender>(s)))
  {
    return share(std::forward<Sender>(s));
  }


  template<class Sender,
           class = std::enable_if_t<
             !has_share_member<Sender&&>::value
           >,
           class = std::enable_if_t<
             !has_share_free_function<Sender&&>::value
           >>
  auto operator()(Sender&& s) const
  {
    using value_type = sender_value_type_t<std::decay_t<Sender>>;
    auto p = op::make_promise<value_type>(op::get_executor(std::forward<Sender>(s)));
    auto result = p.get_future();
    op::submit(std::forward<Sender>(s), std::move(p));
    return result;
  }
};


} // end detail


namespace
{


// define the CPO
#ifndef __CUDA_ARCH__
constexpr auto const& share = ::detail::static_const<detail::share_customization_point>::value;
#else
// CUDA __device__ functions cannot access global variables so make the CPO a __device__ variable in __device__ code
const __device__ detail::share_customization_point share;
#endif


} // end anonymous namespace


template<class Sender>
using share_t = decltype(op::share(std::declval<Sender>()));


} // end namespace op

