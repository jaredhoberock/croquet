#pragma once

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


} // end detail

