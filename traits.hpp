#pragma once

#include <agency/execution/executor/properties/bulk_guarantee.hpp>

struct sender_tag {};

template<class OuterBulkGuarantee, class... InnerBulkGuarantees>
struct bulk_sender_tag {};

using grid_sender_tag = bulk_sender_tag<agency::bulk_guarantee_t::parallel_t, agency::bulk_guarantee_t::concurrent_t>;


template<class T>
struct sender_traits
{
  template<template<class...> class Variant, template<class...> class Tuple>
  using value_types = typename T::template value_types<Variant,Tuple>;
};


namespace detail
{


template<class T>
struct one_value_only { using type = T; };

template<class...>
struct zero_or_one_value {};

template<>
struct zero_or_one_value<>{ using type = void; };

template<class T>
struct zero_or_one_value<T>{ using type = T; };


} // end detail

// this implementation is suggested by P1341
template<class T>
using sender_value_type_t = 
  typename sender_traits<T>
    ::template value_types<detail::one_value_only, detail::zero_or_one_value>
    ::type::type;

