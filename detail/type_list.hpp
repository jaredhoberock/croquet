#pragma once

namespace detail
{


template<class...> struct type_list {};


template<class> struct first_in_type_list;

template<class T, class... Types>
struct first_in_type_list<type_list<T,Types...>>
{
  using type = T;
};

template<class TypeList>
using first_in_type_list_t = typename first_in_type_list<TypeList>::type;


} // end detail

