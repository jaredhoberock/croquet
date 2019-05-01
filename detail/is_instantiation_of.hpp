#pragma once

#include <type_traits>

namespace detail
{


template<class T, template<class...> class Template>
struct is_instantiation_of : std::false_type {};

template<class... Args, template<class...> class Template>
struct is_instantiation_of<Template<Args...>, Template> : std::true_type {};


} // end detail

