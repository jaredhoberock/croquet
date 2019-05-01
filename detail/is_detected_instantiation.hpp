#pragma once

#include "is_detected.hpp"
#include "is_instantiation_of.hpp"

namespace detail
{


template<template<class...> class Expected, template<class...> class Op, class... Args>
using is_detected_instantiation = is_instantiation_of<detected_t<Op,Args...>, Expected>;


} // end detail

