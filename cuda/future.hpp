#pragma once

#include <agency/cuda.hpp>

namespace cuda
{


template<class T>
using future = agency::cuda::async_future<T>;


} // end cuda

