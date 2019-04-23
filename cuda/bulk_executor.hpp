#pragma once

#include <utility>
#include <type_traits>
#include <agency/cuda/execution/executor/grid_executor.hpp>
#include <agency/execution/executor/detail/execution_functions/then_execute.hpp>
#include "../just.hpp"
#include "../traits.hpp"
#include "../submit.hpp"
#include "../detail/receive_arguments_functor.hpp"
#include "future.hpp"
#include "promise.hpp"


namespace cuda
{
namespace detail
{


struct empty {};

struct empty_factory
{
  __host__ __device__
  empty operator()() const { return {}; }
};


template<class Function>
struct ignore_result
{
  Function f;

  template<class Index, class Result, class Outer, class Inner>
  __device__
  void operator()(const Index& idx, Result&, Outer& outer, Inner& inner) const
  {
    f(idx, outer, inner);
  }
};


} // end detail


class bulk_executor
{
  public:
    __host__ __device__
    just<bulk_executor> schedule() const
    {
      return {*this};
    }

    template<class T>
    cuda::promise<T> make_promise() const
    {
      return cuda::promise<T>();
    }

    using shape_type = typename agency::cuda::grid_executor::shape_type;
    using index_type = shape_type;

    template<class Function, class OuterFactory, class InnerFactory>
    __host__ __device__
    void bulk_execute(Function f, shape_type shape,
                      OuterFactory outer_factory,
                      InnerFactory inner_factory) const
    {
      agency::cuda::grid_executor ex;
      agency::cuda::async_future<void> ready = ex.make_ready_future();
      detail::ignore_result<Function> g{f};
      ex.bulk_then_execute(g, shape, ready, detail::empty_factory{}, outer_factory, inner_factory);
    }

  private:
    template<class Sender, class Function, class ResultFactory, class OuterFactory, class InnerFactory>
    struct grid_sender
    {
      using sender_concept = grid_sender_tag;

      template<template<class...> class Variant, template<class...> class Tuple>
      using value_types = Variant<Tuple<std::result_of_t<ResultFactory()>>>;

      using error_type = void;

      Sender predecessor;
      Function f;
      shape_type shape;
      ResultFactory result_factory;
      OuterFactory outer_factory;
      InnerFactory inner_factory;

      // this overload of share is for predecessors which may submit to the host
      // XXX we need a way to inspect the Sender to see where it will be submitted
      future<std::result_of_t<ResultFactory()>> share()
      {
        // create a promise for the predecessor's result
        cuda::promise<sender_value_type_t<Sender>> promise;

        // get the future
        agency::cuda::async_future<sender_value_type_t<Sender>> predecessor_future = promise.get_future().to_agency_cuda_async_future();
        
        // submit the predecessor to the promise
        op::submit(predecessor, std::move(promise));

        // bulk_then_execute on a grid_executor with the future as predecessor
        return agency::cuda::grid_executor{}.bulk_then_execute(f, shape, predecessor_future, result_factory, outer_factory, inner_factory);
      }

      // this overload of share is for predecessors which are known to submit to the device
      // XXX we need a way to inspect the Sender to see where it will be submitted
      // future<std::result_of_t<ResultFactory()>> share();

      template<class Receiver>
      void submit(Receiver r)
      {
        // share to get a future
        agency::cuda::async_future<std::result_of_t<ResultFactory()>> future = share().to_agency_cuda_async_future();

        // execute r once the future becomes ready
        agency::cuda::grid_executor ex;
        agency::detail::then_execute(ex, ::detail::receive_arguments_functor<Receiver>{std::move(r)}, future);
      }
    };

  public:
    template<class Sender, class Function, class ResultFactory, class OuterFactory, class InnerFactory>
    grid_sender<Sender,Function,ResultFactory,OuterFactory,InnerFactory>
      make_bulk_value_task(Sender predecessor, Function f, shape_type shape,
                           ResultFactory result_factory,
                           OuterFactory outer_factory,
                           InnerFactory inner_factory) const
    {
      return {std::move(predecessor), f, shape, result_factory, outer_factory, inner_factory};
    }
};


} // end cuda

