#pragma once

struct sender_tag {};

struct bulk_sender_tag {};

// the idea is that some Executors may be able to execute types
// that are not necessarily Invocables, such as Senders or other types (e.g. CUDA Graphs)
template<class Executor, class Executable>
struct is_executor_of
{
};

