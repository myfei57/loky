API Reference
=============


.. automodule:: loky
    :members: get_reusable_executor

.. autoclass:: ProcessPoolExecutor
    :members:
    :class-doc-from: init


Task & results serialization
----------------------------

To share function definition across multiple python processes, it is necessary to rely on a serialization protocol. The standard protocol in python is :mod:`pickle` but its default implementation in the standard library has several limitations. For instance, it cannot serialize functions which are defined interactively or in the :code:`__main__` module.

To avoid this limitation, :mod:`loky` relies on |cloudpickle| when it is present. |cloudpickle| is a fork of the pickle protocol which allows the serialization of a greater number of objects and it can be installed using :code:`pip install cloudpickle`. |cloudpickle| is used by default to serialize tasks and results, because it can handle interactively defined functions, lambdas, and locally defined classes that the standard pickle module cannot pickle. Recent versions of |cloudpickle| are based on the C implementation of pickle, so this is not a significant performance trade-off compared to plain pickle for most workloads.

There are three ways to temper with the serialization in :mod:`loky`:

- Using the arguments :code:`job_reducers` and :code:`result_reducers`, it is possible to register custom reducers for the serialization process.
- Setting the variable :code:`LOKY_PICKLER` to an available and valid serialization module. This module must present a valid :code:`Pickler` object. Setting the environment variable :code:`LOKY_PICKLER=cloudpickle` will force :mod:`loky` to serialize everything with |cloudpickle| instead of just serializing the object detected to be in the :code:`__main__` module.
- Finally, it is possible to wrap an unpicklable object using the :code:`loky.wrap_non_picklable_objects` decorator. In this case, all other objects are serialized as in the default behavior and the wrapped object is pickled through |cloudpickle|.

The benefits and drawbacks of each method are highlighted in this example_.


.. autofunction:: loky.wrap_non_picklable_objects


Publishing read-only objects once
----------------------------------

When the same large read-only input is needed by many tasks, pickling it with
every :code:`submit` call is wasteful. :code:`ProcessPoolExecutor.publish_object`
serializes such an object a single time, synchronously, using the executor
:code:`job_reducers` rules, and keeps the resulting snapshot in the parent
process. It returns an opaque :class:`~loky.ObjectHandle` bound to the
executor identity and to the publish generation:

.. code-block:: python

    handle = executor.publish_object(large_read_only_input)
    future = executor.submit_with_objects(worker_fn, [handle], other_arg)

:code:`submit_with_objects` only resolves the handles listed in its explicit
:code:`objects` argument and passes the restored objects to the user function
as independent leading positional arguments, in list order. The regular
arguments are never inspected for handles, and a plain :code:`submit` call
never resolves a handle either.

Each worker process deserializes the snapshot at most once and then reuses
its own local copy; worker copies are independent and the contract is
read-only (loky does not provide shared mutable memory). If an idle worker
exits or is recycled, its replacement restores the same snapshot from the
parent on demand. :meth:`ObjectHandle.release` is idempotent: it rejects new
submissions while letting already accepted tasks complete, and the parent
snapshot and the worker copies are released only once the last in-flight user
finishes.

.. autoclass:: loky.ObjectHandle
    :members:
    :special-members: release


Protection against memory leaks
-------------------------------
The memory size of long running worker processes can increase indefinitely if a
memory leak is created. This can result in processes being shut down by the OS if
those leaks are not resolved. To
prevent it, loky provides leak detection, memory cleanups, and workers 
shutdown.

If :mod:`psutil` is installed, each worker periodically [#periodically_fn]_ checks its 
memory usage after it completes its task. If the usage is found to be
unusual [#psutil_unusual_fn]_, an additional :code:`gc.collect()` event is triggered to remove 
objects with potential cyclic references. 
If even after that, the memory usage of a process worker remains too high, 
it will shut down safely, and a fresh process will be automatically spawned by
the executor. 

If :mod:`psutil` is not installed, there is no easy way to monitor worker
processes memory usage. :code:`gc.collect()` events will still be called
periodially [#periodically_fn]_ inside each workers, but there is no guarantee that a leak is
not happening.

The reference is measured after the first task only and is never updated
afterwards, so a workload whose tasks differ a lot in memory footprint can cross
the threshold without leaking anything, and pay for a worker restart every time
it does. Setting the environment variable :code:`LOKY_MAX_MEMORY_LEAK_SIZE` to a
number of bytes raises, or lowers, that threshold. It is read when
:code:`loky.process_executor` is imported, in the workers as well as in the
parent process, so it has to be set before the workers are started.

.. rubric:: Footnotes

.. [#periodically_fn] every 1 second. This constant is defined in :code:`loky.process_executor._MEMORY_LEAK_CHECK_DELAY`
.. [#psutil_unusual_fn] an increase of 300MB compared to a reference, which is defined as the residual memory usage of the worker after it completed its first task. This constant is defined in :code:`loky.process_executor._MAX_MEMORY_LEAK_SIZE` and can be overridden with the :code:`LOKY_MAX_MEMORY_LEAK_SIZE` environment variable.



.. |cloudpickle| raw:: html

    <a href="https://github.com/cloudpipe/cloudpickle"><code>cloudpickle</code></a>

.. _example :  auto_examples/cloudpickle_wrapper.html
