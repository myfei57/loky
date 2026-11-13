###############################################################################
# Re-implementation of the ProcessPoolExecutor more robust to faults
#
# author: Thomas Moreau and Olivier Grisel
#
# adapted from concurrent/futures/process_pool_executor.py (17/02/2017)
#  * Add an extra management thread to detect executor_manager_thread failures,
#  * Improve the shutdown process to avoid deadlocks,
#  * Add timeout for workers,
#  * More robust pickling process.
#
# Copyright 2009 Brian Quinlan. All Rights Reserved.
# Licensed to PSF under a Contributor Agreement.

"""Implements ProcessPoolExecutor.

The follow diagram and text describe the data-flow through the system:

|======================= In-process =====================|== Out-of-process ==|

+----------+     +----------+       +--------+     +-----------+    +---------+
|          |  => | Work Ids |       |        |     | Call Q    |    | Process |
|          |     +----------+       |        |     +-----------+    |  Pool   |
|          |     | ...      |       |        |     | ...       |    +---------+
|          |     | 6        |    => |        |  => | 5, call() | => |         |
|          |     | 7        |       |        |     | ...       |    |         |
| Process  |     | ...      |       | Local  |     +-----------+    | Process |
|  Pool    |     +----------+       | Worker |                      |  #1..n  |
| Executor |                        | Thread |                      |         |
|          |     +----------- +     |        |     +-----------+    |         |
|          | <=> | Work Items | <=> |        | <=  | Result Q  | <= |         |
|          |     +------------+     |        |     +-----------+    |         |
|          |     | 6: call()  |     |        |     | ...       |    |         |
|          |     |    future  |     +--------+     | 4, result |    |         |
|          |     | ...        |                    | 3, except |    |         |
+----------+     +------------+                    +-----------+    +---------+

Executor.submit() called:
- creates a uniquely numbered _WorkItem and adds it to the "Work Items" dict
- adds the id of the _WorkItem to the "Work Ids" queue

Local worker thread:
- reads work ids from the "Work Ids" queue and looks up the corresponding
  WorkItem from the "Work Items" dict: if the work item has been cancelled then
  it is simply removed from the dict, otherwise it is repackaged as a
  _CallItem and put in the "Call Q". New _CallItems are put in the "Call Q"
  until "Call Q" is full. NOTE: the size of the "Call Q" is kept small because
  calls placed in the "Call Q" can no longer be cancelled with Future.cancel().
- reads _ResultItems from "Result Q", updates the future stored in the
  "Work Items" dict and deletes the dict entry

Process #1..n:
- reads _CallItems from "Call Q", executes the calls, and puts the resulting
  _ResultItems in "Result Q"
"""


__author__ = "Thomas Moreau (thomas.moreau.2010@gmail.com)"


import faulthandler
import os
import gc
import sys
import queue
import struct
import weakref
import warnings
import itertools
import traceback
import threading
from time import time, sleep
import multiprocessing as mp
from functools import partial
from pickle import PicklingError
from concurrent.futures import Executor, InvalidStateError
from concurrent.futures._base import LOGGER
from concurrent.futures.process import BrokenProcessPool as _BPPException
from multiprocessing.connection import wait

from ._base import Future
from .backend import get_context
from .backend.context import cpu_count, _MAX_WINDOWS_WORKERS
from .backend.queues import Queue, SimpleQueue
from .backend.reduction import set_loky_pickler, get_loky_pickler_name
from .backend.utils import kill_process_tree, get_exitcodes_terminated_worker
from .initializers import _prepare_initializer


# Mechanism to prevent infinite process spawning. When a worker of a
# ProcessPoolExecutor nested in MAX_DEPTH Executor tries to create a new
# Executor, a LokyRecursionError is raised
MAX_DEPTH = int(os.environ.get("LOKY_MAX_DEPTH", 10))
_CURRENT_DEPTH = 0

# Minimum time interval between two consecutive memory leak protection checks.
_MEMORY_LEAK_CHECK_DELAY = 1.0

# Number of bytes of memory usage allowed over the reference process size.
# Workloads whose tasks differ a lot in size can legitimately exceed the default
# without leaking, so it is configurable; it is read in the worker processes,
# which inherit the environment of the process that created the executor.
_MAX_MEMORY_LEAK_SIZE = int(
    float(os.environ.get("LOKY_MAX_MEMORY_LEAK_SIZE", 3e8))
)


try:
    from psutil import Process

    _USE_PSUTIL = True

    def _get_memory_usage(pid, force_gc=False):
        if force_gc:
            gc.collect()

        mem_size = Process(pid).memory_info().rss
        mp.util.debug(f"psutil return memory size: {mem_size}")
        return mem_size

except ImportError:
    _USE_PSUTIL = False


class _RecycledWorkerPid(int):
    """PID of a worker that the executor itself decided to shut down.

    Behaves like the plain PID the workers usually send back, but lets the
    manager thread tell a recycling it triggered on purpose from a worker that
    stopped on its own.
    """


class _ThreadWakeup:
    def __init__(self):
        self._closed = False
        self._reader, self._writer = mp.Pipe(duplex=False)

    def close(self):
        if not self._closed:
            self._closed = True
            self._writer.close()
            self._reader.close()

    def wakeup(self):
        if not self._closed:
            self._writer.send_bytes(b"")

    def clear(self):
        if not self._closed:
            while self._reader.poll():
                self._reader.recv_bytes()


class _ExecutorFlags:
    """necessary references to maintain executor states without preventing gc

    It permits to keep the information needed by executor_manager_thread
    and crash_detection_thread to maintain the pool without preventing the
    garbage collection of unreferenced executors.
    """

    def __init__(self, shutdown_lock):

        self.shutdown = False
        self.broken = None
        self.kill_workers = False
        self.shutdown_lock = shutdown_lock

    def flag_as_shutting_down(self, kill_workers=None):
        with self.shutdown_lock:
            self.shutdown = True
            if kill_workers is not None:
                self.kill_workers = kill_workers

    def flag_as_broken(self, broken):
        with self.shutdown_lock:
            self.shutdown = True
            self.broken = broken


# Prior to 3.9, executor_manager_thread is created as daemon thread. This means
# that it is not joined automatically when the interpreter is shutting down.
# To work around this problem, an exit handler is installed to tell the
# thread to exit when the interpreter is shutting down and then waits until
# it finishes. The thread needs to be daemonized because the atexit hooks are
# called after all non daemonized threads are joined.
#
# Starting 3.9, there exists a specific atexit hook to be called before joining
# the threads so the executor_manager_thread does not need to be daemonized
# anymore.
#
# The atexit hooks are registered when starting the first ProcessPoolExecutor
# to avoid import having an effect on the interpreter.

_global_shutdown = False
_global_shutdown_lock = threading.Lock()
_threads_wakeups = weakref.WeakKeyDictionary()


def _python_exit():
    global _global_shutdown
    _global_shutdown = True

    # Materialize the list of items to avoid error due to iterating over
    # changing size dictionary.
    items = list(_threads_wakeups.items())
    if len(items) > 0:
        mp.util.debug(
            f"Interpreter shutting down. Waking up {len(items)}"
            f"executor_manager_thread:\n{items}"
        )

    # Wake up the executor_manager_thread's so they can detect the interpreter
    # is shutting down and exit.
    for _, (shutdown_lock, thread_wakeup) in items:
        with shutdown_lock:
            thread_wakeup.wakeup()

    # Collect the executor_manager_thread's to make sure we exit cleanly.
    for thread, _ in items:
        # This locks is to prevent situations where an executor is gc'ed in one
        # thread while the atexit finalizer is running in another thread.
        with _global_shutdown_lock:
            thread.join()


# With the fork context, _thread_wakeups is propagated to children.
# Clear it after fork to avoid some situation that can cause some
# freeze when joining the workers.
mp.util.register_after_fork(_threads_wakeups, lambda obj: obj.clear())


# Module variable to register the at_exit call
process_pool_executor_at_exit = None

# Controls how many more calls than processes will be queued in the call queue.
# A smaller number will mean that processes spend more time idle waiting for
# work while a larger number will make Future.cancel() succeed less frequently
# (Futures in the call queue cannot be cancelled).
EXTRA_QUEUED_CALLS = 1


class _RemoteTraceback(Exception):
    """Embed stringification of remote traceback in local traceback"""

    def __init__(self, tb=None):
        self.tb = f'\n"""\n{tb}"""'

    def __str__(self):
        return self.tb


# Do not inherit from BaseException to mirror
# concurrent.futures.process._ExceptionWithTraceback
class _ExceptionWithTraceback:
    def __init__(self, exc):
        tb = getattr(exc, "__traceback__", None)
        if tb is None:
            _, _, tb = sys.exc_info()
        tb = traceback.format_exception(type(exc), exc, tb)
        tb = "".join(tb)
        self.exc = exc
        self.tb = tb

    def __reduce__(self):
        return _rebuild_exc, (self.exc, self.tb)


def _rebuild_exc(exc, tb):
    exc.__cause__ = _RemoteTraceback(tb)
    return exc


class _WorkItem:

    __slots__ = ["future", "fn", "args", "kwargs"]

    def __init__(self, future, fn, args, kwargs):
        self.future = future
        self.fn = fn
        self.args = args
        self.kwargs = kwargs


class _ResultItem:
    def __init__(self, work_id, exception=None, result=None):
        self.work_id = work_id
        self.exception = exception
        self.result = result


class _CallItem:
    def __init__(self, work_id, fn, args, kwargs):
        self.work_id = work_id
        self.fn = fn
        self.args = args
        self.kwargs = kwargs

        # Store the current loky_pickler so it is correctly set in the worker
        self.loky_pickler = get_loky_pickler_name()

    def __call__(self):
        set_loky_pickler(self.loky_pickler)
        return self.fn(*self.args, **self.kwargs)

    def __repr__(self):
        return (
            f"CallItem({self.work_id}, {self.fn}, {self.args}, {self.kwargs})"
        )


# ---------------------------------------------------------------------------
# Streaming protocol (ProcessPoolExecutor.submit_stream)
#
# Worker -> parent messages travel on the shared result queue. Every message
# carries the stream identity (``work_id``) and data messages additionally
# carry a per-stream strictly monotonic sequence number, so messages of
# interleaved streams can never be mixed up.
#
# Parent -> worker control messages (acknowledgements and close requests)
# travel on a dedicated one-way pipe created per stream. A pipe per stream
# lets a blocked worker wait on its own control channel without stealing
# messages addressed to another stream.
# ---------------------------------------------------------------------------

# Control messages sent on the per-stream control pipe.
_STREAM_ACK = "loky-stream-ack"
_STREAM_CLOSE = "loky-stream-close"

# Kinds of terminal stream messages.
_STREAM_COMPLETE = "complete"
_STREAM_CLOSED = "closed"


class _StreamMessage:
    """Base class for the worker -> parent messages of a stream."""

    __slots__ = ("work_id",)

    def __init__(self, work_id):
        self.work_id = work_id


class _StreamItem(_StreamMessage):
    """A single item yielded by the worker-side generator."""

    __slots__ = ("seq", "item")

    def __init__(self, work_id, seq, item):
        self.work_id = work_id
        self.seq = seq
        self.item = item


class _StreamEnd(_StreamMessage):
    """Terminal message: the generator completed or was closed cooperatively."""

    __slots__ = ("kind", "result")

    def __init__(self, work_id, kind, result=None):
        self.work_id = work_id
        self.kind = kind
        self.result = result


class _StreamError(_StreamMessage):
    """Terminal message: the generator or the serialization of an item failed."""

    __slots__ = ("exception",)

    def __init__(self, work_id, exception):
        self.work_id = work_id
        self.exception = exception


class _StreamCallItem(_CallItem):
    """A call item asking a worker to drive a streaming generator."""

    __slots__ = ("max_buffered_items", "stream_conn")

    def __init__(
        self, work_id, fn, args, kwargs, max_buffered_items, stream_conn
    ):
        super().__init__(work_id, fn, args, kwargs)
        # Size of the backpressure window: the worker may have at most this
        # many yielded items waiting to be consumed by the parent process.
        self.max_buffered_items = max_buffered_items
        # Read end of the per-stream control pipe (ack / close messages).
        self.stream_conn = stream_conn


class _StreamWorkItem(_WorkItem):
    """Parent-side bookkeeping for a submitted stream."""

    __slots__ = ("stream", "max_buffered_items", "stream_conn")

    def __init__(
        self, stream, fn, args, kwargs, max_buffered_items, stream_conn
    ):
        # The ``future`` slot of a regular work item is replaced by the
        # StreamResult, which exposes the stream state to the caller.
        super().__init__(stream, fn, args, kwargs)
        self.stream = stream
        self.max_buffered_items = max_buffered_items
        # Parent-side read end of the control pipe. It is kept open until the
        # stream terminates so that the call queue feeder thread can safely
        # duplicate it into the worker; it is then closed deterministically.
        self.stream_conn = stream_conn


class _SafeQueue(Queue):
    """Safe Queue set exception to the future object linked to a job"""

    def __init__(
        self,
        max_size=0,
        ctx=None,
        pending_work_items=None,
        running_work_items=None,
        thread_wakeup=None,
        shutdown_lock=None,
        executor_flags=None,
        reducers=None,
    ):
        self.thread_wakeup = thread_wakeup
        self.shutdown_lock = shutdown_lock
        self.executor_flags = executor_flags
        self.pending_work_items = pending_work_items
        self.running_work_items = running_work_items
        super().__init__(max_size, reducers=reducers, ctx=ctx)

    def _on_queue_feeder_error(self, e, obj):
        if not isinstance(obj, _CallItem):
            super()._on_queue_feeder_error(e, obj)
            return
        # work_item can be None if another process terminated. In this
        # case, the executor_manager_thread fails all work_items with
        # BrokenProcessPool
        work_item = self.pending_work_items.pop(obj.work_id, None)
        try:
            # format traceback only works on python3
            if isinstance(e, struct.error):
                raised_error = RuntimeError(
                    "The task could not be sent to the workers as it is too "
                    "large for `send_bytes`."
                )
            else:
                raised_error = PicklingError(
                    "Could not pickle the task to send it to the workers."
                )
            tb = traceback.format_exception(
                type(e), e, getattr(e, "__traceback__", None)
            )
            raised_error.__cause__ = _RemoteTraceback("".join(tb))
            self.running_work_items.remove(obj.work_id)
        except BaseException as hook_exc:
            # An error here would kill the feeder thread silently and leave
            # every later task unsent: flag the executor as broken instead
            raised_error = BrokenProcessPool(
                "The call queue feeder thread crashed: the executor is "
                "broken and the pending tasks have been cancelled."
            )
            tb = traceback.format_exception(
                type(hook_exc), hook_exc, hook_exc.__traceback__
            )
            raised_error.__cause__ = _RemoteTraceback("".join(tb))
            LOGGER.critical(
                "Exception in call queue feeder error hook:", exc_info=True
            )
            self.executor_flags.flag_as_broken(raised_error)
        if work_item is not None:
            if isinstance(work_item, _StreamWorkItem):
                # A stream that could not even be sent to a worker fails on
                # its own, without involving any future.
                work_item.stream._fail(raised_error)
                if work_item.stream_conn is not None:
                    try:
                        work_item.stream_conn.close()
                    except OSError:
                        pass
            else:
                work_item.future.set_exception(raised_error)
            del work_item
        with self.shutdown_lock:
            self.thread_wakeup.wakeup()


def _get_chunks(chunksize, *iterables):
    """Iterates over zip()ed iterables in chunks."""
    it = zip(*iterables)
    while True:
        chunk = tuple(itertools.islice(it, chunksize))
        if not chunk:
            return
        yield chunk


def _process_chunk(fn, chunk):
    """Processes a chunk of an iterable passed to map.

    Runs the function passed to map() on a chunk of the
    iterable passed to map.

    This function is run in a separate process.

    """
    return [fn(*args) for args in chunk]


def _sendback_result(result_queue, work_id, result=None, exception=None):
    """Safely send back the given result or exception"""
    try:
        result_queue.put(
            _ResultItem(work_id, result=result, exception=exception)
        )
    except BaseException as e:
        exc = _ExceptionWithTraceback(e)
        result_queue.put(_ResultItem(work_id, exception=exc))


def _safe_stream_exception(exc):
    """Build a picklable exception-with-traceback for a failing stream.

    Falls back to a plain ``RuntimeError`` if the original exception cannot
    itself be serialized, so that a terminal message can always be sent.
    """
    try:
        return _ExceptionWithTraceback(exc)
    except BaseException:
        fallback = RuntimeError(
            f"Stream failed with a non-picklable "
            f"{type(exc).__name__}: {exc}"
        )
        return _ExceptionWithTraceback(fallback)


def _run_stream_call(call_item, result_queue):
    """Drive a streaming generator inside a worker process.

    The generator is advanced one ``next`` at a time. Each yielded item is
    sent individually on the shared result queue wrapped in a ``_StreamItem``
    carrying the stream identity and a monotonic sequence number. The worker
    only advances the generator while fewer than ``max_buffered_items`` items
    are waiting for acknowledgement from the consumer: once the window is
    full it blocks on the per-stream control pipe until an acknowledgement or
    a close request arrives.

    A close request is honoured at the next cooperative yield boundary by
    calling ``generator.close()``, letting ``finally`` clauses run; items not
    delivered yet are discarded.
    """
    work_id = call_item.work_id
    conn = call_item.stream_conn
    window = call_item.max_buffered_items

    in_flight = 0
    seq = 0
    closing = False
    terminal_sent = False

    def send_terminal(message):
        # A terminal message must reach the parent whenever the worker is
        # still able to write to the result channel. Return whether it was
        # actually sent: callers need a picklable fallback when the preferred
        # terminal message cannot itself be serialized. If sending any
        # terminal fails, the parent's worker-exit detection still recovers
        # the stream as a last resort.
        nonlocal terminal_sent
        try:
            result_queue.put(message)
            terminal_sent = True
            return True
        except BaseException:
            return False

    def drain_control():
        # Consume every pending acknowledgement / close request without
        # blocking. Returns False if the control pipe is broken.
        nonlocal in_flight, closing
        while conn.poll():
            try:
                message = conn.recv()
            except (EOFError, OSError):
                closing = True
                return False
            if message == _STREAM_CLOSE:
                closing = True
            else:
                in_flight -= 1
        return True

    try:
        try:
            generator = call_item()
        except BaseException as e:
            send_terminal(_StreamError(work_id, _safe_stream_exception(e)))
            return
        try:
            while True:
                # Pick up acknowledgements / close requests accumulated while
                # the generator was producing its previous item.
                if not drain_control():
                    break
                if closing:
                    break

                if in_flight >= window:
                    # Backpressure: wait for the consumer to take an item (or
                    # for a close request) before advancing the generator.
                    try:
                        message = conn.recv()
                    except (EOFError, OSError):
                        closing = True
                        break
                    if message == _STREAM_CLOSE:
                        closing = True
                        break
                    in_flight -= 1
                    continue

                try:
                    item = next(generator)
                except StopIteration as e:
                    try:
                        # The generator return value travels on the terminal
                        # message. Attempt it directly so that a non-picklable
                        # return value can be reported as a stream error
                        # instead of silently dropping the terminal message.
                        result_queue.put(
                            _StreamEnd(work_id, _STREAM_COMPLETE, e.value)
                        )
                        terminal_sent = True
                    except BaseException as send_error:
                        send_terminal(
                            _StreamError(
                                work_id,
                                _safe_stream_exception(send_error),
                            )
                        )
                    return
                except BaseException as e:
                    # The generator raised: report the failure after the
                    # items already sent, keeping the remote traceback.
                    send_terminal(
                        _StreamError(work_id, _safe_stream_exception(e))
                    )
                    return

                # A close request may have arrived while ``next`` was running
                # in user code. It cannot interrupt that call, but is handled
                # at this cooperative boundary before sending the new item.
                if not drain_control() or closing:
                    break

                try:
                    # SimpleQueue serializes the message before writing any
                    # byte to the pipe: if the item cannot be pickled, the
                    # shared result channel is left untouched and the failure
                    # only terminates this stream.
                    result_queue.put(_StreamItem(work_id, seq, item))
                except BaseException as e:
                    send_terminal(
                        _StreamError(work_id, _safe_stream_exception(e))
                    )
                    return
                seq += 1
                in_flight += 1
        finally:
            if closing:
                # Cooperative close at a yield boundary: throw GeneratorExit
                # into the generator so that its ``finally`` clauses run, then
                # discard every undelivered item and release the resources.
                try:
                    generator.close()
                except BaseException as e:
                    if not terminal_sent:
                        send_terminal(
                            _StreamError(work_id, _safe_stream_exception(e))
                        )
                else:
                    if not terminal_sent:
                        send_terminal(
                            _StreamEnd(work_id, _STREAM_CLOSED, None)
                        )
            elif not terminal_sent:
                # Control pipe broken while the parent is gone: make sure the
                # generator is still finalized even without a consumer.
                try:
                    generator.close()
                except BaseException:
                    pass
    finally:
        try:
            conn.close()
        except OSError:
            pass


def _enable_faulthandler_if_needed():
    if "PYTHONFAULTHANDLER" in os.environ:
        # Respect the environment variable to configure faulthandler. This
        # makes it possible to never enable faulthandler in the loky workers by
        # setting PYTHONFAULTHANDLER=0 explicitly in the environment.
        mp.util.debug(
            f"faulthandler explicitly configured by environment variable: "
            f"PYTHONFAULTHANDLER={os.environ['PYTHONFAULTHANDLER']}."
        )
    else:
        if faulthandler.is_enabled():
            # Fault handler is already enabled, possibly via a custom
            # initializer to customize the behavior.
            mp.util.debug("faulthandler already enabled.")
        else:
            # Enable faulthandler by default with default paramaters otherwise.
            mp.util.debug(
                "Enabling faulthandler to report tracebacks on worker crashes."
            )
            faulthandler.enable()


def _process_worker(
    call_queue,
    result_queue,
    initializer,
    initargs,
    processes_management_lock,
    timeout,
    worker_exit_lock,
    current_depth,
):
    """Evaluates calls from call_queue and places the results in result_queue.

    This worker is run in a separate process.

    Args:
        call_queue: A ctx.Queue of _CallItems that will be read and
            evaluated by the worker.
        result_queue: A ctx.Queue of _ResultItems that will written
            to by the worker.
        initializer: A callable initializer, or None
        initargs: A tuple of args for the initializer
        processes_management_lock: A ctx.Lock avoiding worker timeout while
            some workers are being spawned.
        timeout: maximum time to wait for a new item in the call_queue. If that
            time is expired, the worker will shutdown.
        worker_exit_lock: Lock to avoid flagging the executor as broken on
            workers timeout.
        current_depth: Nested parallelism level, to avoid infinite spawning.
    """
    if initializer is not None:
        try:
            initializer(*initargs)
        except BaseException:
            LOGGER.critical("Exception in initializer:", exc_info=True)
            # The parent will notice that the process stopped and
            # mark the pool broken
            return

    # set the global _CURRENT_DEPTH mechanism to limit recursive call
    global _CURRENT_DEPTH
    _CURRENT_DEPTH = current_depth
    _process_reference_size = None
    _last_memory_leak_check = None
    pid = os.getpid()

    mp.util.debug(f"Worker started with timeout={timeout}")
    _enable_faulthandler_if_needed()

    while True:
        try:
            call_item = call_queue.get(block=True, timeout=timeout)
            if call_item is None:
                mp.util.info("Shutting down worker on sentinel")
        except queue.Empty:
            mp.util.info(f"Shutting down worker after timeout {timeout:0.3f}s")
            if processes_management_lock.acquire(block=False):
                processes_management_lock.release()
                call_item = None
            else:
                mp.util.info("Could not acquire processes_management_lock")
                continue
        except BaseException:
            previous_tb = traceback.format_exc()
            try:
                result_queue.put(_RemoteTraceback(previous_tb))
            except BaseException:
                # If we cannot format correctly the exception, at least print
                # the traceback.
                print(previous_tb)
            mp.util.debug("Exiting with code 1")
            sys.exit(1)
        if call_item is None:
            # Notify queue management thread about worker shutdown
            result_queue.put(pid)
            is_clean = worker_exit_lock.acquire(True, timeout=30)

            # Early notify any loky executor running in this worker process
            # (nested parallelism) that this process is about to shutdown to
            # avoid a deadlock waiting undifinitely for the worker to finish.
            _python_exit()

            if is_clean:
                mp.util.debug("Exited cleanly")
            else:
                mp.util.info("Main process did not release worker_exit")
            return
        if isinstance(call_item, _StreamCallItem):
            # Streaming calls send their own _StreamItem / terminal messages
            # on the result queue and apply backpressure internally.
            _run_stream_call(call_item, result_queue)
        else:
            try:
                r = call_item()
            except BaseException as e:
                exc = _ExceptionWithTraceback(e)
                result_queue.put(_ResultItem(call_item.work_id, exception=exc))
            else:
                _sendback_result(result_queue, call_item.work_id, result=r)
                del r

        # Free the resource as soon as possible, to avoid holding onto
        # open files or shared memory that is not needed anymore
        del call_item

        if _USE_PSUTIL:
            if _process_reference_size is None:
                # Make reference measurement after the first call
                _process_reference_size = _get_memory_usage(pid, force_gc=True)
                _last_memory_leak_check = time()
                continue
            if time() - _last_memory_leak_check > _MEMORY_LEAK_CHECK_DELAY:
                mem_usage = _get_memory_usage(pid)
                _last_memory_leak_check = time()
                if mem_usage - _process_reference_size < _MAX_MEMORY_LEAK_SIZE:
                    # Memory usage stays within bounds: everything is fine.
                    continue

                # Check again memory usage; this time take the measurement
                # after a forced garbage collection to break any reference
                # cycles.
                mem_usage = _get_memory_usage(pid, force_gc=True)
                _last_memory_leak_check = time()
                if mem_usage - _process_reference_size < _MAX_MEMORY_LEAK_SIZE:
                    # The GC managed to free the memory: everything is fine.
                    continue

                # The process is leaking memory: let the master process
                # know that we need to start a new worker.
                mp.util.info("Memory leak detected: shutting down worker")
                result_queue.put(_RecycledWorkerPid(pid))
                with worker_exit_lock:
                    mp.util.debug("Exit due to memory leak")
                    return
        else:
            # if psutil is not installed, trigger gc.collect events
            # regularly to limit potential memory leaks due to reference cycles
            if _last_memory_leak_check is None or (
                time() - _last_memory_leak_check > _MEMORY_LEAK_CHECK_DELAY
            ):
                gc.collect()
                _last_memory_leak_check = time()


class _ExecutorManagerThread(threading.Thread):
    """Manages the communication between this process and the worker processes.

    The manager is run in a local thread.

    Args:
        executor: A reference to the ProcessPoolExecutor that owns
            this thread. A weakref will be own by the manager as well as
            references to internal objects used to introspect the state of
            the executor.
    """

    def __init__(self, executor):
        # Store references to necessary internals of the executor.

        # A _ThreadWakeup to allow waking up the executor_manager_thread from
        # the main Thread and avoid deadlocks caused by permanently
        # locked queues.
        self.thread_wakeup = executor._executor_manager_thread_wakeup
        self.shutdown_lock = executor._shutdown_lock

        # A weakref.ref to the ProcessPoolExecutor that owns this thread. Used
        # to determine if the ProcessPoolExecutor has been garbage collected
        # and that the manager can exit.
        # When the executor gets garbage collected, the weakref callback
        # will wake up the queue management thread so that it can terminate
        # if there is no pending work item.
        def weakref_cb(
            _,
            thread_wakeup=self.thread_wakeup,
            shutdown_lock=self.shutdown_lock,
        ):
            if mp is not None:
                # At this point, the multiprocessing module can already be
                # garbage collected. We only log debug info when still
                # possible.
                mp.util.debug(
                    "Executor collected: triggering callback for"
                    " QueueManager wakeup"
                )
            with shutdown_lock:
                thread_wakeup.wakeup()

        self.executor_reference = weakref.ref(executor, weakref_cb)

        # The flags of the executor
        self.executor_flags = executor._flags

        # A list of the ctx.Process instances used as workers.
        self.processes = executor._processes

        # A ctx.Queue that will be filled with _CallItems derived from
        # _WorkItems for processing by the process workers.
        self.call_queue = executor._call_queue

        # A ctx.SimpleQueue of _ResultItems generated by the process workers.
        self.result_queue = executor._result_queue

        # A queue.Queue of work ids e.g. Queue([5, 6, ...]).
        self.work_ids_queue = executor._work_ids

        # A dict mapping work ids to _WorkItems e.g.
        #     {5: <_WorkItem...>, 6: <_WorkItem...>, ...}
        self.pending_work_items = executor._pending_work_items

        # A list of the work_ids that are currently running
        self.running_work_items = executor._running_work_items

        # A lock to avoid concurrent shutdown of workers on timeout and spawn
        # of new processes or shut down
        self.processes_management_lock = executor._processes_management_lock

        # Long-running executors can recycle workers many times over their
        # lifetime; the user only needs to hear about it once.
        self.recycling_warned = False

        super().__init__(name="ExecutorManagerThread")
        if sys.version_info < (3, 9):
            self.daemon = True

    def run(self):
        # Main loop for the executor manager thread.
        try:
            self._run_loop()
        except BaseException as exc:
            # Nothing in the loop is expected to raise, so this is a bug, a
            # warning filter turning one of the warnings below into an error,
            # or a failure of the runtime itself, such as a queue that cannot
            # start its feeder thread while the interpreter is shutting down
            # (python/cpython#109047). Letting the exception kill this thread
            # would leave every pending future unresolved and the executor
            # looking healthy, so a caller waiting on a result would hang
            # forever.
            # Flag the executor as broken instead, with the error as cause.
            bpe = BrokenProcessPool(
                "The executor manager thread crashed: the executor is broken "
                "and the pending tasks have been cancelled."
            )
            tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
            bpe.__cause__ = _RemoteTraceback("".join(tb))
            # Also log it: with no pending future, nothing else reports it
            LOGGER.critical(
                "Exception in executor manager thread:", exc_info=True
            )
            self.terminate_broken(bpe)

    def _run_loop(self):
        while True:
            self.add_call_item_to_queue()

            result_item, is_broken, bpe = self.wait_result_broken_or_wakeup()

            # The call queue feeder thread flags the executor broken and
            # wakes us up when it cannot report a task failure itself
            if not is_broken and self.executor_flags.broken is not None:
                is_broken, bpe = True, self.executor_flags.broken
            if is_broken:
                self.terminate_broken(bpe)
                return
            if result_item is not None:
                self.process_result_item(result_item)
                # Delete reference to result_item to avoid keeping references
                # while waiting on new results.
                del result_item

            if self.is_shutting_down():
                self.flag_executor_shutting_down()

                # When only canceled futures remain in pending_work_items, our
                # next call to wait_result_broken_or_wakeup would hang forever.
                # This makes sure we have some running futures or none at all.
                self.add_call_item_to_queue()

                # Since no new work items can be added, it is safe to shutdown
                # this thread if there are no pending work items.
                if not self.pending_work_items:
                    self.join_executor_internals()
                    return

    def add_call_item_to_queue(self):
        # Fills call_queue with _WorkItems from pending_work_items.
        # This function never blocks.
        while True:
            if self.call_queue.full():
                return
            try:
                work_id = self.work_ids_queue.get(block=False)
            except queue.Empty:
                return
            else:
                work_item = self.pending_work_items[work_id]

                if isinstance(work_item, _StreamWorkItem):
                    # Streams cannot be cancelled: always dispatch them,
                    # carrying the per-stream control pipe end and window.
                    self.running_work_items += [work_id]
                    self.call_queue.put(
                        _StreamCallItem(
                            work_id,
                            work_item.fn,
                            work_item.args,
                            work_item.kwargs,
                            work_item.max_buffered_items,
                            work_item.stream_conn,
                        ),
                        block=True,
                    )
                elif work_item.future.set_running_or_notify_cancel():
                    self.running_work_items += [work_id]
                    self.call_queue.put(
                        _CallItem(
                            work_id,
                            work_item.fn,
                            work_item.args,
                            work_item.kwargs,
                        ),
                        block=True,
                    )
                else:
                    del self.pending_work_items[work_id]
                    continue

    def wait_result_broken_or_wakeup(self):
        # Wait for a result to be ready in the result_queue while checking
        # that all worker processes are still running, or for a wake up
        # signal send. The wake up signals come either from new tasks being
        # submitted, from the executor being shutdown/gc-ed, or from the
        # shutdown of the python interpreter.
        result_reader = self.result_queue._reader
        wakeup_reader = self.thread_wakeup._reader
        readers = [result_reader, wakeup_reader]
        worker_sentinels = [p.sentinel for p in list(self.processes.values())]
        ready = wait(readers + worker_sentinels)

        bpe = None
        is_broken = True
        result_item = None
        if result_reader in ready:
            try:
                result_item = result_reader.recv()
                if isinstance(result_item, _RemoteTraceback):
                    bpe = BrokenProcessPool(
                        "A task has failed to un-serialize. Please ensure that"
                        " the arguments of the function are all picklable."
                    )
                    bpe.__cause__ = result_item
                else:
                    is_broken = False
            except BaseException as e:
                bpe = BrokenProcessPool(
                    "A result has failed to un-serialize. Please ensure that "
                    "the objects returned by the function are always "
                    "picklable."
                )
                tb = traceback.format_exception(
                    type(e), e, getattr(e, "__traceback__", None)
                )
                bpe.__cause__ = _RemoteTraceback("".join(tb))

        elif wakeup_reader in ready:
            # This is simply a wake-up event that might either trigger putting
            # more tasks in the queue or trigger the clean up of resources.
            is_broken = False
        else:
            # A worker has terminated and we don't know why, set the state of
            # the executor as broken
            exit_codes = ""
            if sys.platform != "win32":
                # In Windows, introspecting terminated workers exitcodes seems
                # unstable, therefore they are not appended in the exception
                # message.
                exit_codes = (
                    "\nThe exit codes of the workers are "
                    f"{get_exitcodes_terminated_worker(self.processes)}"
                )
            mp.util.debug(
                "A worker unexpectedly terminated. Workers that "
                "might have caused the breakage: "
                + str(
                    {
                        p.name: p.exitcode
                        for p in list(self.processes.values())
                        if p is not None and p.sentinel in ready
                    }
                )
            )
            bpe = TerminatedWorkerError(
                "A worker process managed by the executor was unexpectedly "
                "terminated. This could be caused by a segmentation fault "
                "while calling the function or by an excessive memory usage "
                "causing the Operating System to kill the worker.\n"
                f"{exit_codes}\n"
                "Detailed tracebacks of the workers should have been printed "
                "to stderr in the executor process if faulthandler was not "
                "disabled."
            )

        self.thread_wakeup.clear()

        return result_item, is_broken, bpe

    def process_result_item(self, result_item):
        # Process the received a result_item. This can be either the PID of a
        # worker that exited gracefully, a _ResultItem or a stream message.

        if isinstance(result_item, _StreamMessage):
            self._process_stream_message(result_item)
            return

        if isinstance(result_item, int):
            # Clean shutdown of a worker using its PID, either on request
            # by the executor.shutdown method or by the timeout of the worker
            # itself: we should not mark the executor as broken.
            with self.processes_management_lock:
                p = self.processes.pop(result_item, None)

            # p can be None if the executor is concurrently shutting down.
            if p is not None:
                p._worker_exit_lock.release()
                mp.util.debug(
                    f"joining {p.name} when processing {p.pid} as result_item"
                )
                p.join()
                del p

            # Make sure the executor have the right number of worker, even if a
            # worker timeout while some jobs were submitted. If some work is
            # pending or there is less processes than running items, we need to
            # start a new Process and raise a warning.
            n_pending = len(self.pending_work_items)
            n_running = len(self.running_work_items)
            if n_pending - n_running > 0 or n_running > len(self.processes):
                executor = self.executor_reference()
                if (
                    executor is not None
                    and len(self.processes) < executor._max_workers
                ):
                    with executor._processes_management_lock:
                        executor._adjust_process_count()
                    executor = None
                    # Warn only once the pool is back to full strength, so
                    # that a filter turning this into an error costs no worker
                    if not isinstance(result_item, _RecycledWorkerPid):
                        warnings.warn(
                            "A worker stopped while some jobs were given to "
                            "the executor. This can be caused by a too short "
                            "worker timeout or by a memory leak triggered by the "
                            "OS-level out-of-memory (OOM) process killer.",
                            UserWarning,
                        )
                    elif not self.recycling_warned:
                        # Recycling is a deliberate decision of the executor and
                        # it is transparent to the caller, so repeating it for
                        # every worker of a long-running executor is just noise.
                        self.recycling_warned = True
                        warnings.warn(
                            "A worker was restarted while some jobs were given "
                            "to the executor because it was using more than "
                            f"{_MAX_MEMORY_LEAK_SIZE / 1e6:.0f} MB more memory "
                            "than when it started, which usually indicates a "
                            "memory leak. Further restarts of this executor "
                            "will not be reported.",
                            UserWarning,
                        )
        else:
            # Received a _ResultItem so mark the future as completed.
            work_item = self.pending_work_items.pop(result_item.work_id, None)
            # work_item can be None if another process terminated (see above)
            if work_item is not None:
                if result_item.exception is not None:
                    work_item.future.set_exception(result_item.exception)
                else:
                    work_item.future.set_result(result_item.result)
                self.running_work_items.remove(result_item.work_id)

    def _finish_stream_work_item(self, work_id):
        # Remove a terminated stream from the bookkeeping structures and
        # release the parent-side end of its control pipe.
        work_item = self.pending_work_items.pop(work_id, None)
        if work_item is not None:
            if work_item.stream_conn is not None:
                try:
                    work_item.stream_conn.close()
                except OSError:
                    pass
        try:
            self.running_work_items.remove(work_id)
        except ValueError:
            pass

    def _process_stream_message(self, message):
        # Route a worker -> parent stream message to its StreamResult. The
        # manager thread never blocks here: a full stream buffer is made
        # impossible by the worker-side backpressure.
        work_item = self.pending_work_items.get(message.work_id)
        # The stream may already have been failed by a broken-pool shutdown.
        if work_item is None:
            return
        stream = work_item.stream

        if isinstance(message, _StreamItem):
            try:
                stream._deliver_item(message.seq, message.item)
            except queue.Full:
                # Cannot happen under the backpressure protocol: fail the
                # stream instead of stalling the whole manager thread.
                self._finish_stream_work_item(message.work_id)
                stream._fail(
                    RuntimeError(
                        "loky internal error: stream buffer is full while "
                        "the worker is still producing items"
                    )
                )
            return

        # Clean up the bookkeeping first so that, by the time the stream is
        # marked done and consumers are woken up, no stale running/pending
        # entry can be observed.
        self._finish_stream_work_item(message.work_id)
        if isinstance(message, _StreamEnd):
            if message.kind == _STREAM_COMPLETE:
                stream._finish(
                    StreamResult.COMPLETE,
                    return_value=message.result,
                )
            else:
                # Cooperative close: undelivered items are discarded.
                stream._finish(StreamResult.CLOSED, discard=True)
        else:
            # Generator or item-serialization failure: keep the items already
            # delivered so the consumer drains them before the exception.
            stream._finish(StreamResult.ERROR, exception=message.exception)

    def is_shutting_down(self):
        # Check whether we should start shutting down the executor.
        executor = self.executor_reference()
        # No more work items can be added if:
        #   - The interpreter is shutting down OR
        #   - The executor that owns this thread is not broken AND
        #        * The executor that owns this worker has been collected OR
        #        * The executor that owns this worker has been shutdown.
        # If the executor is broken, it should be detected in the next loop.
        return _global_shutdown or (
            (executor is None or self.executor_flags.shutdown)
            and not self.executor_flags.broken
        )

    def terminate_broken(self, bpe):
        # Terminate the executor because it is in a broken state. The bpe
        # argument can be used to display more information on the error that
        # lead the executor into becoming broken.

        # Mark the process pool broken so that submits fail right now.
        self.executor_flags.flag_as_broken(bpe)

        # Mark pending tasks as failed.
        for work_item in list(self.pending_work_items.values()):
            if isinstance(work_item, _StreamWorkItem):
                # Wake up any stream consumer with the broken-pool error and
                # release its control pipe.
                work_item.stream._fail(bpe)
                if work_item.stream_conn is not None:
                    try:
                        work_item.stream_conn.close()
                    except OSError:
                        pass
            else:
                try:
                    work_item.future.set_exception(bpe)
                except InvalidStateError:
                    # set_exception() fails if the future is cancelled:
                    # ignore it. Trying to check if the future is cancelled
                    # before calling set_exception() would leave a race
                    # condition if the future is cancelled between the check
                    # and set_exception().
                    pass
            # Delete references to object. See issue16284
            del work_item
        self.pending_work_items.clear()

        # Terminate remaining workers forcibly: the queues or their
        # locks may be in a dirty state and block forever.
        self.kill_workers(reason="broken executor")

        # clean up resources
        self.join_executor_internals()

    def flag_executor_shutting_down(self):
        # Flag the executor as shutting down and cancel remaining tasks if
        # requested as early as possible if it is not gc-ed yet.
        self.executor_flags.flag_as_shutting_down()

        # Cooperative shutdown: ask every still-open stream to close. The
        # workers finalize their generators at the next yield boundary and
        # send their terminal message, so the executor can shut down without
        # requiring the callers to keep consuming the streams. close() is
        # idempotent so running this on every manager loop is harmless.
        if not self.executor_flags.kill_workers:
            for work_item in list(self.pending_work_items.values()):
                if isinstance(work_item, _StreamWorkItem):
                    work_item.stream.close()
            return

        # Cancel pending work items if requested.
        # Drain the queued ids first: add_call_item_to_queue runs again
        # right after this and looks each of them up in pending_work_items
        while True:
            try:
                self.work_ids_queue.get(block=False)
            except queue.Empty:
                break
        shutdown_error = ShutdownExecutorError(
            "The Executor was shutdown with `kill_workers=True` "
            "before this job could complete."
        )
        while self.pending_work_items:
            _, work_item = self.pending_work_items.popitem()
            if isinstance(work_item, _StreamWorkItem):
                work_item.stream._fail(shutdown_error)
                if work_item.stream_conn is not None:
                    try:
                        work_item.stream_conn.close()
                    except OSError:
                        pass
            else:
                work_item.future.set_exception(shutdown_error)
            del work_item

        # Kill the remaining worker forcibly to no waste time joining them
        self.kill_workers(reason="executor shutting down")

    def kill_workers(self, reason=""):
        # Terminate the remaining workers using SIGKILL. This function also
        # terminates descendant workers of the children in case there is some
        # nested parallelism.
        while self.processes:
            _, p = self.processes.popitem()
            mp.util.debug(f"terminate process {p.name}, reason: {reason}")
            try:
                kill_process_tree(p)
            except ProcessLookupError:  # pragma: no cover
                pass

    def shutdown_workers(self):
        # shutdown all workers in self.processes

        # Create a list to avoid RuntimeError due to concurrent modification of
        # processes. nb_children_alive is thus an upper bound. Also release the
        # processes' _worker_exit_lock to accelerate the shutdown procedure, as
        # there is no need for hand-shake here.
        with self.processes_management_lock:
            n_children_to_stop = 0
            for p in list(self.processes.values()):
                mp.util.debug(f"releasing worker exit lock on {p.name}")
                p._worker_exit_lock.release()
                n_children_to_stop += 1

        mp.util.debug(f"found {n_children_to_stop} processes to stop")

        # Send the right number of sentinels, to make sure all children are
        # properly terminated. Do it with a mechanism that avoid hanging on
        # Full queue when all workers have already been shutdown.
        n_sentinels_sent = 0
        cooldown_time = 0.001
        while (
            n_sentinels_sent < n_children_to_stop
            and self.get_n_children_alive() > 0
        ):
            for _ in range(n_children_to_stop - n_sentinels_sent):
                try:
                    self.call_queue.put_nowait(None)
                    n_sentinels_sent += 1
                except queue.Full as e:
                    if cooldown_time > 5.0:
                        mp.util.info(
                            "failed to send all sentinels and exit with error."
                            f"\ncall_queue size={self.call_queue._maxsize}; "
                            f" full is {self.call_queue.full()}; "
                        )
                        raise e
                    mp.util.info(
                        "full call_queue prevented to send all sentinels at "
                        "once, waiting..."
                    )
                    sleep(cooldown_time)
                    cooldown_time *= 1.2
                    break

        mp.util.debug(f"sent {n_sentinels_sent} sentinels to the call queue")

    def join_executor_internals(self):
        self.shutdown_workers()

        # Release the queue's resources as soon as possible. Flag the feeder
        # thread for clean exit to avoid having the crash detection thread flag
        # the Executor as broken during the shutdown. This is safe as either:
        #  * We don't need to communicate with the workers anymore
        #  * There is nothing left in the Queue buffer except None sentinels
        mp.util.debug("closing call_queue")
        self.call_queue.close()
        self.call_queue.join_thread()

        # Closing result_queue
        mp.util.debug("closing result_queue")
        self.result_queue.close()

        mp.util.debug("closing thread_wakeup")
        with self.shutdown_lock:
            self.thread_wakeup.close()

        # If .join() is not called on the created processes then
        # some ctx.Queue methods may deadlock on macOS.
        with self.processes_management_lock:
            mp.util.debug(f"joining {len(self.processes)} processes")
            n_joined_processes = 0
            while True:
                try:
                    pid, p = self.processes.popitem()
                    mp.util.debug(f"joining process {p.name} with pid {pid}")
                    p.join()
                    n_joined_processes += 1
                except KeyError:
                    break

            mp.util.debug(
                "executor management thread clean shutdown of "
                f"{n_joined_processes} workers"
            )

    def get_n_children_alive(self):
        # This is an upper bound on the number of children alive.
        with self.processes_management_lock:
            return sum(p.is_alive() for p in list(self.processes.values()))


_system_limits_checked = False
_system_limited = None


def _check_system_limits():
    global _system_limits_checked, _system_limited
    if _system_limits_checked and _system_limited:
        raise NotImplementedError(_system_limited)
    _system_limits_checked = True
    try:
        nsems_max = os.sysconf("SC_SEM_NSEMS_MAX")
    except (AttributeError, ValueError):
        # sysconf not available or setting not available
        return
    if nsems_max == -1:
        # undetermined limit, assume that limit is determined
        # by available memory only
        return
    if nsems_max >= 256:
        # minimum number of semaphores available
        # according to POSIX
        return
    _system_limited = (
        f"system provides too few semaphores ({nsems_max} available, "
        "256 necessary)"
    )
    raise NotImplementedError(_system_limited)


def _chain_from_iterable_of_lists(iterable):
    """
    Specialized implementation of itertools.chain.from_iterable.
    Each item in *iterable* should be a list.  This function is
    careful not to keep references to yielded objects.
    """
    for element in iterable:
        element.reverse()
        while element:
            yield element.pop()


def _check_max_depth(context):
    # Limit the maxmal recursion level
    global _CURRENT_DEPTH
    if context.get_start_method() == "fork" and _CURRENT_DEPTH > 0:
        raise LokyRecursionError(
            "Could not spawn extra nested processes at depth superior to "
            "MAX_DEPTH=1. It is not possible to increase this limit when "
            "using the 'fork' start method."
        )

    if 0 < MAX_DEPTH and _CURRENT_DEPTH + 1 > MAX_DEPTH:
        raise LokyRecursionError(
            "Could not spawn extra nested processes at depth superior to "
            f"MAX_DEPTH={MAX_DEPTH}. If this is intendend, you can change "
            "this limit with the LOKY_MAX_DEPTH environment variable."
        )


class LokyRecursionError(RuntimeError):
    """A process tries to spawn too many levels of nested processes."""


class BrokenProcessPool(_BPPException):
    """
    Raised when the executor is broken while a future was in the running state.
    The cause can an error raised when unpickling the task in the worker
    process or when unpickling the result value in the parent process. It can
    also be caused by a worker process being terminated unexpectedly.
    """


class TerminatedWorkerError(BrokenProcessPool):
    """
    Raised when a process in a ProcessPoolExecutor terminated abruptly
    while a future was in the running state.
    """


# Alias for backward compat (for code written for loky 1.1.4 and earlier). Do
# not use in new code.
BrokenExecutor = BrokenProcessPool


class ShutdownExecutorError(RuntimeError):
    """
    Raised when a ProcessPoolExecutor is shutdown while a future was in the
    running or pending state.
    """


class StreamResult:
    """Handle of a streaming task submitted with ``submit_stream``.

    A ``StreamResult`` is a synchronous, single-consumer iterator over the
    items yielded by the generator running in the worker process. Items are
    delivered as soon as they are produced, without waiting for the generator
    to finish. It additionally supports an explicit :meth:`close` and exposes
    the final state of the stream once it has terminated.

    The parent-side buffer is bounded by ``max_buffered_items``: the worker
    can only get that many yielded items ahead of the consumer and blocks on
    backpressure acknowledgements beyond this window.
    """

    # Stream states.
    OPEN = "open"
    COMPLETE = "complete"
    CLOSED = "closed"
    ERROR = "error"
    BROKEN = "broken"

    def __init__(self, work_id, conn, max_buffered_items):
        self._work_id = work_id
        # Write end of the per-stream control pipe. ``None`` once closed.
        self._conn = conn
        self._io_lock = threading.Lock()
        self._cond = threading.Condition()
        # Bounded parent-side buffer of ``(seq, item)`` tuples. Its capacity
        # matches the worker backpressure window, so it can never fill beyond
        # capacity as long as one acknowledgement is sent per retrieved item.
        self._buffer = queue.Queue(maxsize=max_buffered_items)
        self._consumer_thread = None
        self._close_requested = False
        self._done = False
        self._state = self.OPEN
        self._return_value = None
        self._exception = None

    # ------------------------------------------------------------------
    # Consumer-facing API
    # ------------------------------------------------------------------
    def __iter__(self):
        self._register_consumer()
        return self

    def __next__(self):
        self._register_consumer()
        with self._cond:
            while self._buffer.empty() and not self._done:
                self._cond.wait()
            if not self._buffer.empty():
                _seq, item = self._buffer.get_nowait()
            else:
                # Terminal state with nothing left to deliver. Errors are
                # raised only after every previously sent item has been
                # consumed.
                if self._exception is not None:
                    raise self._exception
                raise StopIteration
        # Acknowledge consumption to release one backpressure slot in the
        # worker. This is done outside the condition lock.
        self._send_control(_STREAM_ACK)
        return item

    def close(self):
        """Request the worker to stop the stream cooperatively.

        The request wakes up a worker blocked on backpressure and, at the
        next yield boundary, calls ``close`` on the remote generator so that
        its ``finally`` clauses run. Items not yet delivered are discarded.
        Calling :meth:`close` does not interrupt a generator currently
        executing inside a ``next`` call; it is honoured when that call
        returns. This method is idempotent and thread-safe.
        """
        with self._io_lock:
            if self._close_requested:
                return
            self._close_requested = True
            self._send_control_locked(_STREAM_CLOSE)

    def wait(self, timeout=None):
        """Block until the stream reaches a terminal state.

        Returns ``True`` if the stream is done and ``False`` if the timeout
        expired first.
        """
        with self._cond:
            if not self._done:
                self._cond.wait(timeout)
            return self._done

    @property
    def done(self):
        """Whether the stream has reached a terminal state."""
        with self._cond:
            return self._done

    @property
    def state(self):
        """Current stream state.

        One of ``"open"``, ``"complete"``, ``"closed"``, ``"error"`` or
        ``"broken"``.
        """
        with self._cond:
            return self._state

    @property
    def return_value(self):
        """The generator return value, available once the stream is done.

        Raises :class:`RuntimeError` if the stream is still running. Returns
        ``None`` when the stream was closed or failed.
        """
        with self._cond:
            if not self._done:
                raise RuntimeError(
                    "the stream has not completed yet; call wait() or "
                    "consume it before reading its return value"
                )
            return self._return_value

    def exception(self):
        """Return the exception that terminated the stream, if any.

        Raises :class:`RuntimeError` if the stream is still running. Returns
        ``None`` for a completed or cooperatively closed stream.
        """
        with self._cond:
            if not self._done:
                raise RuntimeError(
                    "the stream has not completed yet; call wait() or "
                    "consume it before reading its exception"
                )
            return self._exception

    # ------------------------------------------------------------------
    # Manager-thread API (not part of the public surface)
    # ------------------------------------------------------------------
    def _register_consumer(self):
        with self._cond:
            ident = threading.get_ident()
            if self._consumer_thread is None:
                self._consumer_thread = ident
            elif self._consumer_thread != ident:
                raise RuntimeError(
                    "a StreamResult supports a single consumer; it is "
                    "already being iterated from another thread"
                )

    def _send_control_locked(self, message):
        if self._conn is None:
            return
        try:
            self._conn.send(message)
        except (BrokenPipeError, EOFError, OSError):
            # The worker is gone or already shutting down: nothing to do,
            # worker-exit detection or the terminal message will settle it.
            pass

    def _send_control(self, message):
        with self._io_lock:
            self._send_control_locked(message)

    def _close_control_conn(self):
        with self._io_lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except OSError:
                    pass
                self._conn = None

    def _deliver_item(self, seq, item):
        # Backpressure guarantees a free slot: every unacknowledged item is
        # either in flight on the result pipe or held in this buffer, and the
        # worker stops producing once their number reaches the buffer size.
        # This therefore never blocks the executor manager thread.
        self._buffer.put_nowait((seq, item))
        with self._cond:
            self._cond.notify_all()

    def _finish(self, state, return_value=None, exception=None, discard=False):
        self._close_control_conn()
        with self._cond:
            if self._done:
                return
            self._done = True
            self._state = state
            self._return_value = return_value
            self._exception = exception
            if discard:
                # Cooperative close or broken pool: drop undelivered items so
                # that the consumer is released immediately.
                while not self._buffer.empty():
                    try:
                        self._buffer.get_nowait()
                    except queue.Empty:
                        break
            self._cond.notify_all()

    def _fail(self, exception):
        # Used when the executor itself is broken, a shutdown with
        # kill_workers is requested, or the stream could not be dispatched.
        self._finish(self.BROKEN, exception=exception, discard=True)


class ProcessPoolExecutor(Executor):

    _at_exit = None

    def __init__(
        self,
        max_workers=None,
        job_reducers=None,
        result_reducers=None,
        timeout=None,
        context=None,
        initializer=None,
        initargs=(),
        env=None,
    ):
        """Initializes a new ProcessPoolExecutor instance.

        Args:
            max_workers: int, optional (default: cpu_count())
                The maximum number of processes that can be used to execute the
                given calls. If None or not given then as many worker processes
                will be created as the number of CPUs the current process
                can use.
            job_reducers, result_reducers: dict(type: reducer_func)
                Custom reducer for pickling the jobs and the results from the
                Executor. If only `job_reducers` is provided, `result_reducer`
                will use the same reducers
            timeout: int, optional (default: None)
                Idle workers exit after timeout seconds. If a new job is
                submitted after the timeout, the executor will start enough
                new Python processes to make sure the pool of workers is full.
            context: A multiprocessing context to launch the workers. This
                object should provide SimpleQueue, Queue and Process.
            initializer: An callable used to initialize worker processes.
            initargs: A tuple of arguments to pass to the initializer.
            env: A dict of environment variable to overwrite in the child
                process. The environment variables are set before any module is
                loaded. Note that this only works with the loky context.
        """
        _check_system_limits()

        if max_workers is None:
            self._max_workers = cpu_count()
        else:
            if max_workers <= 0:
                raise ValueError("max_workers must be greater than 0")
            self._max_workers = max_workers

        if (
            sys.platform == "win32"
            and self._max_workers > _MAX_WINDOWS_WORKERS
        ):
            warnings.warn(
                f"On Windows, max_workers cannot exceed {_MAX_WINDOWS_WORKERS} "
                "due to limitations of the operating system."
            )
            self._max_workers = _MAX_WINDOWS_WORKERS

        if context is None:
            context = get_context()
        self._context = context
        self._env = env

        self._initializer, self._initargs = _prepare_initializer(
            initializer, initargs
        )
        _check_max_depth(self._context)

        if result_reducers is None:
            result_reducers = job_reducers

        # Timeout
        self._timeout = timeout

        # Management thread
        self._executor_manager_thread = None

        # Map of pids to processes
        self._processes = {}

        # Internal variables of the ProcessPoolExecutor
        self._processes = {}
        self._queue_count = 0
        self._pending_work_items = {}
        self._running_work_items = []
        self._work_ids = queue.Queue()
        self._processes_management_lock = self._context.Lock()
        self._executor_manager_thread = None
        self._shutdown_lock = threading.Lock()

        # _ThreadWakeup is a communication channel used to interrupt the wait
        # of the main loop of executor_manager_thread from another thread (e.g.
        # when calling executor.submit or executor.shutdown). We do not use the
        # _result_queue to send wakeup signals to the executor_manager_thread
        # as it could result in a deadlock if a worker process dies with the
        # _result_queue write lock still acquired.
        #
        # _shutdown_lock must be locked to access _ThreadWakeup.wakeup.
        self._executor_manager_thread_wakeup = _ThreadWakeup()

        # Flag to hold the state of the Executor. This permits to introspect
        # the Executor state even once it has been garbage collected.
        self._flags = _ExecutorFlags(self._shutdown_lock)

        # Finally setup the queues for interprocess communication
        self._setup_queues(job_reducers, result_reducers)

        mp.util.debug("ProcessPoolExecutor is setup")

    def _setup_queues(self, job_reducers, result_reducers, queue_size=None):
        # Make the call queue slightly larger than the number of processes to
        # prevent the worker processes from idling. But don't make it too big
        # because futures in the call queue cannot be cancelled.
        if queue_size is None:
            queue_size = 2 * self._max_workers + EXTRA_QUEUED_CALLS
        self._call_queue = _SafeQueue(
            max_size=queue_size,
            pending_work_items=self._pending_work_items,
            running_work_items=self._running_work_items,
            thread_wakeup=self._executor_manager_thread_wakeup,
            shutdown_lock=self._shutdown_lock,
            executor_flags=self._flags,
            reducers=job_reducers,
            ctx=self._context,
        )
        # Killed worker processes can produce spurious "broken pipe"
        # tracebacks in the queue's own worker thread. But we detect killed
        # processes anyway, so silence the tracebacks.
        self._call_queue._ignore_epipe = True

        self._result_queue = SimpleQueue(
            reducers=result_reducers, ctx=self._context
        )

    def _start_executor_manager_thread(self):
        if self._executor_manager_thread is None:
            mp.util.debug("_start_executor_manager_thread called")

            # Start the processes so that their sentinels are known.
            self._executor_manager_thread = _ExecutorManagerThread(self)
            self._executor_manager_thread.start()

            # register this executor in a mechanism that ensures it will wakeup
            # when the interpreter is exiting.
            _threads_wakeups[self._executor_manager_thread] = (
                self._shutdown_lock,
                self._executor_manager_thread_wakeup,
            )

            global process_pool_executor_at_exit
            if process_pool_executor_at_exit is None:
                # Ensure that the _python_exit function will be called before
                # the multiprocessing.Queue._close finalizers which have an
                # exitpriority of 10.

                if sys.version_info < (3, 9):
                    process_pool_executor_at_exit = mp.util.Finalize(
                        None, _python_exit, exitpriority=20
                    )
                else:
                    process_pool_executor_at_exit = threading._register_atexit(
                        _python_exit
                    )

    def _adjust_process_count(self):
        while len(self._processes) < self._max_workers:
            worker_exit_lock = self._context.BoundedSemaphore(1)
            args = (
                self._call_queue,
                self._result_queue,
                self._initializer,
                self._initargs,
                self._processes_management_lock,
                self._timeout,
                worker_exit_lock,
                _CURRENT_DEPTH + 1,
            )
            worker_exit_lock.acquire()
            try:
                # Try to spawn the process with some environment variable to
                # overwrite but it only works with the loky context for now.
                p = self._context.Process(
                    target=_process_worker, args=args, env=self._env
                )
            except TypeError:
                p = self._context.Process(target=_process_worker, args=args)
            p._worker_exit_lock = worker_exit_lock
            p.start()
            self._processes[p.pid] = p
        mp.util.debug(
            f"Adjusted process count to {self._max_workers}: "
            f"{[(p.name, pid) for pid, p in self._processes.items()]}"
        )

    def _ensure_executor_running(self):
        """ensures all workers and management thread are running"""
        with self._processes_management_lock:
            if len(self._processes) != self._max_workers:
                self._adjust_process_count()
            self._start_executor_manager_thread()

    def submit(self, fn, *args, **kwargs):
        with self._flags.shutdown_lock:
            if self._flags.broken is not None:
                raise self._flags.broken
            if self._flags.shutdown:
                raise ShutdownExecutorError(
                    "cannot schedule new futures after shutdown"
                )

            # Cannot submit a new calls once the interpreter is shutting down.
            # This check avoids spawning new processes at exit.
            if _global_shutdown:
                raise RuntimeError(
                    "cannot schedule new futures after interpreter shutdown"
                )

            f = Future()
            w = _WorkItem(f, fn, args, kwargs)

            self._pending_work_items[self._queue_count] = w
            self._work_ids.put(self._queue_count)
            self._queue_count += 1
            # Wake up queue management thread
            self._executor_manager_thread_wakeup.wakeup()

            self._ensure_executor_running()
            return f

    submit.__doc__ = Executor.submit.__doc__

    def submit_stream(self, fn, *args, max_buffered_items=1, **kwargs):
        """Run a generator function in a worker and stream its yielded items.

        Contrary to :meth:`submit`, which only delivers the final object
        returned by the worker, ``fn`` must be a generator function and each
        value it ``yield``\\ s is delivered to the returned
        :class:`StreamResult` as soon as it is produced, without waiting for
        the generator to finish.

        Args:
            fn: A generator function (a callable whose result is a generator
                when called with ``fn(*args, **kwargs)``).
            *args: Positional arguments forwarded to ``fn``.
            max_buffered_items: Positive integer bounding the number of
                yielded items that may be waiting for the consumer at any
                time (the backpressure window). The worker only advances the
                generator past this window once the consumer acknowledges the
                retrieved items. It also bounds the parent-side buffer.
            **kwargs: Keyword arguments forwarded to ``fn``.

        Returns:
            StreamResult: a synchronous, single-consumer iterable yielding
            the items produced by the remote generator, supporting explicit
            :meth:`~StreamResult.close` and exposing the final state of the
            stream (generator return value or remote exception).
        """
        if not isinstance(max_buffered_items, int) or max_buffered_items <= 0:
            raise ValueError(
                "max_buffered_items must be a positive integer, got "
                f"{max_buffered_items!r}"
            )
        with self._flags.shutdown_lock:
            if self._flags.broken is not None:
                raise self._flags.broken
            if self._flags.shutdown:
                raise ShutdownExecutorError(
                    "cannot schedule new streams after shutdown"
                )

            # Cannot submit a new call once the interpreter is shutting down.
            # This check avoids spawning new processes at exit.
            if _global_shutdown:
                raise RuntimeError(
                    "cannot schedule new streams after interpreter shutdown"
                )

            # Per-stream one-way control pipe. The worker receives the read
            # end through the call queue; the parent keeps the write end to
            # send consumption acknowledgements and close requests.
            worker_conn, parent_conn = self._context.Pipe(duplex=False)
            stream = StreamResult(
                self._queue_count, parent_conn, max_buffered_items
            )
            work_item = _StreamWorkItem(
                stream,
                fn,
                args,
                kwargs,
                max_buffered_items,
                worker_conn,
            )

            self._pending_work_items[self._queue_count] = work_item
            self._work_ids.put(self._queue_count)
            self._queue_count += 1
            # Wake up queue management thread
            self._executor_manager_thread_wakeup.wakeup()

            self._ensure_executor_running()
            return stream

    def map(self, fn, *iterables, **kwargs):
        """Returns an iterator equivalent to map(fn, iter).

        Args:
            fn: A callable that will take as many arguments as there are
                passed iterables.
            timeout: The maximum number of seconds to wait. If None, then there
                is no limit on the wait time.
            chunksize: If greater than one, the iterables will be chopped into
                chunks of size chunksize and submitted to the process pool.
                If set to one, the items in the list will be sent one at a
                time.

        Returns:
            An iterator equivalent to: map(func, *iterables) but the calls may
            be evaluated out-of-order.

        Raises:
            TimeoutError: If the entire result iterator could not be generated
                before the given timeout.
            Exception: If fn(*args) raises for any values.
        """
        timeout = kwargs.get("timeout", None)
        chunksize = kwargs.get("chunksize", 1)
        if chunksize < 1:
            raise ValueError("chunksize must be >= 1.")

        results = super().map(
            partial(_process_chunk, fn),
            _get_chunks(chunksize, *iterables),
            timeout=timeout,
        )
        return _chain_from_iterable_of_lists(results)

    def shutdown(self, wait=True, kill_workers=False):
        mp.util.debug(f"shutting down executor {self}")

        self._flags.flag_as_shutting_down(kill_workers)
        executor_manager_thread = self._executor_manager_thread
        executor_manager_thread_wakeup = self._executor_manager_thread_wakeup

        if executor_manager_thread_wakeup is not None:
            # Wake up queue management thread
            with self._shutdown_lock:
                self._executor_manager_thread_wakeup.wakeup()

        if executor_manager_thread is not None and wait:
            # This locks avoids concurrent join if the interpreter
            # is shutting down.
            with _global_shutdown_lock:
                executor_manager_thread.join()
                _threads_wakeups.pop(executor_manager_thread, None)

        # To reduce the risk of opening too many files, remove references to
        # objects that use file descriptors.
        self._executor_manager_thread = None
        self._executor_manager_thread_wakeup = None
        self._call_queue = None
        self._result_queue = None
        self._processes_management_lock = None

    shutdown.__doc__ = Executor.shutdown.__doc__
