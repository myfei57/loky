"""Tests for ``ProcessPoolExecutor.submit_retry``.

The retry scheduling is verified without relying on long real-time sleeps:
executors are built with an injectable :class:`FakeClock`, and manager-backed
``Event``/``Value`` proxies (which are picklable across spawned workers) gate
the worker functions. The test thread drives the fake clock forward and
explicitly wakes the executor manager thread, so correctness only depends on
the observed state transitions.
"""

import time
import threading
import contextlib
import multiprocessing as mp
from pickle import PicklingError

import pytest
from concurrent.futures import wait as futures_wait

from loky.backend import get_context
from loky.process_executor import (
    BrokenProcessPool,
    ProcessPoolExecutor,
    RetryFuture,
    RetryPolicy,
    ShutdownExecutorError,
    _RemoteTraceback,
)
from tests.test_reusable_executor import ErrorAtPickle


_CONTEXT = get_context("loky")


@pytest.fixture
def manager():
    # Manager proxies (Event/Value) can be pickled and sent to spawn-based
    # workers, unlike raw context synchronized objects.
    manager = mp.Manager()
    yield manager
    manager.shutdown()


# ---------------------------------------------------------------------------
# Module-level, importable worker helpers and exception types. Nothing defined
# inside a test function can be sent to spawn-based workers.
# ---------------------------------------------------------------------------


class RetryableError(Exception):
    """An exception type explicitly allowed for retries."""


class OtherRetryableError(Exception):
    """A second retryable exception type."""


class NonRetryableError(Exception):
    """An exception type that must never trigger a retry."""


class FakeClock:
    """Controllable monotonic clock injected into the executor."""

    def __init__(self, start=1000.0):
        self._now = float(start)

    def __call__(self):
        return self._now

    def monotonic(self):
        return self._now

    def advance(self, delta):
        if delta < 0:
            raise ValueError("cannot move the clock backwards")
        self._now += float(delta)


def wait_for(predicate, timeout=20.0, interval=0.005):
    """Poll a predicate with short real-time sleeps, never long ones."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError("condition was not satisfied before the timeout")


def wake_manager(executor):
    """Interrupt the executor manager thread so it re-evaluates its state."""
    with executor._shutdown_lock:
        executor._executor_manager_thread_wakeup.wakeup()


def make_executor(max_workers=2, clock=None):
    return ProcessPoolExecutor(
        max_workers=max_workers,
        context=_CONTEXT,
        clock=clock,
    )


@contextlib.contextmanager
def retry_executor(max_workers=2, clock=None):
    # On failure, tear the pool down forcibly: with a frozen fake clock a
    # graceful shutdown would otherwise block waiting for backoffs that no
    # test thread is going to schedule, masking the original error.
    executor = make_executor(max_workers, clock)
    try:
        yield executor
    except BaseException:
        executor.shutdown(wait=True, kill_workers=True)
        raise
    else:
        executor.shutdown(wait=True)


def drive_until_done(executor, clock, future, max_steps=30):
    """Advance the fake clock over every backoff until the future completes."""
    manager_thread = executor._executor_manager_thread
    for _ in range(max_steps):
        if future.done():
            return
        wait_for(
            lambda: bool(manager_thread.scheduled_retries) or future.done()
        )
        if future.done():
            return
        entry = manager_thread.scheduled_retries[0]
        due_time = entry[0]
        clock.advance(max(0.0, due_time - clock()) + 1e-6)
        wake_manager(executor)
        # Wait until the manager actually consumed this scheduled entry
        # (or the future terminated) instead of re-reading the stale entry
        # and treating it as the next backoff.
        wait_for(
            lambda: future.done()
            or entry not in manager_thread.scheduled_retries
        )
    raise AssertionError("future did not complete within the driven steps")


def wait_until_entry_consumed(executor, entry, future):
    """Block until a given heap entry has been dispatched or removed."""
    manager_thread = executor._executor_manager_thread
    wait_for(
        lambda: future.done() or entry not in manager_thread.scheduled_retries
    )


def block_on_event(event):
    event.wait(timeout=30)
    return "released"


def simple_return(value):
    return value


def fail_n_then_succeed(counter, fail_times, event):
    event.wait(timeout=30)
    # Attempts of a single task never run concurrently, so the manager proxy
    # read-modify-write does not need an explicit lock.
    counter.value += 1
    attempt = counter.value
    if attempt <= fail_times:
        raise RetryableError(f"failing on attempt {attempt}")
    return attempt


def always_fail_retryable(event):
    event.wait(timeout=30)
    raise RetryableError("always failing")


def raise_given_exception(error_type, event):
    event.wait(timeout=30)
    raise error_type()


def raise_system_exit(event):
    event.wait(timeout=30)
    raise SystemExit(7)


def return_unpicklable():
    # The function itself succeeds; only serializing its result fails.
    return ErrorAtPickle()


def mutate_then_maybe_fail(container, counter, fail_times, event):
    event.wait(timeout=30)
    # Every attempt must start from the pristine snapshot captured at
    # submission: a carried-over mutation would make this assertion fail.
    assert container == ["a"], container
    container.append("mutation")
    counter.value += 1
    attempt = counter.value
    if attempt <= fail_times:
        raise RetryableError(f"failing on attempt {attempt}")
    return list(container)


def record_callback(call_log, future):
    call_log.append(future)


class TestRetryPolicyValidation:
    def test_valid_policy(self):
        policy = RetryPolicy(
            retry_on=RetryableError,
            max_attempts=3,
            initial_delay=1.0,
            max_delay=10.0,
        )
        assert policy.retry_on == (RetryableError,)
        assert policy.max_attempts == 3
        # Exponential growth capped by max_delay.
        assert policy.delay_for_next_attempt(1) == 1.0
        assert policy.delay_for_next_attempt(2) == 2.0
        assert policy.delay_for_next_attempt(3) == 4.0
        assert policy.delay_for_next_attempt(4) == 8.0
        assert policy.delay_for_next_attempt(5) == 10.0

    def test_zero_delays_are_allowed(self):
        policy = RetryPolicy(RetryableError, 2, 0, 0)
        assert policy.delay_for_next_attempt(1) == 0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {
                "retry_on": int,
                "max_attempts": 2,
                "initial_delay": 1.0,
                "max_delay": 1.0,
            },
            {
                "retry_on": (),
                "max_attempts": 2,
                "initial_delay": 1.0,
                "max_delay": 1.0,
            },
            {
                "retry_on": KeyboardInterrupt,
                "max_attempts": 2,
                "initial_delay": 1.0,
                "max_delay": 1.0,
            },
            {
                "retry_on": SystemExit,
                "max_attempts": 2,
                "initial_delay": 1.0,
                "max_delay": 1.0,
            },
        ],
    )
    def test_invalid_retry_on(self, kwargs):
        with pytest.raises((TypeError, ValueError)):
            RetryPolicy(**kwargs)

    def test_non_importable_exception_is_rejected(self):
        class LocalError(Exception):
            pass

        with pytest.raises(ValueError):
            RetryPolicy(LocalError, 2, 1.0, 1.0)

    @pytest.mark.parametrize("max_attempts", [0, -1, 1.5, True])
    def test_invalid_max_attempts(self, max_attempts):
        with pytest.raises(ValueError):
            RetryPolicy(RetryableError, max_attempts, 1.0, 1.0)

    def test_invalid_delays(self):
        with pytest.raises(ValueError):
            RetryPolicy(RetryableError, 2, -1.0, 1.0)
        with pytest.raises(ValueError):
            RetryPolicy(RetryableError, 2, 10.0, 1.0)


class TestSubmitRetry:
    def test_retry_succeeds_after_failures(self, manager):
        clock = FakeClock()
        event = manager.Event()
        event.set()
        counter = manager.Value("i", 0)
        with retry_executor(max_workers=2, clock=clock) as executor:
            future = executor.submit_retry(
                fail_n_then_succeed,
                counter,
                2,
                event,
                retry_on=RetryableError,
                max_attempts=4,
                initial_delay=1.0,
                max_delay=10.0,
            )
            assert isinstance(future, RetryFuture)
            drive_until_done(executor, clock, future)

            assert future.result(timeout=15) == 3
            assert future.attempt_count == 3
            assert future.retry_policy.max_attempts == 4

    def test_exhausted_attempts_keep_exception_traceback_and_count(
        self, manager
    ):
        clock = FakeClock()
        event = manager.Event()
        event.set()
        with retry_executor(max_workers=2, clock=clock) as executor:
            future = executor.submit_retry(
                always_fail_retryable,
                event,
                retry_on=RetryableError,
                max_attempts=3,
                initial_delay=1.0,
                max_delay=10.0,
            )
            drive_until_done(executor, clock, future)

            with pytest.raises(RetryableError) as cm:
                future.result(timeout=15)
            # The final exception type and its remote traceback are preserved.
            assert type(cm.value) is RetryableError
            cause = cm.value.__cause__
            assert isinstance(cause, _RemoteTraceback)
            assert "always_fail_retryable" in cause.tb
            assert future.attempt_count == 3
            assert not executor._executor_manager_thread.scheduled_retries

    def test_backoff_schedule_and_cap(self, manager):
        clock = FakeClock()
        event = manager.Event()
        event.set()
        with retry_executor(max_workers=1, clock=clock) as executor:
            future = executor.submit_retry(
                always_fail_retryable,
                event,
                retry_on=RetryableError,
                max_attempts=6,
                initial_delay=2.0,
                max_delay=10.0,
            )
            manager_thread = executor._executor_manager_thread
            expected_delays = [2.0, 4.0, 8.0, 10.0, 10.0]
            for expected_delay in expected_delays:
                wait_for(lambda: bool(manager_thread.scheduled_retries))
                entry = manager_thread.scheduled_retries[0]
                due_time, _ = entry
                # Exponential growth up to the cap; tolerance only absorbs
                # the epsilon injected to move the fake clock forward.
                assert due_time == pytest.approx(
                    clock() + expected_delay, abs=1e-6
                )
                assert expected_delay <= 10.0
                clock.advance(expected_delay + 1e-6)
                wake_manager(executor)
                wait_until_entry_consumed(executor, entry, future)

            drive_until_done(executor, clock, future)
            with pytest.raises(RetryableError):
                future.result()
            assert future.attempt_count == 6

    def test_non_retryable_exception_is_not_retried(self, manager):
        clock = FakeClock()
        event = manager.Event()
        event.set()
        with retry_executor(clock=clock) as executor:
            future = executor.submit_retry(
                raise_given_exception,
                NonRetryableError,
                event,
                retry_on=RetryableError,
                max_attempts=3,
                initial_delay=1.0,
                max_delay=10.0,
            )
            with pytest.raises(NonRetryableError):
                future.result(timeout=15)
            assert future.attempt_count == 1
            assert not executor._executor_manager_thread.scheduled_retries

    def test_retry_on_tuple_matches_any_listed_type(self, manager):
        clock = FakeClock()
        event = manager.Event()
        event.set()
        counter = manager.Value("i", 0)
        with retry_executor(clock=clock) as executor:
            future = executor.submit_retry(
                fail_n_then_succeed,
                counter,
                1,
                event,
                retry_on=(OtherRetryableError, RetryableError),
                max_attempts=3,
                initial_delay=1.0,
                max_delay=10.0,
            )
            drive_until_done(executor, clock, future)
            assert future.result() == 2
            assert future.attempt_count == 2

    def test_base_exception_is_not_retried_and_pool_survives(self, manager):
        clock = FakeClock()
        event = manager.Event()
        event.set()
        with retry_executor(clock=clock) as executor:
            future = executor.submit_retry(
                raise_system_exit,
                event,
                retry_on=RetryableError,
                max_attempts=3,
                initial_delay=1.0,
                max_delay=10.0,
            )
            with pytest.raises(SystemExit):
                future.result(timeout=15)
            assert future.attempt_count == 1
            assert not executor._executor_manager_thread.scheduled_retries
            # The pool must not be broken nor masked by a retry.
            assert executor._flags.broken is None
            assert executor.submit(simple_return, 42).result() == 42

    def test_snapshot_failure_raises_before_first_execution(self):
        clock = FakeClock()
        with retry_executor(clock=clock) as executor:
            pending_before = dict(executor._pending_work_items)
            with pytest.raises(PicklingError):
                executor.submit_retry(
                    simple_return,
                    ErrorAtPickle(),
                    retry_on=RetryableError,
                    max_attempts=3,
                    initial_delay=1.0,
                    max_delay=10.0,
                )
            # Nothing was queued and the executor is still usable.
            assert executor._pending_work_items == pending_before
            assert executor.submit(simple_return, 1).result() == 1

    def test_result_serialization_error_is_not_retried(self):
        # Even with the broadest possible retry_on, a failure happening
        # while serializing a successful result is not a user-function error
        # and must be reported after a single attempt.
        with retry_executor() as executor:
            future = executor.submit_retry(
                return_unpicklable,
                retry_on=Exception,
                max_attempts=3,
                initial_delay=0.0,
                max_delay=0.0,
            )
            with pytest.raises(PicklingError):
                future.result(timeout=20)
            assert future.attempt_count == 1
            assert not executor._executor_manager_thread.scheduled_retries

    def test_snapshot_isolates_arguments_between_attempts(self, manager):
        clock = FakeClock()
        event = manager.Event()
        event.set()
        counter = manager.Value("i", 0)
        with retry_executor(max_workers=1, clock=clock) as executor:
            future = executor.submit_retry(
                mutate_then_maybe_fail,
                ["a"],
                counter,
                2,
                event,
                retry_on=RetryableError,
                max_attempts=4,
                initial_delay=1.0,
                max_delay=10.0,
            )
            drive_until_done(executor, clock, future)
            # Three attempts ran, each on the pristine snapshot.
            assert counter.value == 3
            assert future.result() == ["a", "mutation"]

    def test_done_callback_invoked_once_on_success_and_failure(self, manager):
        clock = FakeClock()
        event = manager.Event()
        event.set()
        counter = manager.Value("i", 0)
        with retry_executor(max_workers=1, clock=clock) as executor:
            success_log = []
            success_future = executor.submit_retry(
                fail_n_then_succeed,
                counter,
                1,
                event,
                retry_on=RetryableError,
                max_attempts=3,
                initial_delay=1.0,
                max_delay=10.0,
            )
            success_future.add_done_callback(
                lambda f: record_callback(success_log, f)
            )

            failure_log = []
            failure_future = executor.submit_retry(
                always_fail_retryable,
                event,
                retry_on=RetryableError,
                max_attempts=2,
                initial_delay=1.0,
                max_delay=10.0,
            )
            failure_future.add_done_callback(
                lambda f: record_callback(failure_log, f)
            )

            drive_until_done(executor, clock, failure_future)
            drive_until_done(executor, clock, success_future)

            assert success_future.result() == 2
            with pytest.raises(RetryableError):
                failure_future.result()
            assert success_log == [success_future]
            assert failure_log == [failure_future]

    def test_cancel_before_first_dispatch(self, manager):
        event = manager.Event()
        # One worker and a call queue of size 2 * 1 + 1 = 3: the first task
        # runs and the next three fill the queue, so the fifth stays pending.
        with retry_executor(max_workers=1) as executor:
            blocking = [
                executor.submit(block_on_event, event) for _ in range(4)
            ]
            retry_future = executor.submit_retry(
                block_on_event,
                event,
                retry_on=RetryableError,
                max_attempts=3,
                initial_delay=1.0,
                max_delay=10.0,
            )
            wait_for(lambda: executor._work_ids.qsize() >= 1)
            assert retry_future.cancel() is True
            assert retry_future.cancelled()

            event.set()
            for future in blocking:
                assert future.result(timeout=15) == "released"

    def test_cancel_after_dispatch_returns_false_and_wait_does_not_complete(
        self, manager
    ):
        clock = FakeClock()
        event = manager.Event()
        event.set()
        with retry_executor(max_workers=1, clock=clock) as executor:
            future = executor.submit_retry(
                always_fail_retryable,
                event,
                retry_on=RetryableError,
                max_attempts=3,
                initial_delay=10.0,
                max_delay=100.0,
            )
            manager_thread = executor._executor_manager_thread
            wait_for(lambda: bool(manager_thread.scheduled_retries))

            # During the backoff the same future stays running and can neither
            # be cancelled nor reported as done by concurrent.futures.wait.
            assert future.running()
            assert future.cancel() is False
            assert future.running()
            done, not_done = futures_wait([future], timeout=0.2)
            assert done == set()
            assert not_done == {future}

            drive_until_done(executor, clock, future)
            with pytest.raises(RetryableError):
                future.result()
            assert future.attempt_count == 3

    def test_normal_task_uses_freed_worker_and_finishes_first(self, manager):
        clock = FakeClock()
        event = manager.Event()
        event.set()
        # The retry task is intentionally left parked in its backoff, so use
        # a manual executor and tear it down forcibly instead of a graceful
        # shutdown that would wait for retries on the frozen clock.
        executor = make_executor(max_workers=1, clock=clock)
        try:
            retry_future = executor.submit_retry(
                always_fail_retryable,
                event,
                retry_on=RetryableError,
                max_attempts=3,
                initial_delay=100.0,
                max_delay=1000.0,
            )
            manager_thread = executor._executor_manager_thread
            wait_for(lambda: bool(manager_thread.scheduled_retries))

            # The worker is free while the retry waits for its backoff: a
            # plain task submitted now gets executed immediately.
            normal_future = executor.submit(simple_return, 123)
            assert normal_future.result(timeout=15) == 123

            assert not retry_future.done()
            assert retry_future.running()
            assert len(manager_thread.scheduled_retries) == 1
        finally:
            executor.shutdown(wait=True, kill_workers=True)

    def test_default_submit_does_not_retry(self, manager):
        event = manager.Event()
        event.set()
        with retry_executor() as executor:
            future = executor.submit(always_fail_retryable, event)
            assert not isinstance(future, RetryFuture)
            assert not hasattr(future, "attempt_count")
            with pytest.raises(RetryableError):
                future.result(timeout=15)
            assert not executor._executor_manager_thread.scheduled_retries

    def test_zero_delay_retries_run_with_real_clock(self, manager):
        # End-to-end check on the default monotonic clock: a zero backoff
        # still performs all attempts without any worker-side sleeping.
        event = manager.Event()
        event.set()
        with retry_executor(max_workers=1) as executor:
            future = executor.submit_retry(
                always_fail_retryable,
                event,
                retry_on=RetryableError,
                max_attempts=3,
                initial_delay=0,
                max_delay=0,
            )
            with pytest.raises(RetryableError):
                future.result(timeout=15)
            assert future.attempt_count == 3


class TestRetryShutdown:
    def test_normal_shutdown_waits_for_finite_retries(self, manager):
        clock = FakeClock()
        event = manager.Event()
        event.set()
        executor = make_executor(max_workers=1, clock=clock)
        future = executor.submit_retry(
            always_fail_retryable,
            event,
            retry_on=RetryableError,
            max_attempts=2,
            initial_delay=5.0,
            max_delay=5.0,
        )
        manager_thread = executor._executor_manager_thread
        wait_for(lambda: bool(manager_thread.scheduled_retries))

        shutdown_thread = threading.Thread(
            target=executor.shutdown, kwargs={"wait": True}
        )
        shutdown_thread.start()
        try:
            # Wait for the shutdown flag itself rather than thread liveness,
            # which is observable before the thread runs any shutdown code.
            wait_for(lambda: executor._flags.shutdown)
            # The executor is shutting down while the backoff is still pending.
            assert not future.done()

            # Letting the deadline elapse during shutdown allows the final
            # attempt to run; shutdown must wait for this finite retry to
            # complete.
            due_time = manager_thread.scheduled_retries[0][0]
            clock.advance(due_time - clock() + 1e-6)
            wake_manager(executor)

            shutdown_thread.join(timeout=20)
            assert not shutdown_thread.is_alive()
            with pytest.raises(RetryableError):
                future.result()
            assert future.attempt_count == 2
            assert not manager_thread.is_alive()
        finally:
            # Never leak a shutdown thread blocked on a frozen clock.
            if shutdown_thread.is_alive():
                clock.advance(1e9)
                wake_manager(executor)
                shutdown_thread.join(timeout=20)

    def test_kill_workers_ends_backoff_and_clears_timed_items(self, manager):
        n_tasks = 50
        clock = FakeClock()
        event = manager.Event()
        event.set()
        executor = make_executor(max_workers=4, clock=clock)
        try:
            futures = [
                executor.submit_retry(
                    always_fail_retryable,
                    event,
                    retry_on=RetryableError,
                    max_attempts=3,
                    initial_delay=10.0,
                    max_delay=100.0,
                )
                for _ in range(n_tasks)
            ]
            manager_thread = executor._executor_manager_thread
            # Every task fails once and parks in the backoff heap.
            wait_for(lambda: len(manager_thread.scheduled_retries) == n_tasks)
            assert all(f.running() and not f.done() for f in futures)

            executor.shutdown(wait=True, kill_workers=True)

            # No timed item survives and every future failed exactly once.
            assert not manager_thread.scheduled_retries
            assert not executor._pending_work_items
            for future in futures:
                with pytest.raises(ShutdownExecutorError):
                    future.result()
        finally:
            if executor._flags.broken is not None:
                executor.shutdown(wait=True, kill_workers=True)

    def test_broken_pool_during_backoff_is_not_masked(self, manager):
        clock = FakeClock()
        event = manager.Event()
        event.set()
        executor = make_executor(max_workers=2, clock=clock)
        try:
            future = executor.submit_retry(
                always_fail_retryable,
                event,
                retry_on=RetryableError,
                max_attempts=3,
                initial_delay=100.0,
                max_delay=1000.0,
            )
            manager_thread = executor._executor_manager_thread
            wait_for(lambda: bool(manager_thread.scheduled_retries))
            attempts_at_break = future.attempt_count
            assert attempts_at_break == 1

            # Killing a worker breaks the pool: the parked retry must fail
            # with the broken-pool error instead of being re-executed.
            worker = next(iter(executor._processes.values()))
            worker.terminate()

            with pytest.raises(BrokenProcessPool):
                future.result(timeout=20)
            assert executor._flags.broken is not None
            assert not manager_thread.scheduled_retries
            assert future.attempt_count == attempts_at_break
        finally:
            executor.shutdown(wait=True, kill_workers=True)
