"""Tests for the publish_object / submit_with_objects API.

The feature publishes a large read-only input once in the executor parent
process and restores it at most once per worker, behind an opaque
ObjectHandle. These tests use real worker pids and custom reduce/rebuild
counters to prove that snapshots are serialized once and deserialized at most
once per worker, instead of being re-pickled with every task.
"""

import math
import os
import pickle
import tempfile
import time
import uuid

import pytest

from loky import (
    CancelledError,
    ObjectHandle,
    ObjectHandleError,
    ProcessPoolExecutor,
    get_reusable_executor,
)
from loky._objects import _ObjectKey


# ---------------------------------------------------------------------------
# Helpers for counting (de)serializations. They must live at module level so
# worker processes can import them by reference.
# ---------------------------------------------------------------------------

_PARENT_REDUCE_COUNT = {}
_WORKER_REBUILD_COUNT = {}


class CountedObject:
    """Object whose pickling and rebuilding are counted on each side."""

    def __init__(self, value, token):
        self.value = value
        self.token = token

    def __reduce__(self):
        _PARENT_REDUCE_COUNT[self.token] = (
            _PARENT_REDUCE_COUNT.get(self.token, 0) + 1
        )
        return _rebuild_counted, (self.value, self.token)


def _rebuild_counted(value, token):
    _WORKER_REBUILD_COUNT[token] = _WORKER_REBUILD_COUNT.get(token, 0) + 1
    obj = CountedObject(value, token)
    obj.rebuild_count = _WORKER_REBUILD_COUNT[token]
    return obj


class Unrebuildable:
    """Serializes fine but always fails when restored in a worker."""

    def __reduce__(self):
        return _raise_on_rebuild, ()


def _raise_on_rebuild():
    raise ValueError("snapshot restoration is expected to fail")


class Unpicklable:
    def __reduce__(self):
        raise pickle.PicklingError("this object cannot be serialized")


class SpecialReduced:
    def __init__(self, x):
        self.x = x


def _reduce_special(obj):
    # Marker proving the executor job_reducers are used at publish time.
    return _rebuild_special, (obj.x + 1000,)


def _rebuild_special(x):
    obj = SpecialReduced.__new__(SpecialReduced)
    obj.x = x
    return obj


# ---------------------------------------------------------------------------
# Task functions run in the workers
# ---------------------------------------------------------------------------


def _task_identity(obj):
    return os.getpid(), obj


def _task_counted(obj):
    return (
        os.getpid(),
        _WORKER_REBUILD_COUNT.get(obj.token, 0),
        obj.value,
    )


def _task_two_positions(a, b, extra=None):
    return os.getpid(), a is b, a, b, extra


def _task_three_positions(a, b, c, extra):
    return a is b, b is c, a["k"], extra


def _barrier_wait(idx, flag_dir, prefix):
    # Two-worker barrier through plain files: the task with ``idx`` signals
    # it started and waits until both tasks are running, which guarantees one
    # task per worker since each worker can only run one task at a time.
    with open(os.path.join(flag_dir, f"{prefix}_{idx}"), "w") as fh:
        fh.write("1")
    deadline = time.time() + 30
    while not all(
        os.path.exists(os.path.join(flag_dir, f"{prefix}_{i}"))
        for i in range(2)
    ):
        if time.time() > deadline:
            raise TimeoutError("the two-worker barrier timed out")
        time.sleep(0.02)


def _task_mutate_copy(obj, idx, flag_dir):
    _barrier_wait(idx, flag_dir, "mut")
    obj.append(os.getpid())
    return os.getpid(), list(obj)


def _task_counted_barrier(obj, idx, flag_dir):
    _barrier_wait(idx, flag_dir, "cnt")
    return (
        os.getpid(),
        _WORKER_REBUILD_COUNT.get(obj.token, 0),
        obj.value,
    )


def _task_handle_passthrough(maybe_handle):
    return (
        os.getpid(),
        isinstance(maybe_handle, ObjectHandle),
        maybe_handle,
    )


def _task_nested_handle(container):
    return os.getpid(), isinstance(container["x"], ObjectHandle)


def _task_wait_flag(obj, started_path, block_path):
    # File-based synchronization: loky synchronization primitives cannot be
    # pickled through task arguments on every platform, while plain paths
    # can. The task signals it started and then waits for the parent to
    # release it.
    with open(started_path, "w") as fh:
        fh.write("1")
    deadline = time.time() + 30
    while not os.path.exists(block_path):
        if time.time() > deadline:
            raise TimeoutError("the parent never released the test flag")
        time.sleep(0.02)
    return os.getpid(), obj


def _task_special_x(obj):
    return obj.x


def _task_sum(values):
    return sum(values)


def _new_flag_path():
    return os.path.join(
        tempfile.gettempdir(), f"loky_test_flag_{uuid.uuid4().hex}"
    )


def _wait_flag(path, timeout=30):
    deadline = time.time() + timeout
    while not os.path.exists(path):
        if time.time() > deadline:
            raise AssertionError(f"flag {path} never appeared")
        time.sleep(0.02)


def _set_flag(path):
    with open(path, "w") as fh:
        fh.write("1")


def _cleanup_flags(*paths):
    for path in paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPublishedObjects:
    def test_publish_serializes_once_and_worker_deserializes_once(self):
        token = "serialize-once"
        with ProcessPoolExecutor(max_workers=1) as executor:
            handle = executor.publish_object(CountedObject("hello", token))
            # The parent serialized the object exactly once at publish time.
            assert _PARENT_REDUCE_COUNT[token] == 1

            results = [
                executor.submit_with_objects(_task_counted, [handle]).result()
                for _ in range(4)
            ]

            pids = {pid for pid, _, _ in results}
            assert len(pids) == 1
            # The single worker restored the snapshot once and reused it.
            assert all(rebuild_count == 1 for _, rebuild_count, _ in results)
            assert all(value == "hello" for _, _, value in results)

            # A regular task does not interact with the object cache.
            assert executor.submit(math.sqrt, 16).result() == 4.0

            # The cached copy is still served without any extra rebuild.
            pid, rebuild_count, value = executor.submit_with_objects(
                _task_counted, [handle]
            ).result()
            assert pid in pids
            assert rebuild_count == 1
            assert value == "hello"

    def test_two_workers_hold_independent_copies(self):
        import shutil

        flag_dir = tempfile.mkdtemp(prefix="loky_two_workers_")
        try:
            with ProcessPoolExecutor(max_workers=2) as executor:
                handle = executor.publish_object([])
                futures = [
                    executor.submit_with_objects(
                        _task_mutate_copy, [handle], idx, flag_dir
                    )
                    for idx in range(2)
                ]
                results = [f.result() for f in futures]
                pids = {pid for pid, _ in results}
                assert len(pids) == 2

                # Each worker mutated its own independent copy exactly once:
                # neither can see the other worker's mutation and every copy
                # started from the same empty snapshot.
                for pid, snapshot in results:
                    assert snapshot == [pid]

                # Each worker restores the snapshot exactly once. The
                # barrier forces both workers to run one task each. All
                # tasks must be submitted before waiting on any result,
                # otherwise the barrier would deadlock itself.
                counted = executor.publish_object(
                    CountedObject("x", "two-workers")
                )
                counted_futures = [
                    executor.submit_with_objects(
                        _task_counted_barrier, [counted], idx, flag_dir
                    )
                    for idx in range(2)
                ]
                barrier_results = [f.result() for f in counted_futures]
                per_pid_counts = {
                    pid: rebuild_count
                    for pid, rebuild_count, _ in barrier_results
                }
                assert len(per_pid_counts) == 2
                assert set(per_pid_counts.values()) == {1}

                # Further tasks never trigger a second deserialization,
                # whichever worker picks them up.
                follow_up = [
                    executor.submit_with_objects(_task_counted, [counted])
                    for _ in range(4)
                ]
                assert all(future.result()[1] == 1 for future in follow_up)
        finally:
            shutil.rmtree(flag_dir, ignore_errors=True)

    def test_normal_submit_does_not_scan_or_resolve_handles(self):
        with ProcessPoolExecutor(max_workers=1) as executor:
            handle = executor.publish_object({"k": 1})

            # A handle travelling through the regular submit path is treated
            # as an opaque value and is never resolved.
            pid, is_handle, received = executor.submit(
                _task_handle_passthrough, handle
            ).result()
            assert is_handle is True
            assert isinstance(received, ObjectHandle)
            assert received._key == handle._key

            # Handles nested inside arbitrary argument graphs are not scanned
            # either.
            pid, nested_is_handle = executor.submit(
                _task_nested_handle, {"x": handle}
            ).result()
            assert nested_is_handle is True

            # The worker can still resolve the same handle explicitly.
            _, resolved = executor.submit_with_objects(
                _task_identity, [handle]
            ).result()
            assert resolved == {"k": 1}

    def test_parent_mutation_after_publish_does_not_change_snapshot(self):
        data = {"k": [1, 2]}
        with ProcessPoolExecutor(max_workers=1) as executor:
            handle = executor.publish_object(data)
            data["k"].append(3)
            data["new"] = 9

            _, snapshot = executor.submit_with_objects(
                _task_identity, [handle]
            ).result()
            assert snapshot == {"k": [1, 2]}

    def test_duplicate_handles_keep_position_and_share_local_object(self):
        with ProcessPoolExecutor(max_workers=1) as executor:
            handle = executor.publish_object({"k": 7})

            pid, same, a, b, extra = executor.submit_with_objects(
                _task_two_positions, [handle, handle], "z"
            ).result()
            assert same is True
            assert a == b == {"k": 7}
            assert extra == "z"

            same_ab, same_bc, value, extra = executor.submit_with_objects(
                _task_three_positions, [handle, handle, handle], "w"
            ).result()
            assert same_ab and same_bc
            assert value == 7
            assert extra == "w"

    def test_release_rejects_new_submissions_but_keeps_inflight_tasks(self):
        started_path = _new_flag_path()
        block_path = _new_flag_path()
        try:
            with ProcessPoolExecutor(max_workers=1) as executor:
                handle = executor.publish_object([1, 2, 3])
                key = handle._key
                future = executor.submit_with_objects(
                    _task_wait_flag, [handle], started_path, block_path
                )
                _wait_flag(started_path)

                handle.release()
                # Releasing twice is idempotent.
                handle.release()

                # The snapshot cannot be removed while the task is running.
                assert key in executor._objects.registry._entries

                with pytest.raises(ObjectHandleError):
                    executor.submit_with_objects(_task_identity, [handle])

                _set_flag(block_path)
                pid, value = future.result()
                assert value == [1, 2, 3]

                # The last in-flight user completed: the parent snapshot is
                # gone.
                assert key not in executor._objects.registry._entries
        finally:
            _cleanup_flags(started_path, block_path)

    def test_release_with_no_user_finalizes_immediately(self):
        with ProcessPoolExecutor(max_workers=1) as executor:
            handle = executor.publish_object("unused")
            key = handle._key
            handle.release()
            assert key not in executor._objects.registry._entries

    def test_cancelled_tasks_return_their_object_usages(self):
        started_path = _new_flag_path()
        block_path = _new_flag_path()
        try:
            with ProcessPoolExecutor(max_workers=1) as executor:
                handle = executor.publish_object({"v": 1})
                key = handle._key

                blocking = executor.submit_with_objects(
                    _task_wait_flag,
                    [handle],
                    started_path,
                    block_path,
                )
                _wait_flag(started_path)

                # Flood the pending queue: with one worker and a small call
                # queue, at least some of these futures must remain
                # cancellable.
                queued = [
                    executor.submit_with_objects(_task_identity, [handle])
                    for _ in range(8)
                ]
                cancelled = None
                for future in reversed(queued):
                    if future.cancel():
                        cancelled = future
                        break
                assert cancelled is not None

                handle.release()
                # The blocking task still holds one usage.
                assert key in executor._objects.registry._entries

                _set_flag(block_path)
                assert blocking.result()[1] == {"v": 1}
                for future in queued:
                    if future is cancelled:
                        with pytest.raises(CancelledError):
                            future.result()
                    else:
                        assert future.result()[1] == {"v": 1}

                # The cancelled usage was returned, so finalization can happen.
                assert key not in executor._objects.registry._entries
        finally:
            _cleanup_flags(started_path, block_path)

    def test_load_failure_only_fails_the_tasks_using_the_object(self):
        with ProcessPoolExecutor(max_workers=1) as executor:
            broken_handle = executor.publish_object(Unrebuildable())
            failed = executor.submit_with_objects(
                _task_identity, [broken_handle]
            )
            with pytest.raises(ObjectHandleError):
                failed.result()

            # The worker is still alive and runs ordinary tasks.
            assert executor.submit(_task_sum, [1, 2, 3]).result() == 6

            # It can also load another, healthy published object.
            good_handle = executor.publish_object("healthy")
            _, value = executor.submit_with_objects(
                _task_identity, [good_handle]
            ).result()
            assert value == "healthy"

    def test_worker_rebuild_restores_snapshot_from_parent(self):
        with ProcessPoolExecutor(max_workers=1, timeout=1) as executor:
            handle = executor.publish_object(
                CountedObject(42, "worker-rebuild")
            )
            pid1, rebuild1, value1 = executor.submit_with_objects(
                _task_counted, [handle]
            ).result()
            assert rebuild1 == 1
            assert value1 == 42

            # Wait for the idle worker to exit cleanly.
            deadline = time.time() + 15
            while time.time() < deadline and pid1 in executor._processes:
                time.sleep(0.1)
            assert pid1 not in executor._processes

            # The replacement worker starts fresh and restores the same
            # snapshot on its own, with exactly one new deserialization.
            pid2, rebuild2, value2 = executor.submit_with_objects(
                _task_counted, [handle]
            ).result()
            assert pid2 != pid1
            assert rebuild2 == 1
            assert value2 == 42

    def test_cross_executor_released_and_forged_handles_fail(self):
        with (
            ProcessPoolExecutor(max_workers=1) as executor,
            ProcessPoolExecutor(max_workers=1) as other,
        ):
            foreign_handle = other.publish_object([1])
            with pytest.raises(ObjectHandleError):
                executor.submit_with_objects(_task_identity, [foreign_handle])

            local_handle = executor.publish_object([2])
            local_handle.release()
            with pytest.raises(ObjectHandleError):
                executor.submit_with_objects(_task_identity, [local_handle])

            # Forged generation / object id, bound to the right executor.
            forged_key = _ObjectKey(
                executor._executor_id, object_id=10**9, generation=10**9
            )
            forged = ObjectHandle(None, forged_key)
            with pytest.raises(ObjectHandleError):
                executor.submit_with_objects(_task_identity, [forged])

            # Non-handle values are rejected explicitly.
            with pytest.raises(ObjectHandleError):
                executor.submit_with_objects(_task_identity, ["not-a-handle"])

    def test_pickling_failure_creates_no_handle(self):
        with ProcessPoolExecutor(max_workers=1) as executor:
            with pytest.raises(pickle.PicklingError):
                executor.publish_object(Unpicklable())
            assert executor._objects.registry._entries == {}

    def test_worker_load_requests_fail_explicitly_for_unknown_handles(self):
        with ProcessPoolExecutor(max_workers=1) as executor:
            registry = executor._objects.registry
            handle = executor.publish_object([1])
            key = handle._key
            fake_pid = 424242

            # Forged or unknown generation: the worker-side load request is
            # answered with an explicit error instead of a snapshot.
            forged_key = _ObjectKey(
                executor._executor_id, object_id=10**9, generation=10**9
            )
            payload, error = registry.request_load(fake_pid, forged_key)
            assert payload is None
            assert error is not None

            # A snapshot released before the worker loads it is rejected too.
            handle.release()
            payload, error = registry.request_load(fake_pid, key)
            assert payload is None
            assert error is not None

    def test_worker_holder_markers_are_pid_scoped_and_cleared(self):
        with ProcessPoolExecutor(max_workers=1) as executor:
            registry = executor._objects.registry
            handle = executor.publish_object([1])
            key = handle._key

            payload, error = registry.request_load(777, key)
            assert error is None and payload is not None
            assert 777 in registry._entries[key].holders

            # Simulating a worker exit drops its loaded markers only, so a
            # replacement pid can never be considered already loaded.
            registry.remove_holder(777)
            assert 777 not in registry._entries[key].holders

    def test_publish_uses_job_reducers_rules(self):
        with ProcessPoolExecutor(
            max_workers=1, job_reducers={SpecialReduced: _reduce_special}
        ) as executor:
            handle = executor.publish_object(SpecialReduced(1))
            assert (
                executor.submit_with_objects(
                    _task_special_x, [handle]
                ).result()
                == 1001
            )

    def test_shutdown_cleans_all_snapshots_and_references(self):
        executor = ProcessPoolExecutor(max_workers=1)
        handle = executor.publish_object([1])
        executor.submit_with_objects(_task_identity, [handle]).result()
        assert len(executor._objects.registry._entries) == 1
        assert len(executor._objects.response_writers) == 1

        executor.shutdown()
        assert executor._objects.registry._entries == {}
        assert executor._objects.response_writers == {}

        # Releasing after the shutdown cleanup is an idempotent no-op.
        handle.release()
        handle.release()

    def test_handle_is_opaque_and_picklable(self):
        with ProcessPoolExecutor(max_workers=1) as executor:
            handle = executor.publish_object([1])
            assert repr(handle) == "ObjectHandle(<opaque>)"
            restored = pickle.loads(pickle.dumps(handle))
            assert isinstance(restored, ObjectHandle)
            assert restored._key == handle._key

    def test_reusable_executor_supports_published_objects(self):
        executor = get_reusable_executor(max_workers=2)
        handle = executor.publish_object(CountedObject("r", "reusable"))
        try:
            pid, rebuild_count, value = executor.submit_with_objects(
                _task_counted, [handle]
            ).result()
            assert value == "r"
            assert rebuild_count == 1
        finally:
            handle.release()
