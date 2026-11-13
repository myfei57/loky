###############################################################################
# Publish-once, resolve-once-per-worker read-only objects
#
# A large read-only input can be serialized a single time in the executor
# parent process through ``ProcessPoolExecutor.publish_object``. The returned
# opaque ``ObjectHandle`` can then be passed in the explicit ``objects``
# argument of ``submit_with_objects``. Each worker process deserializes the
# snapshot at most once, caches the resulting local object, and reuses it for
# every subsequent task running in that worker.
#
# Handles are never resolved implicitly: a handle travelling inside the
# regular ``args``/``kwargs`` of a plain ``submit`` call is pickled like any
# other object and reaches the worker untouched.
#
# The contract is read-only: workers receive independent copies in their own
# address space and loky does not attempt to intercept in-place mutations, nor
# does it provide any form of shared mutable memory.
#
import collections
import itertools
import threading
import weakref
from dataclasses import dataclass

from multiprocessing.connection import wait

from .backend.queues import SimpleQueue
from .backend.reduction import loads


__all__ = ["ObjectHandle", "ObjectHandleError"]


class ObjectHandleError(RuntimeError):
    """Raised when an :class:`ObjectHandle` cannot be resolved or released.

    This covers handles created by another executor, already released or
    forged handles, as well as workers receiving a handle they know nothing
    about or failing to restore the published snapshot.
    """


@dataclass(frozen=True)
class _ObjectKey:
    """Immutable identity of a published object.

    It binds the executor identity (``executor_id``), a per-executor object
    id and the publish generation, so handles from another executor or
    replayed/forged generations cannot be confused with live snapshots.
    """

    executor_id: int
    object_id: int
    generation: int


def _rebuild_object_handle(key):
    # Reconstructed handles (for instance after a pickle round-trip) are not
    # bound to a live executor and can only be used as opaque key carriers.
    return ObjectHandle(None, key)


class ObjectHandle:
    """Opaque reference to a read-only object published by an executor.

    Instances are created by :meth:`ProcessPoolExecutor.publish_object` and
    are only valid for the executor that created them. They are picklable so
    they can travel to worker processes, but they are never resolved
    implicitly: only :meth:`ProcessPoolExecutor.submit_with_objects`
    interprets them, and only when listed in its explicit ``objects``
    argument.
    """

    __slots__ = ("_executor_ref", "_key")

    def __init__(self, executor, key):
        self._executor_ref = (
            weakref.ref(executor) if executor is not None else None
        )
        self._key = key

    def release(self):
        """Release the snapshots associated with this handle.

        Release is idempotent. New submissions using the handle are rejected
        once release has been requested, while tasks accepted before the
        release are still allowed to complete. The parent snapshot and the
        worker copies are only dropped once the last in-flight user has
        finished.
        """
        executor = (
            self._executor_ref() if self._executor_ref is not None else None
        )
        if executor is None:
            raise ObjectHandleError(
                "this ObjectHandle is not bound to a live executor and "
                "cannot be released"
            )
        executor._release_object(self._key)

    def __reduce__(self):
        return _rebuild_object_handle, (self._key,)

    def __repr__(self):
        return "ObjectHandle(<opaque>)"


@dataclass(frozen=True)
class _ObjectLoadRequest:
    """Worker -> parent: please send the snapshot for ``key``."""

    pid: int
    key: _ObjectKey


@dataclass(frozen=True)
class _ObjectLoadResponse:
    """Parent -> worker: snapshot bytes or an explicit load error."""

    key: _ObjectKey
    payload: bytes | None = None
    error: str | None = None


@dataclass(frozen=True)
class _ObjectReleaseCommand:
    """Parent -> worker: drop the local copy of ``key`` when convenient."""

    key: _ObjectKey


class _ObjectEntry:
    __slots__ = ("payload", "released", "in_flight", "holders")

    def __init__(self, payload):
        # Frozen serialized snapshot, produced once at publish time.
        self.payload = payload
        # Once True, new submissions using the handle are rejected.
        self.released = False
        # Number of accepted tasks (one count per handle occurrence) that
        # have not completed, been cancelled or been cleaned up yet.
        self.in_flight = 0
        # Pids of the workers that currently hold a deserialized copy.
        self.holders = set()


class _ObjectRegistry:
    """Parent-side bookkeeping of published snapshots."""

    def __init__(self, executor_id):
        self._executor_id = executor_id
        self._entries = {}
        self._lock = threading.RLock()
        self._object_ids = itertools.count()
        self._generations = itertools.count()

    def publish(self, payload, executor):
        """Register a freshly serialized snapshot and return a handle."""
        with self._lock:
            object_id = next(self._object_ids)
            generation = next(self._generations)
            key = _ObjectKey(self._executor_id, object_id, generation)
            self._entries[key] = _ObjectEntry(payload)
        return ObjectHandle(executor, key)

    def acquire(self, handles):
        """Validate handles and reserve one usage per handle occurrence.

        All handles are validated before any usage is reserved, so a failed
        validation never leaves partial reservations. Duplicate handles are
        kept in position and reserve one usage per occurrence.

        Returns the list of :class:`_ObjectKey` in the same order.
        """
        handles = tuple(handles)
        with self._lock:
            keys = []
            for handle in handles:
                if not isinstance(handle, ObjectHandle):
                    raise ObjectHandleError(
                        "expected an ObjectHandle in 'objects', got "
                        f"{type(handle).__name__}"
                    )
                key = handle._key
                if key.executor_id != self._executor_id:
                    raise ObjectHandleError(
                        "this ObjectHandle was created by another "
                        "ProcessPoolExecutor and cannot be used here"
                    )
                entry = self._entries.get(key)
                if entry is None:
                    raise ObjectHandleError(
                        "this ObjectHandle is unknown, has already been "
                        "released, or carries an invalid generation"
                    )
                if entry.released:
                    raise ObjectHandleError(
                        "this ObjectHandle has been released and cannot be "
                        "used for new submissions"
                    )
                keys.append(key)

            for key in keys:
                self._entries[key].in_flight += 1
            return keys

    def return_usages(self, keys):
        """Give back usage reservations after a task ended or was cancelled.

        Returns the list of ``(key, holder_pids)`` entries whose parent
        snapshot can now be dropped because release was requested and no
        in-flight user remains.
        """
        finalized = []
        with self._lock:
            for key in keys:
                entry = self._entries.get(key)
                if entry is None:
                    continue
                entry.in_flight -= 1
                if entry.released and entry.in_flight <= 0:
                    self._entries.pop(key, None)
                    holders = set(entry.holders)
                    entry.holders.clear()
                    entry.payload = None
                    finalized.append((key, holders))
        return finalized

    def release(self, key):
        """Mark a snapshot as released; finalize immediately if unused."""
        finalized = []
        with self._lock:
            entry = self._entries.get(key)
            # Idempotent: unknown or already released handles are no-ops.
            if entry is None or entry.released:
                return finalized
            entry.released = True
            if entry.in_flight <= 0:
                self._entries.pop(key, None)
                holders = set(entry.holders)
                entry.holders.clear()
                entry.payload = None
                finalized.append((key, holders))
        return finalized

    def request_load(self, pid, key):
        """Resolve a worker load request.

        Returns ``(payload, error_message)``: exactly one of the two values
        is ``None``. Registering the worker as a holder here, in the parent,
        keeps the loaded markers tied to the actual worker pid, so a recycled
        worker (new pid, empty cache) is never mistaken for an already
        loaded one.
        """
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None, (
                    "worker received an unknown ObjectHandle: no published "
                    "object matches this handle"
                )
            if entry.released:
                return None, (
                    "the ObjectHandle was released before the worker could "
                    "load the published object"
                )
            entry.holders.add(pid)
            return entry.payload, None

    def remove_holder(self, pid):
        """Forget every copy held by a worker that just exited."""
        with self._lock:
            for entry in self._entries.values():
                entry.holders.discard(pid)

    def clear(self):
        """Drop every snapshot and reference (shutdown / broken cleanup)."""
        with self._lock:
            for entry in self._entries.values():
                entry.payload = None
                entry.holders.clear()
            self._entries.clear()


class _ControlWakeup:
    """Small one-way pipe used to wake up the object control thread."""

    def __init__(self, context):
        self._reader, self._writer = context.Pipe(duplex=False)
        self._closed = False

    def wakeup(self):
        if self._closed:
            return
        try:
            self._writer.send_bytes(b"")
        except OSError:
            pass

    def clear(self):
        if self._closed:
            return
        while self._reader.poll():
            self._reader.recv_bytes()

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._writer.close()
        finally:
            self._reader.close()


class _ObjectControlThread(threading.Thread):
    """Parent-side thread serving worker object load and release messages.

    It runs independently of the executor manager thread so that
    transferring a potentially large snapshot to a worker never blocks task
    dispatch or result management.
    """

    def __init__(self, infrastructure):
        super().__init__(name="ObjectControlThread", daemon=True)
        self._infra = infrastructure
        self._shutdown = threading.Event()
        self._release_queue = collections.deque()
        self._release_lock = threading.Lock()

    def dispatch_releases(self, finalized):
        """Queue worker release commands for finalized snapshots."""
        with self._release_lock:
            self._release_queue.extend(finalized)
        self._infra.wakeup.wakeup()

    def stop(self):
        self._shutdown.set()
        self._infra.wakeup.wakeup()

    def run(self):
        infra = self._infra
        reader = infra.request_queue._reader
        wakeup_reader = infra.wakeup._reader
        while not self._shutdown.is_set():
            try:
                ready = wait((reader, wakeup_reader))
            except OSError:
                # The queues are being closed during interpreter shutdown.
                break
            if wakeup_reader in ready:
                infra.wakeup.clear()
            if reader in ready:
                # SimpleQueue.get() is blocking and takes no timeout on
                # modern Python: wait() above guarantees that one request is
                # available, and it level-triggers again for the next ones.
                try:
                    request = infra.request_queue.get()
                except OSError:
                    break
                self._handle_request(request)
            self._drain_releases()
        self._drain_releases()

    def _handle_request(self, request):
        infra = self._infra
        try:
            payload, error = infra.registry.request_load(
                request.pid, request.key
            )
            response = _ObjectLoadResponse(
                request.key, payload=payload, error=error
            )
        except BaseException as e:  # pragma: no cover - defensive
            response = _ObjectLoadResponse(request.key, error=repr(e))
        writer = infra.get_writer(request.pid)
        if writer is None:
            # The worker exited before its request could be answered; its
            # cache died with it, nothing else to do.
            return
        try:
            writer.send(response)
        except OSError:
            # The worker vanished while the response was being written.
            pass

    def _drain_releases(self):
        with self._release_lock:
            pending = list(self._release_queue)
            self._release_queue.clear()
        for key, holders in pending:
            for pid in holders:
                writer = self._infra.get_writer(pid)
                if writer is None:
                    continue
                try:
                    writer.send(_ObjectReleaseCommand(key))
                except OSError:
                    # The holder worker exited; its copy is gone anyway.
                    continue


class _ObjectInfrastructure:
    """All parent-side objects backing the publish/submit_with_objects API."""

    def __init__(self, executor_id, context):
        self.executor_id = executor_id
        self.registry = _ObjectRegistry(executor_id)
        # All workers share a single request queue (worker -> parent); the
        # requests carry the worker pid and answers go through per-worker
        # response pipes.
        self.request_queue = SimpleQueue(ctx=context)
        self.response_writers = {}
        self.io_lock = threading.Lock()
        self.wakeup = _ControlWakeup(context)
        self.control_thread = None

    def ensure_control_thread(self):
        if self.control_thread is None:
            thread = _ObjectControlThread(self)
            self.control_thread = thread
            thread.start()
        return self.control_thread

    def add_writer(self, pid, writer):
        with self.io_lock:
            self.response_writers[pid] = writer

    def pop_writer(self, pid):
        with self.io_lock:
            return self.response_writers.pop(pid, None)

    def get_writer(self, pid):
        with self.io_lock:
            return self.response_writers.get(pid)

    def dispatch_finalized(self, finalized):
        if finalized and self.control_thread is not None:
            self.control_thread.dispatch_releases(finalized)

    def shutdown(self):
        """Stop the control thread and release every pipe and snapshot."""
        thread = self.control_thread
        if thread is not None:
            thread.stop()
            thread.join(timeout=5)
            self.control_thread = None
        with self.io_lock:
            writers = list(self.response_writers.values())
            self.response_writers.clear()
        for writer in writers:
            try:
                writer.close()
            except OSError:
                pass
        self.registry.clear()
        try:
            self.request_queue.close()
        except OSError:
            pass
        self.wakeup.close()


class _WorkerObjectCache:
    """Per-process cache of published objects.

    A worker deserializes each snapshot at most once: subsequent tasks and
    duplicate handles inside a single task reuse the exact same local object.
    The cache lives in the worker process only, so recycled workers start
    with an empty cache and restore snapshots from the parent on demand.
    """

    def __init__(self, pid, request_queue, response_reader):
        self._pid = pid
        self._request_queue = request_queue
        self._reader = response_reader
        self._cache = {}

    def process_release_commands(self):
        # Release commands are only processed between tasks, so a snapshot
        # can never be removed while a task is still using it.
        while self._reader.poll():
            message = self._reader.recv()
            if isinstance(message, _ObjectReleaseCommand):
                self._cache.pop(message.key, None)

    def resolve(self, keys):
        """Return the local objects for ``keys``, in order.

        Duplicate keys map to the exact same local object while keeping
        their position in the returned list.
        """
        self.process_release_commands()
        resolved = []
        for key in keys:
            if key not in self._cache:
                self._cache[key] = self._load(key)
            resolved.append(self._cache[key])
        return resolved

    def _load(self, key):
        self._request_queue.put(_ObjectLoadRequest(self._pid, key))
        while True:
            message = self._reader.recv()
            if isinstance(message, _ObjectReleaseCommand):
                self._cache.pop(message.key, None)
                continue
            if (
                not isinstance(message, _ObjectLoadResponse)
                or message.key != key
            ):
                # Defensive: ignore stale messages unrelated to this request.
                continue
            if message.error is not None:
                # Failing to load a snapshot only fails the tasks that use
                # it: the exception propagates through the task result and
                # the worker keeps running other tasks normally.
                raise ObjectHandleError(message.error)
            try:
                return loads(message.payload)
            except BaseException as e:
                raise ObjectHandleError(
                    "failed to deserialize the published object for "
                    f"{key!r}: {e!r}"
                ) from e
