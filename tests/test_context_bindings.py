"""Tests for the executor-level ``context_bindings`` configuration.

These tests check the end-to-end propagation of explicitly bound
:class:`contextvars.ContextVar` values from the submitting threads to the
worker processes. The worker-side functions always read the variables
directly from an importable module (see ``tests._context_binding_vars``),
so the assertions observe the real ContextVar values rather than an
internal transport dictionary.
"""

import threading
from pickle import PicklingError
from time import sleep

import pytest

from loky import ProcessPoolExecutor, get_reusable_executor
from loky.backend import get_context
from loky.process_executor import _CallItem, _normalize_context_bindings
from loky.reusable_executor import _ReusablePoolExecutor

from . import _context_binding_vars as cbv


# The workers resolve bindings by importing this module path, exactly like
# user code would.
MODULE = cbv.__name__


def binding(attr):
    return f"{MODULE}:{attr}"


# A representative configuration exercising every kind of test variable.
FULL_BINDINGS = {
    "shared_a": (MODULE, "shared_name_var_a"),
    "shared_b": binding("shared_name_var_b"),
    "no_default": (MODULE, "no_default_var"),
    "defaulted": (MODULE, "defaulted_var"),
}


@pytest.fixture
def make_executor():
    executors = []

    def _make(**kwargs):
        kwargs.setdefault("max_workers", 2)
        kwargs.setdefault("context", get_context("loky"))
        executor = ProcessPoolExecutor(**kwargs)
        executors.append(executor)
        return executor

    yield _make

    for executor in executors:
        executor.shutdown(
            wait=True, kill_workers=executor._flags.broken is not None
        )


class TestContextBindingsValidation:
    def test_invalid_configurations_raise_at_construction(self):
        # The configuration must be a mapping.
        with pytest.raises(TypeError):
            ProcessPoolExecutor(
                context_bindings=[("a", binding("no_default_var"))]
            )

        # Malformed "module:attribute" string.
        with pytest.raises(ValueError):
            ProcessPoolExecutor(
                context_bindings={"a": "tests._context_binding_vars"}
            )

        # Non-string logical name.
        with pytest.raises(TypeError):
            ProcessPoolExecutor(
                context_bindings={1: binding("no_default_var")}
            )

        # Bad tuple shape.
        with pytest.raises(TypeError):
            ProcessPoolExecutor(context_bindings={"a": (MODULE,)})

        # Unimportable module.
        with pytest.raises(ImportError):
            ProcessPoolExecutor(
                context_bindings={"a": "loky_not_a_module_xyz:var"}
            )

        # Importable module but missing attribute.
        with pytest.raises(AttributeError):
            ProcessPoolExecutor(
                context_bindings={"a": binding("does_not_exist")}
            )

        # Existing attribute that is not a ContextVar.
        with pytest.raises(TypeError):
            ProcessPoolExecutor(
                context_bindings={"a": binding("read_shared_name_a")}
            )

    def test_string_and_tuple_specs_are_equivalent(self):
        from_string = _normalize_context_bindings(
            {"a": binding("no_default_var")}
        )
        from_tuple = _normalize_context_bindings(
            {"a": (MODULE, "no_default_var")}
        )
        assert from_string == from_tuple

    def test_none_and_empty_dict_leave_legacy_mode(self):
        for config in (None, {}):
            executor = ProcessPoolExecutor(context_bindings=config)
            try:
                assert executor._context_bindings == {}
                assert executor._context_vars == {}
            finally:
                executor.shutdown(wait=True)


class TestContextBindingsPropagation:
    def test_same_name_distinct_contextvars_are_resolved_independently(
        self, make_executor
    ):
        # The two variables intentionally share the same ContextVar.name.
        assert cbv.shared_name_var_a.name == cbv.shared_name_var_b.name
        assert cbv.shared_name_var_a is not cbv.shared_name_var_b

        executor = make_executor(
            context_bindings={
                "shared_a": (MODULE, "shared_name_var_a"),
                "shared_b": (MODULE, "shared_name_var_b"),
            }
        )

        # Unset variables fall back to their own default in the workers.
        assert executor.submit(cbv.read_shared_name_a).result() == "default-a"
        assert executor.submit(cbv.read_shared_name_b).result() == "default-b"

        token_a = cbv.shared_name_var_a.set("value-a")
        token_b = cbv.shared_name_var_b.set("value-b")
        try:
            assert (
                executor.submit(cbv.read_shared_name_a).result() == "value-a"
            )
            assert (
                executor.submit(cbv.read_shared_name_b).result() == "value-b"
            )
        finally:
            cbv.shared_name_var_a.reset(token_a)
            cbv.shared_name_var_b.reset(token_b)

    def test_unset_none_and_defaulted_states(self, make_executor):
        executor = make_executor(
            context_bindings={
                "no_default": (MODULE, "no_default_var"),
                "defaulted": (MODULE, "defaulted_var"),
            }
        )

        # Unset variable without a default behaves as unset in the worker.
        assert executor.submit(cbv.read_no_default).result() == (
            "unset",
            None,
        )
        # An unset variable with a default follows its default rule.
        assert executor.submit(cbv.read_defaulted).result() == "the-default"

        # None is a regular, distinguishable value.
        token = cbv.no_default_var.set(None)
        try:
            assert executor.submit(cbv.read_no_default).result() == (
                "set",
                None,
            )
        finally:
            cbv.no_default_var.reset(token)

        token = cbv.defaulted_var.set("explicit-value")
        try:
            assert executor.submit(cbv.read_defaulted).result() == (
                "explicit-value"
            )
        finally:
            cbv.defaulted_var.reset(token)

        # Back to unset once the parent context is reset.
        assert executor.submit(cbv.read_no_default).result() == (
            "unset",
            None,
        )
        assert executor.submit(cbv.read_defaulted).result() == "the-default"

    def test_unregistered_variable_is_not_propagated(self, make_executor):
        executor = make_executor(
            context_bindings={"no_default": (MODULE, "no_default_var")}
        )
        token = cbv.unregistered_var.set("parent-only-value")
        try:
            # The worker must observe the variable default, never the value
            # set in the parent thread.
            assert (
                executor.submit(cbv.read_unregistered).result()
                == "unreg-default"
            )
        finally:
            cbv.unregistered_var.reset(token)

    def test_snapshot_taken_at_submit_with_delayed_serialization(
        self, make_executor
    ):
        from multiprocessing import Manager

        manager = Manager()
        try:
            event = manager.Event()
            executor = make_executor(
                max_workers=1,
                context_bindings={"no_default": (MODULE, "no_default_var")},
            )

            # Occupy the single worker so the second call item stays queued
            # and is only consumed after the parent changes its context.
            blocking = executor.submit(cbv.wait_on_event, event)
            token = cbv.no_default_var.set("snapshot-value")
            queued = executor.submit(cbv.read_no_default)

            # Reset the parent context before the queued task can run. Task
            # serialization itself is delayed to the queue feeder thread,
            # but the value was already captured at submit time.
            cbv.no_default_var.reset(token)
            assert (
                cbv.no_default_var.get("unset-in-parent") == "unset-in-parent"
            )

            event.set()
            assert blocking.result() == "released"
            # The task keeps the value captured at submit time.
            assert queued.result() == ("set", "snapshot-value")
        finally:
            manager.shutdown()

    def test_parent_set_and_reset_after_submit_do_not_change_task(
        self, make_executor
    ):
        executor = make_executor(
            context_bindings={"no_default": (MODULE, "no_default_var")}
        )
        token = cbv.no_default_var.set("first")
        try:
            future = executor.submit(cbv.read_no_default)
        finally:
            cbv.no_default_var.reset(token)
        second_token = cbv.no_default_var.set("second")
        try:
            assert future.result() == ("set", "first")
        finally:
            cbv.no_default_var.reset(second_token)

    def test_in_task_mutation_and_exception_do_not_leak(self, make_executor):
        executor = make_executor(max_workers=1, context_bindings=FULL_BINDINGS)
        token = cbv.no_default_var.set("original")
        try:
            with pytest.raises(ValueError, match="boom"):
                executor.submit(cbv.mutate_no_default_then_raise).result()

            # The pool is still healthy and the same worker reuses the
            # executor-level configuration without any residual value.
            assert executor._flags.broken is None
            assert executor.submit(cbv.read_no_default).result() == (
                "set",
                "original",
            )

            # An unset submission stays unset even though the previous task
            # mutated the variable inside its isolated Context.
        finally:
            cbv.no_default_var.reset(token)
        assert executor.submit(cbv.read_no_default).result() == ("unset", None)

    def test_initializer_value_does_not_leak_into_unset_binding(
        self, make_executor
    ):
        executor = make_executor(
            max_workers=1,
            initializer=cbv.initializer_set_no_default_var,
            context_bindings={"no_default": (MODULE, "no_default_var")},
        )

        # The initializer sets the variable in the worker root context, but a
        # binding that was unset in the parent must still behave as unset.
        assert executor.submit(cbv.read_no_default).result() == ("unset", None)

        token = cbv.no_default_var.set("from-parent")
        try:
            assert executor.submit(cbv.read_no_default).result() == (
                "set",
                "from-parent",
            )
        finally:
            cbv.no_default_var.reset(token)

        assert executor.submit(cbv.read_no_default).result() == ("unset", None)

    def test_concurrent_submitting_threads_keep_their_own_values(
        self, make_executor
    ):
        executor = make_executor(
            max_workers=1,
            context_bindings={"no_default": (MODULE, "no_default_var")},
        )
        start = threading.Event()
        errors = []
        results = {}
        n_tasks = 25

        def submit_from_thread(label):
            start.wait(10)
            token = cbv.no_default_var.set(label)
            try:
                futures = [
                    executor.submit(cbv.read_no_default)
                    for _ in range(n_tasks)
                ]
                results[label] = [future.result() for future in futures]
            except BaseException as exc:  # pragma: no cover - debugging
                errors.append(exc)
            finally:
                cbv.no_default_var.reset(token)

        threads = [
            threading.Thread(target=submit_from_thread, args=(f"thread-{i}",))
            for i in range(2)
        ]
        for thread in threads:
            thread.start()
        start.set()
        for thread in threads:
            thread.join(15)
            assert not thread.is_alive()

        assert not errors
        for label, values in results.items():
            assert values == [("set", label)] * n_tasks

    def test_unpicklable_value_fails_only_its_future(self, make_executor):
        executor = make_executor(
            max_workers=2,
            context_bindings={"no_default": (MODULE, "no_default_var")},
        )

        good_before = executor.submit(cbv.read_no_default)

        token = cbv.no_default_var.set(cbv.UnpicklableValue())
        try:
            bad_future = executor.submit(cbv.read_no_default)
        finally:
            cbv.no_default_var.reset(token)

        good_after = executor.submit(cbv.read_no_default)

        with pytest.raises(PicklingError):
            bad_future.result()

        # The healthy pool is not marked broken and the other tasks complete.
        assert executor._flags.broken is None
        assert good_before.result() == ("unset", None)
        assert good_after.result() == ("unset", None)
        assert executor.submit(cbv.get_pid).result() > 0

    def test_worker_rebuild_keeps_propagating_bindings(self, make_executor):
        executor = make_executor(
            max_workers=1,
            timeout=1,
            context_bindings={"no_default": (MODULE, "no_default_var")},
        )

        token = cbv.no_default_var.set("rebuild-value")
        try:
            first_pid, first_value = executor.submit(
                cbv.read_no_default_and_pid
            ).result()
            assert first_value == ("set", "rebuild-value")
        finally:
            cbv.no_default_var.reset(token)

        # Wait for the idle worker to exit and the manager to notice it.
        for _ in range(200):
            if not executor._processes:
                break
            sleep(0.1)
        assert not executor._processes

        token = cbv.no_default_var.set("after-rebuild")
        try:
            second_pid, second_value = executor.submit(
                cbv.read_no_default_and_pid
            ).result()
        finally:
            cbv.no_default_var.reset(token)

        assert second_pid != first_pid
        assert second_value == ("set", "after-rebuild")


class TestNoConfigLegacyMode:
    def test_call_items_carry_no_binding_by_default(self, make_executor):
        executor = make_executor(max_workers=1)
        item = _CallItem(0, cbv.get_pid, (), {})
        assert item.context_bindings is None
        assert executor.submit(cbv.get_pid).result() > 0

    def test_parent_context_is_not_propagated_without_bindings(
        self, make_executor
    ):
        executor = make_executor(max_workers=1)
        token = cbv.no_default_var.set("should-not-cross")
        try:
            assert executor.submit(cbv.read_no_default).result() == (
                "unset",
                None,
            )
            assert (
                executor.submit(cbv.read_defaulted).result() == "the-default"
            )
        finally:
            cbv.no_default_var.reset(token)


class TestReusableExecutorContextBindings:
    def teardown_method(self):
        # Leave the thread-local reusable executor cache in its default state
        # so the rest of the test suite is not affected.
        get_reusable_executor().shutdown(wait=True)

    def test_bindings_participate_in_reuse_compatibility(self):
        no_default_binding = {"no_default": (MODULE, "no_default_var")}
        defaulted_binding = {"defaulted": (MODULE, "defaulted_var")}

        first = get_reusable_executor(
            max_workers=2, context_bindings=no_default_binding
        )
        # Equivalent description (string instead of tuple) reuses the pool.
        same = get_reusable_executor(
            max_workers=2,
            context_bindings={"no_default": binding("no_default_var")},
        )
        assert same is first

        _, reused = _ReusablePoolExecutor.get_reusable_executor(
            max_workers=2, context_bindings=no_default_binding
        )
        assert reused

        # A different binding configuration forces a new executor.
        changed = get_reusable_executor(
            max_workers=2, context_bindings=defaulted_binding
        )
        assert changed is not first

        _, reused = _ReusablePoolExecutor.get_reusable_executor(
            max_workers=2, context_bindings=defaulted_binding
        )
        assert reused

        # Going back to the first configuration cannot reuse the old binding.
        back_again = get_reusable_executor(
            max_workers=2, context_bindings=no_default_binding
        )
        assert back_again is not changed

    def test_reusable_executor_propagates_values(self):
        executor = get_reusable_executor(
            max_workers=2,
            context_bindings={"no_default": (MODULE, "no_default_var")},
        )
        token = cbv.no_default_var.set("reusable-value")
        try:
            assert executor.submit(cbv.read_no_default).result() == (
                "set",
                "reusable-value",
            )
        finally:
            cbv.no_default_var.reset(token)
        assert executor.submit(cbv.read_no_default).result() == ("unset", None)

    def test_invalid_bindings_fail_before_reusing_any_pool(self):
        with pytest.raises(AttributeError):
            get_reusable_executor(
                max_workers=2,
                context_bindings={"a": binding("missing_attribute")},
            )
        # The default reusable executor is still usable afterwards.
        assert (
            get_reusable_executor(max_workers=2).submit(cbv.get_pid).result()
            > 0
        )
