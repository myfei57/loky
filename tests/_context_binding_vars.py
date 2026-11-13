"""Importable module holding the ContextVars used by the context_bindings tests.

The loky workers resolve context bindings by importing a module and reading
an attribute from it, so both the variables and the functions reading them
must live in a regular importable module. Defining them directly in the test
file would rely on cloudpickle by-value serialization instead of the explicit
module/attribute resolution that the feature is built around.
"""

import os
from pickle import PicklingError

from contextvars import ContextVar


# Two distinct ContextVar instances sharing the exact same ``name``. The
# binding identity must come from the explicit module/attribute path, never
# from ContextVar.name.
shared_name_var_a = ContextVar("shared_name", default="default-a")
shared_name_var_b = ContextVar("shared_name", default="default-b")

# A ContextVar without a default: reading it while unset raises LookupError.
no_default_var = ContextVar("no_default_var")

# A ContextVar owning a default value.
defaulted_var = ContextVar("defaulted_var", default="the-default")

# A variable that is never registered through context_bindings: the value it
# has in the parent process must never be propagated to the workers.
unregistered_var = ContextVar("unregistered_var", default="unreg-default")


def read_shared_name_a():
    """Read the variable directly from the module, as user code would."""
    return shared_name_var_a.get()


def read_shared_name_b():
    """Read the same-named but distinct variable directly from the module."""
    return shared_name_var_b.get()


def read_no_default():
    """Return a marker tuple so the unset state is observable in a result."""
    try:
        return ("set", no_default_var.get())
    except LookupError:
        return ("unset", None)


def read_defaulted():
    return defaulted_var.get()


def read_unregistered():
    return unregistered_var.get()


def read_no_default_and_pid():
    return os.getpid(), read_no_default()


def get_pid():
    return os.getpid()


def wait_on_event(event):
    """Block until the test releases the worker, keeping its task pending."""
    assert event.wait(30)
    return "released"


def mutate_no_default_then_raise():
    """Mutate the bound variable inside the task and then fail."""
    seen = no_default_var.get()
    no_default_var.set("mutated-in-worker")
    raise ValueError(f"boom after reading {seen!r}")


def initializer_set_no_default_var():
    """Pollute the worker root context from an initializer.

    A bound variable that was unset in the submitting context must still be
    unset in later tasks, so this value must never leak into them.
    """
    no_default_var.set("initializer-value")


class UnpicklableValue:
    """Value that cannot be serialized, used to exercise future-only failure."""

    def __reduce__(self):
        raise PicklingError("UnpicklableValue cannot be pickled")
