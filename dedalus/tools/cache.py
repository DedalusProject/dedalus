"""
Tools for caching computations.

"""

import inspect
import types
from weakref import WeakValueDictionary
from collections import OrderedDict
from functools import partial


class CachedAttribute:
    """Descriptor for building attributes during first access."""

    def __init__(self, method):

        self.method = method
        self.__name__ = method.__name__
        self.__doc__ = method.__doc__

    def __get__(self, instance, owner):

        # Return self when accessed from class
        if instance is None:
            return self
        # Set method output as instance attribute
        # (overrides reference to descriptor from instance)
        attribute = self.method(instance)
        setattr(instance, self.__name__, attribute)

        return attribute


class CachedFunction:
    """Decorator for caching function outputs."""

    def __new__(cls, function=None, max_size=None):
        # Intercept calls without function assignment
        if function is None:
            # Return decorator with max_size partially applied
            return partial(cls, max_size=max_size)
        else:
            return object.__new__(cls)

    def __init__(self, function, max_size=None):
        # Function info
        self.function = function
        self.__name__ = function.__name__
        self.__doc__ = function.__doc__
        # Caching attributes
        self.cache = OrderedDict()
        self.max_size = max_size
        # Retrieve arg names and default kw
        argnames, _, _, defaults = inspect.getargspec(function)
        self.argnames = argnames
        if defaults:
            self.defaults = dict(zip(reversed(argnames), reversed(defaults)))
        else:
            self.defaults = dict()

    def __call__(self, *args, **kw):
        # Serialize call from provided/default args/kw
        call = serialize_call(args, kw, self.argnames, self.defaults)
        # Check cache for call
        if call in self.cache:
            return self.cache[call]
        else:
            if len(self.cache) == self.max_size:
                self.cache.popitem(last=False)
            self.cache[call] = result = self.function(*args, **kw)
            return result


class CachedMethod(CachedFunction):
    """Descriptor for caching method outputs."""
    # To do: find way remove instance self from args/call

    def __get__(self, instance, owner):
        # Return self when accessed from class
        if instance is None:
            return self
        # Build new cached method and bind to instance
        # (allows the cache to be deallocated with the instance)
        new_cached_method = CachedMethod(self.function, self.max_size)
        bound_method = types.MethodType(new_cached_method, instance)
        # Set bound method as instance attribute
        # (overrides reference to descriptor from instance)
        setattr(instance, self.__name__, bound_method)

        return bound_method


class CachedClass(type):
    """Metaclass for caching instantiation."""

    def __init__(cls, *args, **kw):

        # Perform regular class initialization from type
        super().__init__(*args, **kw)
        # Cache instances using weakrefs
        cls._cache = WeakValueDictionary()

        # Retrieve argument names and defaults, dropping 'self'
        (_, *argnames), _, _, defaults = inspect.getargspec(cls.__init__)
        cls._argnames = argnames
        if defaults:
            cls._defaults = dict(zip(reversed(argnames), reversed(defaults)))
        else:
            cls._defaults = dict()

    def __call__(cls, *args, **kw):

        # Serialize call from provided/default args/kw
        call = serialize_call(args, kw, cls._argnames, cls._defaults)
        # Instantiate for new call
        if call not in cls._cache:
            # Bind to local variable so weakref persists until return
            cls._cache[call] = instance = super().__call__(*args, **kw)

        return cls._cache[call]


def serialize_call(args, kw, argnames, defaults):
    """Serialize args/kw into cache key."""

    call = list(args)
    for name in argnames[len(args):]:
        try:
            arg = kw[name]
        except KeyError:
            arg = defaults[name]
        call.append(arg)

    return tuple(call)

