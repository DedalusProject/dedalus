"""
Tools for caching computations.

"""

import inspect
import types
from weakref import WeakValueDictionary
import numpy as np


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
        attribute = self.method(instance)
        setattr(instance, self.__name__, attribute)

        return attribute


class CachedFunction:
    """Decorator for caching function outputs, hashing numpy arrays by id."""

    def __init__(self, function):

        self.function = function
        self.__name__ = function.__name__
        self.__doc__ = function.__doc__
        self.cache = dict()

        # Retrieve argument names and defaults
        argnames, _, _, defaults = inspect.getargspec(function)
        self.argnames = argnames
        if defaults:
            self.defaults = dict(zip(reversed(argnames), reversed(defaults)))
        else:
            self.defaults = dict()

    def __call__(self, *args, **kw):

        # Serialize call from provided/default args/kw
        call = serialize_call(args, kw, self.argnames, self.defaults)
        # Evaluate function for new call
        if call not in self.cache:
            self.cache[call] = self.function(*args, **kw)

        return self.cache[call]


class CachedMethod(CachedFunction):
    """Descriptor for caching method outputs, hashing numpy arrays by id."""

    def __get__(self, instance, owner):

        # Return self when accessed from class
        if instance is None:
            return self
        # Build new cached method and bind to instance
        # This allows the cache to be deallocated with the instance
        new_cached_method = CachedMethod(self.function)
        bound_method = types.MethodType(new_cached_method, instance)
        # Override reference to self (class descriptor) from instance
        setattr(instance, self.__name__, bound_method)

        return bound_method


class CachedClass(type):
    """Metaclass for caching instantiation, hashing numpy arrays by id."""

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
    """Serialize args/kw into cache key, hashing numpy arrays by id."""

    call = list(serialize(arg) for arg in args)
    for name in argnames[len(args):]:
        try:
            arg = kw[name]
        except KeyError:
            arg = defaults[name]
        call.append(serialize(arg))

    return tuple(call)


def serialize(arg):
    """Catch numpy arrays and replace with object repr / id."""

    if isinstance(arg, np.ndarray):
        return object.__repr__(arg)
    else:
        return arg

