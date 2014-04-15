"""
Tools for caching computations.

"""

import types
from weakref import WeakValueDictionary
import numpy as np


class CachedAttribute:
    """Descriptor for building attributes during first access."""

    def __init__(self, method):

        # Parameters
        self.method = method
        self.__name__ = method.__name__
        self.__doc__ = method.__doc__

    def __get__(self, instance, owner):

        # Return self when accessed from class
        if instance is None:
            return self

        # Build attribute
        attribute = self.method(instance)

        # Set as instance attribute
        setattr(instance, self.__name__, attribute)

        return attribute


class CachedMethod:
    """Descriptor for caching method outputs during first call."""

    def __init__(self, method):

        # Parameters
        self.method = method
        self.__name__ = method.__name__
        self.__doc__ = method.__doc__
        self.cache = dict()

    def __get__(self, instance, owner):

        # Return self when accessed from class
        if instance is None:
            return self

        # Build new cached method and bind to instance
        new_cached_method = CachedMethod(self.method)
        bound_method = types.MethodType(new_cached_method, instance)

        # Set as instance method
        setattr(instance, self.__name__, bound_method)

        return bound_method

    def __call__(self, instance, *args):

        # Call method for new arguments
        if args not in self.cache:
            self.cache[args] = self.method(instance, *args)

        return self.cache[args]


class CachedFunction:
    """Decorator for caching function outputs during first call."""

    def __init__(self, function):

        self.function = function
        self.__name__ = function.__name__
        self.__doc__ = function.__doc__
        self.cache = dict()

    def __call__(self, *args):

        # Call function for new arguments
        if args not in self.cache:
            self.cache[args] = self.function(*args)

        return self.cache[args]


class CachedClass(type):
    """Metaclass for caching instantiation."""

    def __init__(cls, *args, **kw):

        # Perform regular class initialization from type
        type.__init__(cls, *args, **kw)
        # Cache instances using weakrefs
        cls._cache = WeakValueDictionary()

    def __call__(cls, *args, **kw):

        # Serialize arguments and keywords
        call = serialize_call(args, kw)
        # Instantiate for new call
        if call not in cls._cache:
            # Bind to local variable so weakref persists until return
            instance = type.__call__(cls, *args, **kw)
            cls._cache[call] = instance

        return cls._cache[call]


def serialize_call(args, kw):
    """Serialize call into standard form for cache key."""

    s_args = tuple(serialize(arg) for arg in args)
    s_kw = tuple((key, serialize(value)) for (key, value) in sorted(kw.items()))

    return (s_args, s_kw)


def serialize(arg):
    """Catch numpy arrays and replace with object repr / id."""

    if isinstance(arg, np.ndarray):
        return object.__repr__(arg)
    else:
        return arg

