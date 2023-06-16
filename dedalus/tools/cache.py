"""
Tools for caching computations.

"""

import numpy as np
import inspect
import types
from weakref import WeakValueDictionary
from collections import OrderedDict
from functools import partial


class CachedAttribute:
    """Descriptor for building attributes during first access."""

    def __init__(self, method):
        self.method = method
        # Copy name and docstring
        self.__name__ = method.__name__
        self.__doc__ = method.__doc__

    def __get__(self, instance, owner):
        if instance is None:
            # Return self when accessed from not from instance
            return self
        else:
            # Set method output as instance attribute
            # (overrides reference to descriptor from instance)
            attribute = self.method(instance)
            setattr(instance, self.__name__, attribute)
            return attribute

    def __call__(self, *args, **kw):
        # Pass through calls when accessed not from instance
        return self.method(*args, **kw)


class CachedFunction:
    """Decorator for caching function outputs."""

    def __new__(cls, function=None, max_size=np.inf):
        # Intercept calls without function assignment
        if function is None:
            # Return decorator with max_size partially applied
            return partial(cls, max_size=max_size)
        else:
            return object.__new__(cls)

    def __init__(self, function, max_size=np.inf):
        # Function info
        self.function = function
        self.__name__ = function.__name__
        self.__doc__ = function.__doc__
        # Caching attributes
        self.cache = OrderedDict()
        self.max_size = max_size
        # Retrieve arg names and default kw
        argnames, _, _, defaults, _, _, _ = inspect.getfullargspec(function)
        self.argnames = argnames
        if defaults:
            self.defaults = dict(zip(reversed(argnames), reversed(defaults)))
        else:
            self.defaults = dict()

    def __call__(self, *args, **kw):
        # First try direct call signature
        direct_call = (args, tuple(kw.items()))
        try:
            return self.cache[direct_call]
        except KeyError:
            pass
        # Then try full resolved call
        resolved_call = serialize_call(args, kw, self.argnames, self.defaults)
        try:
            result = self.cache[resolved_call]
            # Add direct call key to avoid resolving it in subsequent calls
            self.cache[direct_call] = result
            return result
        except KeyError:
            pass
        # Reduce cache size if necessary
        while len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        # Compute and cache result
        result = self.function(*args, **kw)
        self.cache[direct_call] = result
        self.cache[resolved_call] = result
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
        cls._instance_cache = WeakValueDictionary()
        # Store class signature
        cls._signature = inspect.signature(cls.__init__)

    def __call__(cls, *args, **kw):
        # Preprocess arguments and keywords
        args, kw = cls._preprocess_args(*args, **kw)
        # Build call from class signature
        call = cls._signature.bind(None, *args, **kw)
        call.apply_defaults()
        # Preprocess call with all arguments except self
        if call.kwargs:
            raise ValueError("Required keyword arguments not supported.")
        full_args = cls._preprocess_cache_args(*call.args[1:])
        # Instantiate for new call
        if full_args not in cls._instance_cache:
            # Bind to local variable so weakref persists until return
            cls._instance_cache[full_args] = instance = super().__call__(*full_args)
        return cls._instance_cache[full_args]

    def _preprocess_args(cls, *args, **kw):
        """Process arguments and keywords prior to checking cache."""
        return args, kw

    @classmethod
    def _preprocess_cache_args(cls, *args):
        """Process call prior to checking cache."""
        return args
        # call = cls._signature.bind(None, *args, **kw)
        # call.apply_defaults()
        # if call.kwargs:
        #     raise ValueError("Required keyword arguments not supported.")
        # return call.args[1:]


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

