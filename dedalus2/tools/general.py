

import types


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

        # Parameters
        self.function = function
        self.__name__ = function.__name__
        self.__doc__ = function.__doc__
        self.cache = dict()

    def __call__(self, *args):

        # Call function for new arguments
        if args not in self.cache:
            self.cache[args] = self.function(*args)

        return self.cache[args]

