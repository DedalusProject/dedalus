

import functools


class CachedAttribute:
    """Descriptor for building attributes during first access."""

    def __init__(self, method):

        # Parameters
        self.method = method
        self.__doc__ = method.__doc__

    def __get__(self, instance, owner):

        # Return self when accessed from class
        if instance is None:
            return self

        # Build attribute
        attribute = self.method(instance)

        # Cache attribute (overwriting descriptor)
        setattr(instance, self.method.__name__, attribute)

        return attribute


class CachedMethod:
    """Descriptor for caching method outputs during first call."""

    def __init__(self, method):

        # Parameters
        self.method = method
        self.__doc__ = method.__doc__
        self.cache = dict()

    def __call__(self, instance, *args):

        # Call method for new arguments
        if args not in self.cache:
            self.cache[args] = self.method(instance, *args)

        return self.cache[args]

    def __get__(self, instance, owner):

        # Return self when accessed from class
        if instance is None:
            return self

        # Return partial call (with access to 'instance')
        return functools.partial(self.__call__, instance)

