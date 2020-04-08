"""
Tools for emulating multiple dispatch.

"""

from .cache import CachedClass


class SkipDispatchException(Exception):
    """Exceptions for shortcutting MultiClass dispatch."""

    def __init__(self, output):
        self.output = output


class SkipDispatch(type):
    """Metclass for skipping dispatch based on arguments."""

    def __call__(cls, *args, **kw):
        try:
            args, kw = cls.__dispatch__(*args, **kw)
            return super().__call__(*args, **kw)
        except SkipDispatchException as skip:
            return skip.output

    def __dispatch__(cls, *args, **kw):
        """Preprocess args or skip dispatch."""
        return args, kw


class MultiClass(SkipDispatch):
    """Metaclass for dispatching instantiation to subclasses."""

    def __call__(cls, *args, **kw):
        """Dispatch instantiation based on the supplied arguments."""

        # Create instance if no subclasses
        if not cls.__subclasses__():
            # Check arguments
            if cls._check_args(*args, **kw):
                return super().__call__(*args, **kw)
            else:
                raise TypeError("Provided types do not pass dispatch check.")

        # Preprocess arguments and keywords
        args, kw = cls._preprocess_args(*args, **kw)

        # Find applicable subclasses
        passlist = []
        for subclass in cls.__subclasses__():
            if subclass._check_args(*args, **kw):
                passlist.append(subclass)

        if len(passlist) == 0:
            return NotImplemented
        elif len(passlist) > 1:
            raise ValueError("Degenerate subclasses of {} found for the supplied arguments: {}, {}".format(cls, args, kw))
        else:
            subclass, = passlist

        # Postprocess arguments and keywords
        args, kw = subclass._postprocess_args(*args, **kw)

        return subclass(*args, **kw)

    def _preprocess_args(cls, *args, **kw):
        """Process arguments and keywords prior to checking dispatch."""
        return args, kw

    def _check_args(cls, *args, **kw):
        """Check arguments and keywords to determine proper subclass for dispatch."""
        return True

    def _postprocess_args(cls, *args, **kw):
        """Process arguments and keywords after checking dispatch."""
        return args, kw


class CachedMultiClass(MultiClass, CachedClass):
    """Metaclass for dispatching and caching instantiantation to subclasses."""
    pass

