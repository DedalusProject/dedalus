"""
Tools for emulating multiple dispatch.

"""

from .cache import CachedClass
from .exceptions import SkipDispatchException


class MultiClass(type):
    """Metaclass for dispatching instantiation to subclasses."""

    def __call__(cls, *args, **kw):
        """Dispatch instantiation based on the supplied arguments."""
        try:
            # Preprocess arguments and keywords
            args, kw = cls._preprocess_args(*args, **kw)
            # Create instance if no subclasses
            subclasses = [sc for sc in cls.__subclasses__() if not getattr(sc, "stop_dispatch", False)]
            if not subclasses:
                # Check arguments
                if cls._check_args(*args, **kw):
                    return super().__call__(*args, **kw)
                else:
                    raise TypeError("Provided types do not pass dispatch check.")
            # Find applicable subclasses
            passlist = []
            for subclass in subclasses:
                if subclass._check_args(*args, **kw):
                    passlist.append(subclass)
        except SkipDispatchException as exception:
            return exception.output

        if len(passlist) == 0:
            raise NotImplementedError("No subclasses of {} found for the supplied arguments: {}, {}".format(cls, args, kw))
        elif len(passlist) > 1:
            raise ValueError("Degenerate subclasses of {} found for the supplied arguments {}, {}: {}".format(cls, args, kw, passlist))
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
        raise NotImplementedError()

    def _postprocess_args(cls, *args, **kw):
        """Process arguments and keywords after checking dispatch."""
        return args, kw


class CachedMultiClass(MultiClass, CachedClass):
    """Metaclass for dispatching and caching instantiantation to subclasses."""
    pass

