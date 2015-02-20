"""
Tools for emulating multiple dispatch.

"""

from .cache import CachedClass


class MultiClass(type):
    """Metaclass for dispatching instantiation to subclasses."""

    def __call__(cls, *args, **kw):
        """Dispatch instantiation based on the supplied arguments."""

        # Create instance if no subclasses
        if not cls.__subclasses__():
            return super().__call__(*args, **kw)

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

