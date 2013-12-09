"""
Tools for emulating multiple dispatch.

"""


class MultiClass(type):
    """Metaclass for dispatching instantiation to subclasses."""

    def __call__(cls, *args, **kw):
        """Dispatch instantiation based on the supplied arguments."""

        # Dispatch if subclasses exist
        if cls.__subclasses__():

            # Find applicable subclasses
            passlist = []
            for sc in cls.__subclasses__():
                if sc._check_args(*args, **kw):
                    passlist.append(sc)

            if len(passlist) == 0:
                raise ValueError("No subclass supports the supplied arguments.")
            if len(passlist) > 1:
                raise ValueError("Degenerate subclasses found for the supplied arguments.")

            return passlist[0](*args, **kw)

        # Otherwise create instance
        else:
            return type.__call__(cls, *args, **kw)

