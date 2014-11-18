"""
Custom exception classes.

"""


class SymbolicParsingError(Exception):
    """Exceptions for syntactic and mathematical problems in equations."""
    pass

class UnsupportedEquationError(Exception):
    """Exceptions for valid but unsupported equations."""
    pass
