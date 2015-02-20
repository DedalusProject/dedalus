"""
Custom exception classes.

"""


class NonlinearOperatorError(Exception):
    """Exceptions for nonlinear LHS terms."""
    pass

class SymbolicParsingError(Exception):
    """Exceptions for syntactic and mathematical problems in equations."""
    pass

class UnsupportedEquationError(Exception):
    """Exceptions for valid but unsupported equations."""
    pass

class UndefinedParityError(Exception):
    """Exceptions for data/operations with undefined parity."""
    pass
