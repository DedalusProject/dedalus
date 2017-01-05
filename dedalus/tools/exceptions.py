"""
Custom exception classes.

"""


class NonlinearOperatorError(Exception):
    """Exceptions for operands that fail linearity tests."""
    pass

class DependentOperatorError(Exception):
    """Exception for operands that fail independence tests."""
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

class SkipDispatchException(Exception):
    """Exceptions for shortcutting MultiClass dispatch."""

    def __init__(self, output):
        self.output = output

