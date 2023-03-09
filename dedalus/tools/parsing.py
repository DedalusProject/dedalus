"""Tools for equation parsing."""

import re

from .exceptions import SymbolicParsingError

TOP_LEVEL_EQUALS_REGEX = re.compile(r"^[^=]+=\s*")


def split_equation(equation):
    """
    Split equation string into LHS and RHS strings.
    Examples
    --------
    >>> split_equation('f(x, y=5) = x**2')
    ('f(x, y=5)', 'x**2')
    """
    # Find top-level equals sign using regular expression
    match = TOP_LEVEL_EQUALS_REGEX.match(equation)
    if not match:
        raise SymbolicParsingError("Equation contains no top-level equals signs.")
    lhs = match.group().strip()[:-1]
    rhs = equation[match.end():].strip()
    return lhs, rhs


def split_call(call):
    """
    Convert math-style function definitions into head and arguments.
    Examples
    --------
    >>> split_call('f(x, y)')
    ('f', ('x', 'y'))
    >>> split_call('f')
    ('f', ())
    """
    head, _, argstring = call.partition('(')
    if not argstring:
        return head, ()
    args = tuple(argstring.rstrip(')').split(','))
    return head, args


def lambdify_functions(call, result):
    """
    Convert math-style function definitions into lambda expressions.
    Pass other statements without modification.
    Examples
    --------
    >>> lambdify_functions('f(x, y)', 'x*y')
    ('f', 'lambda x,y: x*y')
    >>> lambdify_functions('f', 'a*b')
    ('f', 'a*b')
    """
    head, args = split_call(call)
    if args:
        # Build lambda expression
        argstring = ','.join(args)
        return head, 'lambda {}: {}'.format(argstring, result)
    else:
        # Return original rule
        return call, result
