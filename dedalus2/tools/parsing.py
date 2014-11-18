"""Tools for equation parsing."""

import re

from .exceptions import SymbolicParsingError


def split_equation(equation):
    """
    Split equation string into LHS and RHS strings.

    Examples
    --------
    >>> split_equation('f(x, y=5) = x**2')
    ('f(x, y=5)', 'x**2')

    """
    # Find top-level equals signs by tracking parenthetical level
    # (to avoid capturing equals signs in keyword assignments)
    parentheses = 0
    top_level_equals = []
    for i, character in enumerate(equation):
        if character == '(':
            parentheses += 1
        elif character == ')':
            parentheses -= 1
        elif (character == '=') and (parentheses == 0):
            top_level_equals.append(i)
    # Raise if there isn't exactly one top-level equals sign
    if len(top_level_equals) == 0:
        raise SymbolicParsingError("Equation contains no top-level equals signs.")
    elif len(top_level_equals) > 1:
        raise SymbolicParsingError("Equation contains multiple top-level equals signs.")
    # Return sides
    i, = top_level_equals
    return equation[:i].strip(), equation[i+1:].strip()


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
    # Check if signature matches a function call
    match = re.match('(.+)\((.*)\)', call)
    # Return head and arguments
    if match:
        head, argstring = match.groups()
        args = tuple(argstring.replace(' ','').split(','))
        return head, args
    else:
        return call, ()


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


