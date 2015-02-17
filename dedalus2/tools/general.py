"""
Extended built-ins, etc.

"""

import collections
import re


class OrderedSet(collections.OrderedDict):
    """Ordered set based on uniqueness of dictionary keys."""

    def update(self, *args):

        for arg in args:
            for item in arg:
                self.add(item)

    def add(self, item):

        self[item] = None


def rev_enumerate(sequence):
    """Simple reversed enumerate."""

    n = len(sequence) - 1
    for element in reversed(sequence):
        yield n, element
        n -= 1


def natural_sort(iterable, reverse=False):
    """
    Sort alphanumeric strings naturally, i.e. with "1" before "10".
    Based on http://stackoverflow.com/a/4836734.

    """

    convert = lambda sub: int(sub) if sub.isdigit() else sub.lower()
    key = lambda item: [convert(sub) for sub in re.split('([0-9]+)', str(item))]

    return sorted(iterable, key=key, reverse=reverse)


def oscillate(iterable):
    """Oscillate forward and backward through an iterable."""

    forward = iterable
    backward = iterable[::-1][1:-1]  # Weird slicing to work with range
    while True:
        yield from forward
        yield from backward


def unify(objects):
    """
    Check if all objects in a collection are equal.
    If so, return one of them.  If not, raise.
    """
    for i, object in enumerate(objects):
        if i == 0:
            OBJECT = object
        else:
            if object != OBJECT:
                raise ValueError("Objects are not all equal.")
    return OBJECT
