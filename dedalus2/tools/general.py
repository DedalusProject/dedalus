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

