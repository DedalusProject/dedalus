"""
Extended built-ins, etc.

"""

import types
import collections


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

