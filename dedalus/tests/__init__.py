"""Dedalus testing module."""

import pytest
import pathlib


file = pathlib.Path(__file__)
root = file.parent

def test():
    """Run tests."""
    return pytest.main(["-k", "not ncc", "--workers=auto", "--benchmark-disable", str(root)])

def bench():
    """Run benchmarks."""
    return pytest.main(["-k", "not ncc", "--workers=auto", "--benchmark-only", str(root)])

def cov():
    """Print test coverage."""
    return pytest.main(["-k", "not ncc", "--workers=auto", "--benchmark-disable", "--cov=dedalus.core", str(root)])
