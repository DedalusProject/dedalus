"""Dedalus testing module."""

import os
import pytest
import pathlib


file = pathlib.Path(__file__)
root = file.parent
testpath = str(root)

def base_cmd():
    workers = os.getenv("PYTEST_WORKERS", "auto")
    ignore_options = [f"--ignore={testpath}/test_spherical_ncc.py", f"--ignore={testpath}/test_cylinder_ncc.py"]
    xdist_options = [f"-n={workers}", "--dist=worksteal"]
    return ignore_options + xdist_options

def test(report=False):
    """Run tests."""
    cmd = base_cmd()
    cmd.extend(["--benchmark-disable"])
    if report:
        cmd.append("--junitxml=dedalus-test-junit.xml")
    return pytest.main(cmd + [testpath])

def bench():
    """Run benchmarks."""
    cmd = base_cmd()
    cmd.extend(["--benchmark-only"])
    return pytest.main(cmd + [testpath])

def cov():
    """Print test coverage."""
    cmd = base_cmd()
    cmd.extend(["--benchmark-disable", "--cov=dedalus.core"])
    return pytest.main(cmd + [testpath])
