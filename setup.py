"""
Setup script for Dedalus.
The following environment variables may be set:

    MPI_PATH
        Path to MPI installation prefix
    FFTW_PATH
        Path to FFTW installation prefix
    FFTW_STATIC
        Binary flag to statically link FFTW (default: 0)
    CYTHON_PROFILE
        Binary flag to enable profiling within cython modules (default: 0)

"""

import setuptools
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
from distutils.util import strtobool
from functools import cache
import numpy as np
import mpi4py
import os
import sys
import glob
import pathlib
import tarfile
import codecs


# Helper functions
def bool_env(name, unset=False):
    if var := os.getenv(name):
        return bool(strtobool(var))
    else:
        return unset

def get_env(name):
    if var := os.getenv(name):
        print(f"  Found env var {name} = {var}")
        return var

def get_library_prefix(name):
    """Get library prefix for a given library name."""
    name = name.lower()
    NAME = name.upper()
    # Check for environment variable
    env_vars = [f"{NAME}_PATH", f"{NAME}_PREFIX"]
    for env_var in env_vars:
        if path := get_env(env_var):
            return path
    # Check likely places
    places = [os.getenv("CONDA_PREFIX"),
              "/usr/lib/openmpi", # MPI under Ubuntu
              "/usr",
              "/usr/local",
              os.path.expanduser("~/build")]
    for place in places:
        if place:
            place_lib = os.path.join(place, "lib")
            guess = os.path.join(place_lib, f"*{name}*")
            if glob.glob(guess):
                print(f"  Found matching library in {place_lib}")
                return place
    print(f"  Cannot find {NAME}_PATH, {NAME}_PREFIX, or libraries matching {name}.")
    print(f"  If {name} isn't in your LD_LIBRARY_PATH, compilation will likely fail.")

@cache
def get_include(name):
    print(f"Looking for {name} include path:")
    if path := get_env(f"{name.upper()}_INCLUDE_PATH"):
        return path
    if prefix := get_library_prefix(name):
        return os.path.join(prefix, "include")

@cache
def get_lib(name):
    print(f"Looking for {name} library path:")
    if path := get_env(f"{name.upper()}_LIBRARY_PATH"):
        return path
    if prefix := get_library_prefix(name):
        return os.path.join(prefix, "lib")

def get_version(rel_path):
    """Read version from a file via text parsing, following PyPA guide."""
    def read(rel_path):
        here = os.path.abspath(os.path.dirname(__file__))
        with codecs.open(os.path.join(here, rel_path), "r") as fp:
            return fp.read()
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

# Configuratin
INCLUDE_MPI = True
INCLUDE_FFTW = True
FFTW_STATIC = bool_env("FFTW_STATIC", unset=False)
CYTHON_PROFILE = bool_env("CYTHON_PROFILE", unset=False)
print()
print("Configuration:")
print(f"  INCLUDE_MPI = {INCLUDE_MPI}")
print(f"  INCLUDE_FFTW = {INCLUDE_FFTW}")
print(f"  FFTW_STATIC = {FFTW_STATIC}")
print(f"  CYTHON_PROFILE = {CYTHON_PROFILE}")
print()

# C-dependency paths for extension compilation and linking.
include_dirs = [np.get_include()]
libraries = ["m"]
library_dirs = []
if INCLUDE_MPI:
    include_dirs += [mpi4py.get_include(), get_include("mpi")]
    libraries += ["mpi"]
    library_dirs += [get_lib("mpi")]
if INCLUDE_FFTW:
    include_dirs += ["dedalus/libraries/fftw/", get_include("fftw")]
    libraries += ["fftw3_mpi", "fftw3"]
    library_dirs += [get_lib("fftw")]
include_dirs = [dir for dir in include_dirs if dir is not None]
library_dirs = [dir for dir in library_dirs if dir is not None]

# Warning supression
extra_compile_args = [
    "-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
    "-Wno-unused-function",
    "-fopenmp"]

# Optionally set static linking for FFTW
extra_link_args = ["-fopenmp"]
if INCLUDE_FFTW and FFTW_STATIC:
    print("Statically linking FFTW")
    fftw_lib_path = get_lib("fftw")
    clang_extra_link_args = [
        "-Wl,-force_load",
        "{}/libfftw3.a".format(fftw_lib_path),
        "{}/libfftw3_mpi.a".format(fftw_lib_path),
        "-Wl,-noall_load"]
    gcc_extra_link_args = [
        "-Xlinker", "-Bsymbolic",
        "-Wl,--whole-archive",
        "{}/libfftw3.a".format(fftw_lib_path),
        "{}/libfftw3_mpi.a".format(fftw_lib_path),
        "-Wl,--no-whole-archive"]
    # Choose linker flags based on CC
    CC = os.getenv("CC")
    if "clang" in CC:
        print("CC set to clang; using clang linker flags")
        extra_link_args.extend(clang_extra_link_args)
    elif CC:
        print("CC set to {}; using gcc linker flags".format(CC))
        extra_link_args.extend(gcc_extra_link_args)
    elif sys.platform == "darwin":
        print("CC not set; defaulting to clang linker flags")
        extra_link_args.extend(clang_extra_link_args)
    else:
        print("CC not set; defaulting to gcc linker flags")
        extra_link_args.extend(gcc_extra_link_args)

# Extension objects for cython
pyx_modules = [
    "dedalus.core.transposes",
    "dedalus.tools.linalg",
    "dedalus.libraries.fftw.fftw_wrappers",
    "dedalus.libraries.spin_recombination"]
extensions = []
for name in pyx_modules:
    extensions.append(
        Extension(
            name=name,
            sources=[name.replace(".", "/") + ".pyx"],
            include_dirs=include_dirs,
            libraries=libraries,
            library_dirs=library_dirs,
            runtime_library_dirs=library_dirs,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args))

# Runtime requirements
install_requires = [
    "docopt",
    "h5py >= 3.0.0",
    "matplotlib >= 3.7.0",
    "mpi4py >= 2.0.0",
    "numexpr",
    "numpy >= 1.20.0",
    "py",
    "pytest",
    "pytest-benchmark",
    "pytest-cov",
    "pytest-xdist",
    "scipy >= 1.4.0",
    "xarray"]

# Grab long_description from README
with open("README.md") as f:
    long_description = f.read()

# Cython directives
compiler_directives = {}
compiler_directives["language_level"] = 3
if CYTHON_PROFILE:
    compiler_directives["profile"] = True

# Override build command to pack up examples
from distutils.command.build import build as _build
class build(_build):
    def run(self):
        # Create tar file with example scripts
        with tarfile.open("dedalus/examples.tar.gz", mode="w:gz") as archive:
            examples_dir = pathlib.Path("examples")
            for file in examples_dir.glob("**/*.py"):
                archive.add(str(file), str(file.relative_to(examples_dir)))
        # Run the original build command
        _build.run(self)

# Setup
print()
setup(
    name="dedalus",
    version=get_version("dedalus/__init__.py"),
    author="Keaton J. Burns",
    author_email="keaton.burns@gmail.com",
    description="A flexible framework for solving PDEs with modern spectral methods.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://dedalus-project.org",
    classifiers=["Programming Language :: Python :: 3"],
    python_requires=">=3.9",
    install_requires=install_requires,
    license="GPL3",
    packages=setuptools.find_packages(),
    package_data={"": ["dedalus.cfg", "examples.tar.gz"]},
    ext_modules=cythonize(extensions, compiler_directives=compiler_directives),
    cmdclass={"build": build},
    entry_points={"xarray.backends": ["dedalus=dedalus.tools.post:DedalusXarrayBackend"]})

