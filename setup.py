"""
Setup script for Dedalus.
The following environment variables may be set:

FFTW_PATH
    Path to FFTW installation prefix
FFTW_STATIC
    Binary flag to statically link FFTW, 0 by default
CYTHON_PROFILE
    Binary flag to enable profiling within cython modules, 0 by default

"""

import setuptools
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
from distutils.util import strtobool
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
    env_var = os.getenv(name)
    if env_var is None:
        return unset
    else:
        return bool(strtobool(env_var))

def check_env_var(env_var):
    path = None
    if env_var in os.environ:
        path = os.environ[env_var]
        print("  Found env var %s = %s" %(env_var, path))
    return path

def get_prefix(name):
    """
    Get prefix path for libraries containing string "name".
    First checks environment variables for <NAME>_PATH and NAME_PREFIX.
    Then searches a few other likely places.
    """
    print("Looking for %s prefix" %name)
    # Check for environment variable
    patterns = ['%s_PATH',
                '%s_PREFIX']
    for pattern in patterns:
        env_var = pattern %name.upper()
        path = check_env_var(env_var)
        if path:
            return path
    # Check likely places
    places = [os.environ.get('CONDA_PREFIX', ''),
              '/usr/lib/openmpi', # MPI under Ubuntu
              '/usr',
              '/usr/local',
              os.path.expanduser('~/build')]
    for place in places:
        placelib = os.path.join(place, 'lib')
        guess = os.path.join(placelib, '*%s*' %name)
        matches = glob.glob(guess)
        if matches:
            print("  Found matching library in %s" %placelib)
            return place
    print("  Cannot find env var %s_PATH or libraries matching %s." %(name.upper(), name))
    print("  If %s isn't in your LD_LIBRARY_PATH, compilation will likely fail." %name)

def get_include(name):
    print("Looking for %s include path" %name)
    env_var = "%s_INCLUDE_PATH" % name.upper()
    path = check_env_var(env_var)
    if path:
        return path
    print("  Cannot find env var %s" %env_var)
    prefix = get_prefix(name)
    if prefix:
        return os.path.join(prefix, 'include')

def get_lib(name):
    print("Looking for %s library path" %name)
    env_var = "%s_LIBRARY_PATH" % name.upper()
    path = check_env_var(env_var)
    if path:
        return path
    print("  Cannot find env var %s" %env_var)
    prefix = get_prefix(name)
    if prefix:
        return os.path.join(prefix, 'lib')

def get_version(rel_path):
    """Read version from a file via text parsing, following PyPA guide."""
    def read(rel_path):
        here = os.path.abspath(os.path.dirname(__file__))
        with codecs.open(os.path.join(here, rel_path), 'r') as fp:
            return fp.read()
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

# C-dependency paths for extension compilation and linking
include_dirs = ['dedalus/libraries/fftw/',
                np.get_include(),
                mpi4py.get_include(),
                get_include('mpi'),
                get_include('fftw')]
libraries = ['fftw3_mpi', 'fftw3', 'm']
library_dirs = [get_lib('fftw'), get_lib('mpi')]

# Warning supression
extra_compile_args = ["-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION",
                      "-Wno-unused-function"]

# Optionally set static linking for FFTW
extra_link_args = []
if bool_env('FFTW_STATIC', unset=False):
    print("Statically linking FFTW")
    fftw_lib_path = get_lib('fftw')
    clang_extra_link_args = ["-Wl,-force_load",
                             "{}/libfftw3.a".format(fftw_lib_path),
                             "{}/libfftw3_mpi.a".format(fftw_lib_path),
                             "-Wl,-noall_load"]
    gcc_extra_link_args = ["-Xlinker", "-Bsymbolic",
                           "-Wl,--whole-archive",
                           "{}/libfftw3.a".format(fftw_lib_path),
                           "{}/libfftw3_mpi.a".format(fftw_lib_path),
                           "-Wl,--no-whole-archive"]
    # Choose linker flags based on CC
    CC = os.getenv('CC')
    if "clang" in CC:
        print("CC set to clang; using clang linker flags")
        extra_link_args = clang_extra_link_args
    elif CC:
        print("CC set to {}; using gcc linker flags".format(CC))
        extra_link_args = gcc_extra_link_args
    elif sys.platform == "darwin":
        print("CC not set; defaulting to clang linker flags")
        extra_link_args = clang_extra_link_args
    else:
        print("CC not set; defaulting to gcc linker flags")
        extra_link_args = gcc_extra_link_args

# Extension objects for cython
extensions = [
    Extension(
        name='dedalus.libraries.fftw.fftw_wrappers',
        sources=['dedalus/libraries/fftw/fftw_wrappers.pyx'],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        runtime_library_dirs=library_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args),
    Extension(
        name='dedalus.core.transposes',
        sources=['dedalus/core/transposes.pyx'],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        runtime_library_dirs=library_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args),
    Extension(
        name='dedalus.core.polynomials',
        sources=['dedalus/core/polynomials.pyx'],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        runtime_library_dirs=library_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args)]

# Runtime requirements
install_requires = [
    "docopt",
    "h5py >= 2.10.0",
    "matplotlib",
    "mpi4py >= 2.0.0",
    "numpy >= 1.20.0",
    "py",
    "pytest",
    "pytest-benchmark",
    "pytest-cov",
    "pytest-parallel",
    "scipy >= 1.4.0"]

# Grab long_description from README
with open('README.md') as f:
    long_description = f.read()

# Cython directives
compiler_directives = {}
compiler_directives['language_level'] = 3
if bool_env('CYTHON_PROFILE', unset=False):
    compiler_directives['profile'] = True

# Override build command to pack up examples
from distutils.command.build import build as _build
class build(_build):
    def run(self):
        # Create tar file with example scripts
        with tarfile.open('dedalus/examples.tar.gz', mode='w:gz') as archive:
            examples_dir = pathlib.Path('examples')
            for file in examples_dir.glob('**/*.py'):
                archive.add(str(file), str(file.relative_to(examples_dir)))
        # Run the original build command
        _build.run(self)

# Setup
setup(
    name='dedalus',
    version=get_version("dedalus/__init__.py"),
    author='Keaton J. Burns',
    author_email='keaton.burns@gmail.com',
    description="A flexible framework for solving PDEs with modern spectral methods.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='http://dedalus-project.org',
    classifiers=['Programming Language :: Python :: 3'],
    install_requires=install_requires,
    license='GPL3',
    packages=setuptools.find_packages(),
    package_data={'': ['dedalus.cfg', 'examples.tar.gz']},
    ext_modules=cythonize(extensions, compiler_directives=compiler_directives),
    cmdclass={"build": build})

