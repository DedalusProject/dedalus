
import setuptools
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
from distutils.util import strtobool
import numpy as np
import mpi4py
import os
import glob


# Helper functions for finding C-dependency paths
def check_env_var(env_var):
    path = None
    if env_var in os.environ:
        path = os.environ[env_var]
        print("  Found env var %s = %s" %(env_var, path))
    return path

def get_prefix(name):
    """
    Get prefix path for libraries containing string <name>. First checks environment
    variables for <NAME>_PATH, then searches a few other likely places.
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
    places = ['/usr/lib/openmpi', # MPI under Ubuntu
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
    env_var = "%s_INCLUDE_PATH" % name.upper()
    path = check_env_var(env_var)
    if path:
        return path
    prefix = get_prefix(name)
    return os.path.join(prefix, 'include')

def get_lib(name):
    env_var = "%s_LIBRARY_PATH" % name.upper()
    path = check_env_var(env_var)
    if path:
        return path
    prefix = get_prefix(name)
    return os.path.join(prefix, 'lib')

# C-dependency paths for extension compilation and linking
include_dirs = ['dedalus/libraries/fftw/',
                np.get_include(),
                mpi4py.get_include(),
                get_include('fftw'),
                get_include('mpi')]
libraries = ['fftw3_mpi', 'fftw3', 'm']
library_dirs = [get_lib('fftw')]

# Optionally set static linking for FFTW
FFTW_STATIC='FFTW_STATIC'
fs = os.getenv(FFTW_STATIC)
if fs:
    fftw_static=bool(strtobool(fs))
else:
    fftw_static=False
if fftw_static:
    print("Statically linking FFTW to allow Dedalus to work with Intel MKL.")
    fftw_lib_path = get_lib('fftw')
    extra_link_args = ["-Xlinker",
                       "-Bsymbolic",
                       "-Wl,--whole-archive",
                       "{}/libfftw3.a".format(fftw_lib_path),
                       "{}/libfftw3_mpi.a".format(fftw_lib_path),
                       "-Wl,--no-whole-archive"]
else:
    extra_link_args = []

# Extension objects for cython
extensions = [
    Extension(
        name='dedalus.libraries.fftw.fftw_wrappers',
        sources=['dedalus/libraries/fftw/fftw_wrappers.pyx'],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        extra_compile_args=["-Wno-error=declaration-after-statement"],
        extra_link_args=extra_link_args),
    Extension(
        name='dedalus.core.transposes',
        sources=['dedalus/core/transposes.pyx'],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        extra_compile_args=["-Wno-error=declaration-after-statement"],
        extra_link_args=extra_link_args),
    Extension(
        name='dedalus.core.polynomials',
        sources=['dedalus/core/polynomials.pyx'],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        extra_compile_args=["-Wno-error=declaration-after-statement"])]

# Runtime requirements
install_requires = [
    "docopt",
    "h5py >= 2.6.0",
    "matplotlib",
    "mpi4py >= 2.0.0",
    "numpy",
    "scipy >= 0.13.0"]

# Grab long_description from README
with open('README.md') as f:
    long_description = f.read()

setup(
    name='dedalus',
    version='2.1808a2',
    author='Keaton J. Burns',
    author_email='keaton.burns@gmail.com',
    description="A flexible framework for solving differential equations using spectral methods.",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='http://dedalus-project.org',
    classifiers=['Programming Language :: Python :: 3'],
    install_requires=install_requires,
    license='GPL3',
    packages=setuptools.find_packages(),
    package_data={'': ['dedalus.cfg']},
    ext_modules=cythonize(extensions))
