

from distutils.core import setup
from distutils.extension import Extension
from distutils.command.build_ext import build_ext
from Cython.Build import cythonize
import pathlib
import numpy as np
import mpi4py
import os
import glob


# Look for Mercurial tools
try:
    import hgapi
except ImportError:
    hgapi = None

def get_prefix(name):
    """
    Get prefix path for libraries containing string <name>.

    First checks environment variables for <NAME>_PATH, then searches a few
    other likely places.

    """

    print("Looking for %s prefix" %name)

    # Check for environment variable
    patterns = ['%s_PATH',
                '%s_PREFIX']
    for pattern in patterns:
        env_var = pattern %name.upper()
        if env_var in os.environ:
            path = os.environ[env_var]
            print("  Found env var %s = %s" %(env_var, path))
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

    print("  Cannot find env var %s_NAME or libraries matching %s." %(name.upper(), name))
    print("  If %s isn't in your LD_LIBRARY_PATH, compilation will likely fail." %name)

def get_include(name):
    prefix = get_prefix(name)
    return os.path.join(prefix, 'include')

def get_lib(name):
    prefix = get_prefix(name)
    return os.path.join(prefix, 'lib')

def hg_info(path):
    # Resolve symoblic paths
    path = pathlib.Path(path).resolve()
    # Get mercurial repository
    repo = hgapi.Repo(str(path))
    return repo.hg_id(), repo.hg_diff()

# Modify build_ext to record hg repository info, if possible
if hgapi:
    try:
        class custom_build_ext(build_ext):
            def run(self):
                # honor the --dry-run flag
                if not self.dry_run:
                    target_dir = os.path.join(self.build_lib,'dedalus2')
                    src_dir =  os.getcwd()
                    id, diff = hg_info(src_dir)
                    self.mkpath(target_dir)
                    with open(os.path.join(target_dir, '__hg_version__.py'), 'w') as fobj:
                        fobj.write("hg_version = '%s'\n" %id)
                        fobj.write("diff = %s\n" %diff)
                    build_ext.run(self)
    except hgapi.HgException:
        custom_build_ext = build_ext
else:
    custom_build_ext = build_ext

fftw_ext = Extension(
    name='dedalus2.libraries.fftw.fftw_wrappers',
    sources=['dedalus2/libraries/fftw/fftw_wrappers.pyx'],
    include_dirs=['dedalus2/libraries/fftw/',
                  np.get_include(),
                  mpi4py.get_include(),
                  get_include('fftw'),
                  get_include('mpi')],
    libraries=['fftw3_mpi', 'fftw3', 'm'],
    library_dirs=[get_lib('fftw')]
    )

setup(
    name='Dedalus',
    version='2',
    author='Keaton J. Burns',
    author_email='keaton.burns@gmail.com',
    license='GPL3',
    packages = ['dedalus2',
                'dedalus2.data',
                'dedalus2.pde',
                'dedalus2.libraries',
                'dedalus2.libraries.fftw'],
    cmdclass = {'build_ext': custom_build_ext},
    ext_modules = cythonize([fftw_ext])
    )
