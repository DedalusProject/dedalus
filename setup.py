from distutils.core import setup
from distutils.extension import Extension
from distutils.command.build_py import build_py
from Cython.Build import cythonize
import numpy as np
import mpi4py
import os
# import distribute_setup
# distribute_setup.use_setuptools()
# import setuptools

def find_path(name):
    """looks for <name> in <NAME>_PATH, then searches a few other likely
    places.

    """
    import glob
    import os
    try:
        pn = '%s_PATH' % name.upper()
        path = os.environ[pn]
        return path
    except KeyError:
        # for MPI under Ubuntu
        path = "/usr/lib/openmpi/lib"
        glob_string = "*%s*" % name
        l = glob.glob(os.path.join(path, glob_string))
        if len(l) != 0:
            return os.path.split(path)[0]

        path = "/usr/lib"
        glob_string = "*%s*" % name
        l = glob.glob(os.path.join(path, glob_string))
        if len(l) != 0:
            return os.path.split(path)[0]

        path = "/usr/local/lib"
        glob_string = "*%s*" % name
        l = glob.glob(os.path.join(path, glob_string))
        if len(l) != 0:
            return os.path.split(path)[0]

        path = os.path.expanduser("~/build/lib")
        l = glob.glob(os.path.join(path,glob_string))
        if len(l) != 0:
            return os.path.split(path)[0]

        print("%s is not set, and we can't find %s in any of the standard places. If %s isn't in your LD_LIBRARY_PATH, compilation will likely fail." % (pn, name, name))

def fftw_get_include():
    base = find_path('fftw')
    return os.path.join(base,"include")

def fftw_get_lib():
    base = find_path('fftw')
    return os.path.join(base,"lib")

def mpi_get_include():
    base = find_path('mpi')
    print(base)
    return os.path.join(base,"include")

def mpi_get_lib():
    base = find_path('mpi')
    return os.path.join(base,"lib")

def get_mercurial_changeset_id(targetDir):
    """adapted from a script by Jason F. Harris, published at

    http://jasonfharris.com/blog/2010/05/versioning-your-application-with-the-mercurial-changeset-hash/

    """
    import subprocess
    import re
    getChangeset = subprocess.Popen('hg parent --template "{node|short}" --cwd ' + targetDir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)

    if (getChangeset.stderr.read() != ""):
        print("Error in obtaining current changeset of the Mercurial repository")
        changeset = None

    changeset = getChangeset.stdout.read()
    if (not re.search("^[0-9a-f]{12}$", changeset)):
        print("Current changeset of the Mercurial repository is malformed")
        changeset = None

    return changeset

class my_build_py(build_py):
    def run(self):
        # honor the --dry-run flag
        if not self.dry_run:
            target_dir = os.path.join(self.build_lib,'dedalus2')
            src_dir =  os.getcwd()
            changeset = get_mercurial_changeset_id(src_dir)
            self.mkpath(target_dir)
            with open(os.path.join(target_dir, '__hg_version__.py'), 'w') as fobj:
                fobj.write("hg_version = '%s'\n" % changeset)

            build_py.run(self)

fftw_ext = Extension(
    name='dedalus2.libraries.fftw.fftw_wrappers',
    sources=['dedalus2/libraries/fftw/fftw_wrappers.pyx'],
    include_dirs=["dedalus2/libraries/fftw", mpi4py.get_include(), fftw_get_include(), mpi_get_include()],
    libraries=['fftw3_mpi', 'fftw3', 'm'],
    library_dirs=[fftw_get_lib()]
    )

setup(
    name='Dedalus',
    version='2.1',
    author='Keaton J. Burns',
    author_email='keaton.burns@gmail.com',
    license='GPL3',
    packages = ['dedalus',
                'dedalus.data',
                'dedalus.pde',
                'dedalus.libraries',
                'dedalus.libraries.fftw'],
    cmdclass = {'build_py': my_build_py},
    include_dirs = [np.get_include()],
    ext_modules = cythonize([fftw_ext], include_path=[mpi4py.get_include()])
    )
