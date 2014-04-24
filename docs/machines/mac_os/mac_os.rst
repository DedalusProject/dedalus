.. converted from Installing_Dedalus.md using pandoc, for now via http://johnmacfarlane.net/pandoc/try/

Install notes for Mac OS X (10.9)
*******************************************

To cut to the chase, look for the appropriate cookbook for your system
in the table of contents. More detailed guides preceed each cookbook.

These instructions assume you're starting with a clean Mac OS X system,
which will need ``python3`` and all scientific packages installed. These
instructions are based off an excellent  `guide`_
and are the result of extensive attempts to properly install ``numpy``
and ``scipy`` in particular (including building directly from source).

Mac OS X cookbook
-----------------

::

    #!bash
    ruby -e "$(curl -fsSL https://raw.github.com/Homebrew/homebrew/go/install)"
    brew update
    brew doctor
    # check whether any errors arise with brew doctor before proceeding

    brew install swig
    brew install gfortran
    brew install python3

    # get UMFPACK libraries
    brew tap homebrew/science
    brew install suite-sparse

    # now, on with the scientific python install
    pip3 install nose
    pip3 install numpy
    pip3 install scipy
    pip3 install sympy

    brew install freetype
    pip3 install matplotlib
    pip3 install ipython


    brew install hdf5
    pip3 install h5py
    brew install mpich2
    brew install fftw
    pip3 install cython
    pip3 install mpi4py
    pip3 install tqdm
    pip3 install pathlib


Detailed install notes for Mac OS X (10.9)
*******************************************

Preparing a Mac system
----------------------

First, install Xcode and seperately install the Xcode Command Line
Tools. To install the command line tools, open Xcode, go to
``Preferences``, select the ``Downloads`` tab and ``Components``. These
command line tools install ``make`` and other requisite tools that are
no longer automatically included in Mac OS X (as of 10.8).

Next, you should install the `homebrew`_ package manager for OS X. Do
the following:

::

    #!bash
    ruby -e "$(curl -fsSL https://raw.github.com/Homebrew/homebrew/go/install)"
    brew update
    brew doctor

Cleanup any problems identified by ``brew doctor`` before proceeding.

You'll need a gfortran compiler to complete the ``scipy`` install.

.. _guide: http://www.lowindata.com/2013/installing-scientific-python-on-mac-os-x/
.. _homebrew: http://brew.sh/

.. _MesaSDK: http://www.astro.wisc.edu/~townsend/static.php?ref=mesasdk
.. _MADSDK: http://www.astro.wisc.edu/~townsend/static.php?ref=madsdk

If you don't have one, try downloading Rich Townsend's `MesaSDK`_ or
`MADSDK`_ (if you want MPI capability) software development kits. There
is excellent documentation on the `MesaSDK`_ page. You can also install
one directly with `homebrew`_, which is the path we follow here:

::

    #!bash
    brew install gfortran
    brew install swig

From building on my clean workstation, it looks like the ``scipy``
install process depends on ``swig``.

Install python3
---------------

Now, install python3

::

    #!bash
    brew install python3
    

Scientific packages for Python3
-------------------------------

Next install the numpy and scipy scientific packages. To adequately warn
you before proceeding, properly installing numpy and scipy on a Mac can
be a frustrating experience.

Start by proactively installing UMFPACK from suite-sparse, located in
homebrew-science on https://github.com/Homebrew/homebrew-science.
Failing to do this may lead to a series of perplexing UMFPACK errors
during the scipy install.

::

    #!bash
    brew tap homebrew/science
    brew install suite-sparse

Now, use ``pip3`` (installed with ``brew install python3`` above) to
install ``numpy`` and then ``scipy``, in that order; here we use
``pip3`` to force a ``python3`` install:

::

    #!bash
    pip3 install nose
    pip3 install numpy
    pip3 install scipy
    pip3 install sympy

The ``scipy`` install can fail in a number of surprising ways. Be
especially wary of custom settings to ``LDFLAGS``, ``CPPFLAGS``, etc.
within your shell; these may cause the ``gfortran`` compile step to fail
spectacularly.

Python plotting libraries
-----------------------------

::

    #!bash
    brew install freetype
    pip3 install matplotlib
    pip3 install ipython
    pip3 install brewer2mpl


Further Dedalus dependancies
------------------------------

Output is done via HDF5 files, for which we'll need both the hdf5
libraries and h5py.  We also require mpi4py, cython, fftw3 (for
parallel transposes and transforms) and an mpi implementation.  
Here we use mpich2.  We also suggest tqdm and pathlib in case you're
using the development branch.

::

    brew install hdf5
    pip3 install h5py
    brew install mpich2
    brew install fftw
    pip3 install cython
    pip3 install mpi4py
    pip3 install tqdm
    pip3 install pathlib





Optional packages
-----------------

For those who use the VAPOR volume rendering package, you may have a
conflict with Vapor's own install of szip. You can force usage of
homebrew's szip with

::

    brew link --overwrite szip

You may additionally want to install these PyQT4 libraries (these seem
to be broken in my install, so beware until I sort this out):

::

    #!bash
    # install the QT libraries as described
    # in the text on http://www.lowindata.com/2013/installing-scientific-python-on-mac-os-x/ before continuing
    brew install pyqt
    brew install zmq
    pip3 install pyzmq
    pip3 install pygments

Profiling
---------

For profiling the code, ``pyprof2html`` is very helpful. It is only
currently available under a python2 install, but can be run directly
from the command line once installed. To install,

::

    #!bash
    pip install pyprof2html

Installations under python3 (with ``pip3``) will likely fail with a
string of errors about importing ``Converter``. You may also need
``jinja2``.

To use, first run the code in profiling mode, then execute
``pyprof2html``::

      #!bash python3 -m cProfile -o profiling_output test_script.py 
      pyprof2html profiling_output 
      open html/index-all.html

Bash shell modifications
------------------------

You'll need to introduce some path information into your shell; here
we'll assume you're using bash and the modifications are within
``.bash_profile``:

::

    #!bash
    export MYUSERNAME=bbrown

    #prefix /usr/local/bin so that homebrew can override path settings
    export PATH=/usr/local/bin:$PATH

    # make homebrew pip work
    export PATH=/usr/local/share/python:$PATH

    # homebrew pathing for pyqt
    export PYTHONPATH=/usr/local/lib/python:$PYTHONPATH

    # Dedalus python pathing
    export PYTHONPATH=/Users/$MYUSERNAME/code/dedalus2:$PYTHONPATH

Installing OpenBLAS
-------------------

The Mac accelerate framwork BLAS is fast and automatically threads tasks
like ``dgemm`` (expressed through ``numpy.dot()``). It may be useful to
be able to OpenBLAS rather than the accelerate libraries. To do a direct
comparison, the OpenBLAS library needs to be compiled with the
environment variable ``OPENBLAS_NUM_THREADS`` set to an appropriate
value, or OpenBLAS will only use a subset of the system resources. For a
macbook air, this correct value is ``OPENBLAS_NUM_THREADS=4`` (otherwise
only 2 threads are utilized).

::

    !#bash
    export OPENBLAS_NUM_THREADS=4
    brew install openblas

This is sufficient for fortran programs, etc., that link directly
against OpenBLAS. As yet, it's unclear how to build numpy against
OpenBLAS rather than accelerate, but Jeff Oishi's notes describe this in
detail for other systems and other libraries. I'll check and update.

A note on f2py with python3
---------------------------

I'm just starting to play around with ``f2py``. As a warning, I've
needed to hand-modify ``/usr/local/bin/f2py`` so that the opening line
reads

::

       #!/usr/bin/env python3

rather than

::

       #!/usr/bin/env Python


Other resources:

http://www.lowindata.com/2013/installing-scientific-python-on-mac-os-x/

http://stackoverflow.com/questions/12574604/scipy-install-on-mountain-lion-failing

https://github.com/jonathansick/dotfiles/wiki/Notes-for-Mac-OS-X


