Install notes for TACC/Stampede
***************************************************************************
(based on Jeff's notes for Trestles)

.. note ::

  Big note to self/others: we may be building against the wrong nodes
  with these instructions.  We may need to do a ``-xHost`` command and use
  the intel compilers for the compute nodes.  I have a ticket opened
  with TACC to find out and will update this when I hear back (BPB, 1/7/14).


Modules
==========================================

Here is my current build environment (from running ``module list``)

  1) TACC-paths
  2) Linux
  3) cluster-paths
  4) intel/13.0.2.146
  5) TACC  
  6) cluster  
  7) mvapich2/1.9a2   
  8) fftw3/3.3.2

Stampede does not appear to have OpenMPI, but does have ``fftw3`` built
properly on ``mvapich2``.  See the `Stampede user guide <https://www.tacc.utexas.edu/user-services/user-guides/stampede-user-guide#compenv-modules-login>`_
if you need to modify the modules that are auto-loaded at startup.

Python stack
=========================

Building Python3
--------------------------

Create ``~\build`` and then proceed with downloading and installing Python-3.3::

    cd ~/build
    wget http://www.python.org/ftp/python/3.3.3/Python-3.3.3.tgz
    tar -xzf Python-3.3.3.tgz
    cd Python-3.3.3
    ./configure --prefix=$HOME/build

    make
    make install

Here we are building everything in ``~/build``; you can do it
whereever, but adjust things appropriately in the above instructions.
Also, it may be beneficial to do ``make -j4`` to build on 4 cores
(see :doc:`trestles` notes), but doing a single core build only took a
few minutes on stampede.

*The warning messages from Trestles were not encountered on Stampede.*

Updating shell settings
------------------------------

At this point, ``python3`` is installed in ``~/build/bin/``.  Add this
to your path and confirm (currently there is no ``python3`` in the
default path, so doing a ``which python3`` will fail if you haven't
added ``~/build/bin``).  

On Stampede, login shells (interactive connections via ssh) source
only ``~/.bash_profile``, ``~/.bash_login`` or ``~/.profile``, in that
order, and do not source ``~/.bashrc``.  Meanwhile non-login shells
only launch ``~/.bashrc`` 
(see Stampede `user guide <https://www.tacc.utexas.edu/user-services/user-guides/stampede-user-guide#compenv-startup-technical>`_).

In the bash shell, add the following to
``.bashrc``::

     export PATH=~/build/bin:$PATH
     export LD_LIBRARY_PATH=~/build/lib:$LD_LIBRARY_PATH

and the following to ``.profile``::

     if [ -f ~/.bashrc ]; then . ~/.bashrc; fi

(from `bash reference manual <https://www.gnu.org/software/bash/manual/html_node/Bash-Startup-Files.html>`_) 
to obtain the same behaviour in both shell types.

Installing pip
-------------------------

We'll use ``pip`` to install our python library depdencies.
Instructions on doing this are available `here <http://www.pip-installer.org/en/latest/installing.htm>`_ 
and summarized below.  First
download and install setup tools::

    cd ~/build
    wget https://bitbucket.org/pypa/setuptools/raw/bootstrap/ez_setup.py
    python3 ez_setup.py

Then install ``pip``::

    wget https://raw.github.com/pypa/pip/master/contrib/get-pip.py
    python3 get-pip.py

You will now have ``pip3`` and ``pip`` installed in ``~/build/bin``.
You might try doing ``pip -V`` to confirm that ``pip`` is built
against python 3.3.  We will use ``pip3`` throughout this
documentation to remain compatible with systems (e.g., Mac OS) where
multiple versions of python coexist.




Installing virtualenv
-------------------------

In order to test multiple numpys and scipys (and really, their
underlying BLAS libraries), we will use ``virtualenv``::

     pip3 install virtualenv

Next, construct a virtualenv to hold all of your python modules. We
suggest doing this in your home directory::

     mkdir ~/venv



BLAS libraries
======================================

Intel MKL
--------------------------
On Stampede we will likely generally use the Intel MKL libraries for
BLAS; these are loaded through a module command and should be
available by default.  

.. note ::

   I'm hitting frustrating errors with MKL (1/7/14), so for now we'll
   proceed with OpenBLAS.  I have a ticket open with TACC to start
   resolving some of this.

Building numpy against MKL
----------------------------------

`See some useful but outdated notes here <https://www.cac.cornell.edu/stampede/python/nscompile.aspx>`_

First, create an Intel MKL virtualenv instance::

     cd ~/venv
     virtualenv mkl
     source ~/venv/mkl/bin/activate

Now, acquire ``numpy`` (1.8.0)::

     cd ~/venv/mkl
     wget http://sourceforge.net/projects/numpy/files/NumPy/1.8.0/numpy-1.8.0.tar.gz
     tar -xvf numpy-1.8.0.tar.gz
     cd numpy-1.8.0

We'll now need to make sure that ``numpy`` is building against the MKL
libraries.  Start by making a ``site.cfg`` file::

     cp site.cfg.example site.cfg


OpenBLAS
--------------------------

We may also wish to build and test against
OpenBLAS.

To download and install openBLAS, first do the following::

      cd ~/build
      git clone https://github.com/xianyi/OpenBLAS.git
      cd OpenBLAS
      make
      make PREFIX=$HOME/build install

This builds and automatically makes a multi-threaded version of
OpenBLAS (16 threads right now). 

 .. note :: 

  I'm uncertain whether this is all working correctly.  Namely,
  we may need to do a compute-node targeted build, rather than a
  login-targeted build, and use ifort.  We'll see.

Building numpy against openblas
------------------------------------


First, create an OpenBLAS virtualenv instance::

     cd ~/venv
     virtualenv openblas
     source ~/venv/openblas/bin/activate

Now, acquire ``numpy`` (1.8.0)::

     cd ~/venv/openblas
     wget http://sourceforge.net/projects/numpy/files/NumPy/1.8.0/numpy-1.8.0.tar.gz
     tar -xvf numpy-1.8.0.tar.gz
     cd numpy-1.8.0

Next, make a site specific config file::

      cp site.cfg.example site.cfg
      emacs -nw site.cfg

Edit ``site.cfg`` to uncomment the ``[openblas]`` section; modify the
library and include directories so that they correctly point to your
``~/build/lib`` and ``~/build/include`` (note, you may need to do fully expanded
paths).

Then proceed with::

     python3 setup.py config

After executing config, check that numpy has correctly found the
OpenBLAS install.  You should see something like this:

::

      (openblas)login2$ python3 setup.py config
      Running from numpy source directory.
      F2PY Version 2
      blas_opt_info:
      blas_mkl_info:
      /home1/00364/tg456434/venv/openblas/numpy-1.8.0/numpy/distutils/system_info.py:576: UserWarning: Specified path /opt/apps/intel/13/composer_xe_2013.2.146/mkl/lib/em64t is invalid.
        warnings.warn('Specified path %s is invalid.' % d)
        libraries mkl,vml,guide not found in []
        NOT AVAILABLE

      openblas_info:
        FOUND:
          language = f77
          library_dirs = ['/home1/00364/tg456434/build/lib']
          libraries = ['openblas', 'openblas']

        FOUND:
          language = f77
          library_dirs = ['/home1/00364/tg456434/build/lib']
          libraries = ['openblas', 'openblas']

      non-existing path in 'numpy/lib': 'benchmarks'
      lapack_opt_info:
        FOUND:
          language = f77
          library_dirs = ['/home1/00364/tg456434/build/lib']
          libraries = ['openblas', 'openblas']

      /home1/00364/tg456434/build/lib/python3.3/distutils/dist.py:257: UserWarning: Unknown distribution option: 'define_macros'
        warnings.warn(msg)
      running config
      (openblas)login2$

Next do::

     python3 setup.py build
     python3 setup.py install

Test that things worked by launching ``python3`` and then doing::

     import numpy as np
     np.__config__.show()


*Note: if on ``import numpy as np`` you get an error on loading the
OpenBLAS shared library, see above note about ``$LD_LIBRARY_PATH``.*


Python library stack
=====================


Installing Scipy
-------------------------

Scipy is easier, because it just gets its config from numpy.  Download
an install in your appropriate ``~/venv/INSTANCE`` directory:

     wget http://sourceforge.net/projects/scipy/files/scipy/0.13.2/scipy-0.13.2.tar.gz
     tar -xvf scipy-0.13.2.tar.gz
     cd scipy-0.13.2


* `$ python setup.py config`

This notes a lack of UMFPACK...will that make a speed difference? Nevertheless, it works ok.

Do

* `$ python setup.py build`

if successful, 

* `$ python setup.py install`


Installing mpi4py
-----------------

This should just be pip installed, `$ pip install mpi4py`

Installing cython
-----------------

This should just be pip installed, `$ pip install cython`

Installing matplotlib
-----------------

This should just be pip installed, `$ pip install matplotlib`

UMFPACK
-------

Requires AMD (another package by the same group, not processor) and SuiteSparse_config, too.

Dedalus2
--------

With the modules set as above (for NOW), set `$ export
FFTW_PATH=/opt/fftw/3.3.3/gnu/openmpi/ib` and `$ export
MPI_PATH=/opt/openmpi/gnu/ib`. Then do `$ python setup.py build_ext
--inplace`.




