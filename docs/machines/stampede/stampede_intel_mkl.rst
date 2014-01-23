Stampede MKL libraries
************************************

On Stampede we will likely generally use the Intel MKL libraries for
BLAS; these are loaded through a module command and should be
available by default.  Here are instructions for building against the
intel compilers.


Building numpy against MKL
----------------------------------

First, create an Intel MKL virtualenv instance::

     cd ~/venv
     virtualenv intel_mkl
     source ~/venv/intel_mkl/bin/activate

Now, acquire ``numpy`` (1.8.0)::

     cd ~/venv/mkl
     wget http://sourceforge.net/projects/numpy/files/NumPy/1.8.0/numpy-1.8.0.tar.gz
     tar -xvf numpy-1.8.0.tar.gz
     cd numpy-1.8.0

We'll now need to make sure that ``numpy`` is building against the MKL
libraries.  Start by making a ``site.cfg`` file::

     cp site.cfg.example site.cfg
     emacs -nw site.cfg

Edit ``site.cfg`` in the ``[mkl]`` section; modify the
library directory so that it correctly point to TACC's
``$MKLROOT/lib/intel64/``.  
With the modules loaded above, this looks like::

     [mkl]
     library_dirs = /opt/apps/intel/13/composer_xe_2013.2.146/mkl/lib/intel64
     include_dirs = /opt/apps/intel/13/composer_xe_2013.2.146/mkl/include
     mkl_libs = mkl_rt
     lapack_libs =

These are based on intels instructions for 
`compiling numpy with ifort <http://software.intel.com/en-us/articles/numpyscipy-with-intel-mkl>`_
and they seem to work so far.

Further following those instructions, you'll need to hand edit two
files in ``numpy/distutils``; these are ``intelccompiler.py`` and
``fcompiler/intel.py``.  I've built a crude patch,
:download:`numpy_intel_patch.tar<numpy_intel_patch.tar>` 
which can be auto-deployed by within the ``numpy-1.8.0`` directory by
doing::
    
      tar -xvf numpy_intel_patch.tar

This will unpack and overwrite::

      numpy/distutils/intelccompiler.py
      numpy/distutils/fcompiler/intel.py

Then proceed with::

    python3 setup.py config --compiler=intelem build_clib --compiler=intelem build_ext --compiler=intelem install

This will config, build and install numpy.


Test numpy install
------------------------------

Test that things worked with this executable script
:download:`numpy_test_full<numpy_test_full>`, 
or do so manually by launching ``python3`` 
and then doing::

     import numpy as np
     np.__config__.show()

If you've installed ``nose`` (with ``pip3 install nose``), 
we can further test our numpy build with::

     np.test()
     np.test('full')

We fail ``np.test()`` with two failures, while ``np.test('full')`` has
3 failures and 19 errors.  But we do successfully link against the
fast BLAS libraries (look for ``FAST BLAS`` output, and fast dot
product time).

.. note::
     We should check what impact these failed tests have on our results.



Numpy config check
------------------------------
After executing config, check that numpy has correctly found the
MKL install.  You should see something like this:

::

  Running from numpy source directory.
  F2PY Version 2
  blas_opt_info:
  blas_mkl_info:
    FOUND:
      define_macros = [('SCIPY_MKL_H', None)]
      include_dirs = ['/opt/apps/intel/13/composer_xe_2013.2.146/mkl/include']
      library_dirs = ['/opt/apps/intel/13/composer_xe_2013.2.146/mkl/lib/intel64']
      libraries = ['mkl_rt', 'pthread']

    FOUND:
      define_macros = [('SCIPY_MKL_H', None)]
      include_dirs = ['/opt/apps/intel/13/composer_xe_2013.2.146/mkl/include']
      library_dirs = ['/opt/apps/intel/13/composer_xe_2013.2.146/mkl/lib/intel64']
      libraries = ['mkl_rt', 'pthread']

    non-existing path in 'numpy/lib': 'benchmarks'
    lapack_opt_info:
    openblas_info:
      libraries openblas not found in ['/home1/00364/tg456434/build_ifort/lib', '/usr/local/lib64', '/usr/local/lib', '/usr/lib64', '/usr/lib']
      NOT AVAILABLE

  lapack_mkl_info:
  mkl_info:
    FOUND:
      define_macros = [('SCIPY_MKL_H', None)]
      include_dirs = ['/opt/apps/intel/13/composer_xe_2013.2.146/mkl/include']
      library_dirs = ['/opt/apps/intel/13/composer_xe_2013.2.146/mkl/lib/intel64']
      libraries = ['mkl_rt', 'pthread']

    FOUND:
      define_macros = [('SCIPY_MKL_H', None)]
      include_dirs = ['/opt/apps/intel/13/composer_xe_2013.2.146/mkl/include']
      library_dirs = ['/opt/apps/intel/13/composer_xe_2013.2.146/mkl/lib/intel64']
      libraries = ['mkl_rt', 'pthread']

    FOUND:
      define_macros = [('SCIPY_MKL_H', None)]
      include_dirs = ['/opt/apps/intel/13/composer_xe_2013.2.146/mkl/include']
      library_dirs = ['/opt/apps/intel/13/composer_xe_2013.2.146/mkl/lib/intel64']
      libraries = ['mkl_rt', 'pthread']

  /home1/00364/tg456434/build_ifort/lib/python3.3/distutils/dist.py:257: UserWarning: Unknown distribution option: 'define_macros'
    warnings.warn(msg)
