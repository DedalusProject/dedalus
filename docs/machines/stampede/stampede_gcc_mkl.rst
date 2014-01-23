Stampede MKL libraries
************************************

On Stampede we will likely generally use the Intel MKL libraries for
BLAS; these are loaded through a module command and should be
available by default.  

.. note ::

   We have a ticket open with TACC to start
   resolving problems with numpy builds against MKL.

   Note: do this under ifort

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
     emacs -nw site.cfg

Edit ``site.cfg`` in the ``[mkl]`` section; modify the
library directory so that it correctly point to TACC's
``$MKLROOT/lib/intel64/``.  
With the modules loaded above, this looks like::

     [mkl]
     library_dirs = /opt/apps/intel/13/composer_xe_2013.2.146/mkl/lib/intel64/
     include_dirs = /opt/apps/intel/13/composer_xe_2013.2.146/mkl/include/
     mkl_libs = mkl_rt
     lapack_libs =

These are based on intels instructions for 
`compiling numpy with ifort <http://software.intel.com/en-us/articles/numpyscipy-with-intel-mkl>`_
and they seem to work so far.

Then proceed with::

     python3 setup.py config

After executing config, check that numpy has correctly found the
MKL install.  You should see something like this:

::

    (mkl)login2$ python3 setup.py config
    Running from numpy source directory.
    F2PY Version 2
    blas_opt_info:
    blas_mkl_info:
      FOUND:
        define_macros = [('SCIPY_MKL_H', None)]
        libraries = ['mkl_rt', 'pthread']
        include_dirs = ['/opt/apps/intel/13/composer_xe_2013.2.146/mkl/include/']
        library_dirs = ['/opt/apps/intel/13/composer_xe_2013.2.146/mkl/lib/intel64/']

      FOUND:
        define_macros = [('SCIPY_MKL_H', None)]
        libraries = ['mkl_rt', 'pthread']
          include_dirs = ['/opt/apps/intel/13/composer_xe_2013.2.146/mkl/include/']
          library_dirs = ['/opt/apps/intel/13/composer_xe_2013.2.146/mkl/lib/intel64/']

      non-existing path in 'numpy/lib': 'benchmarks'
      lapack_opt_info:
      openblas_info:
        libraries openblas not found in ['/home1/00364/tg456434/venv/mkl/lib', '/usr/local/lib64', '/usr/local/lib', '/usr/lib64', '/usr/lib']
        NOT AVAILABLE

    lapack_mkl_info:
    mkl_info:
      FOUND:
        define_macros = [('SCIPY_MKL_H', None)]
        libraries = ['mkl_rt', 'pthread']
        include_dirs = ['/opt/apps/intel/13/composer_xe_2013.2.146/mkl/include/']
        library_dirs = ['/opt/apps/intel/13/composer_xe_2013.2.146/mkl/lib/intel64/']

      FOUND:
        define_macros = [('SCIPY_MKL_H', None)]
        libraries = ['mkl_rt', 'pthread']
        include_dirs = ['/opt/apps/intel/13/composer_xe_2013.2.146/mkl/include/']
        library_dirs = ['/opt/apps/intel/13/composer_xe_2013.2.146/mkl/lib/intel64/']

      FOUND:
        define_macros = [('SCIPY_MKL_H', None)]
        libraries = ['mkl_rt', 'pthread']
        include_dirs = ['/opt/apps/intel/13/composer_xe_2013.2.146/mkl/include/']
        library_dirs = ['/opt/apps/intel/13/composer_xe_2013.2.146/mkl/lib/intel64/']

    /home1/00364/tg456434/build/lib/python3.3/distutils/dist.py:257: UserWarning: Unknown distribution option: 'define_macros'
      warnings.warn(msg)
    running config
    (mkl)login2$

Next do::

     python3 setup.py build
     python3 setup.py install

Test that things worked by launching ``python3`` and then doing::

     import numpy as np
     np.__config__.show()

If you've installed ``nose`` (with ``pip3 install nose``), 
we can further test our numpy build with::

     np.test()
     np.test('full')

We pass ``np.test()`` with no errors (takes roughly 54 seconds); we
have one failure on ``np.test('full')`` ::

      ======================================================================
      FAIL: test_allnans (test_nanfunctions.TestNanFunctions_Sum)
      ----------------------------------------------------------------------
      Traceback (most recent call last):
        File "/home1/00364/tg456434/venv/mkl/lib/python3.3/site-packages/numpy/lib/tests/test_nanfunctions.py", line 308, in test_allnans
          assert_(len(w) == 1, 'no warning raised')
        File "/home1/00364/tg456434/venv/mkl/lib/python3.3/site-packages/numpy/testing/utils.py", line 44, in assert_
          raise AssertionError(msg)
      AssertionError: no warning raised

      ----------------------------------------------------------------------
      Ran 5000 tests in 253.836s

      FAILED (KNOWNFAIL=6, SKIP=4, failures=1)
      <nose.result.TextTestResult run=5000 errors=0 failures=1>
      >>> 

This is the same error as in the OpenBLAS install
(:doc:`stampede_openblas`).  Also, overall test times are very similar
for the full test.


MKL BLAS failure
=====================

Though we've built numpy successfully, it has failed to link properly
against the BLAS libraries in MKL.  The TACC python 2.7 build does
work correctly ::

    login3$ ./numpy_test
    FAST BLAS
    ('version:', '1.6.1')
    ()
    ('dot:', 0.16252517700195312, 'sec')
    login3$ python
    Enthought Python Distribution -- www.enthought.com
    Version: 7.3-2 (64-bit)

    Python 2.7.3 |EPD 7.3-2 (64-bit)| (default, Apr 11 2012, 17:52:16) 
    [GCC 4.1.2 20080704 (Red Hat 4.1.2-44)] on linux2
    Type "credits", "demo" or "enthought" for more information.
    >>> import numpy as np
    >>> np.__config__.show()
    lapack_opt_info:
        libraries = ['mkl_lapack95_lp64', 'mkl_intel_lp64', 'mkl_intel_thread', 'mkl_core', 'iomp5', 'pthread']
        library_dirs = ['/home/builder/master/lib']
        define_macros = [('SCIPY_MKL_H', None)]
        include_dirs = ['/home/builder/master/include']
    blas_opt_info:
        libraries = ['mkl_intel_lp64', 'mkl_intel_thread', 'mkl_core', 'iomp5', 'pthread']
        library_dirs = ['/home/builder/master/lib']
        define_macros = [('SCIPY_MKL_H', None)]
        include_dirs = ['/home/builder/master/include']
    lapack_mkl_info:
        libraries = ['mkl_lapack95_lp64', 'mkl_intel_lp64', 'mkl_intel_thread', 'mkl_core', 'iomp5', 'pthread']
        library_dirs = ['/home/builder/master/lib']
        define_macros = [('SCIPY_MKL_H', None)]
        include_dirs = ['/home/builder/master/include']
    blas_mkl_info:
        libraries = ['mkl_intel_lp64', 'mkl_intel_thread', 'mkl_core', 'iomp5', 'pthread']
        library_dirs = ['/home/builder/master/lib']
        define_macros = [('SCIPY_MKL_H', None)]
        include_dirs = ['/home/builder/master/include']
    mkl_info:
        libraries = ['mkl_intel_lp64', 'mkl_intel_thread', 'mkl_core', 'iomp5', 'pthread']
        library_dirs = ['/home/builder/master/lib']
        define_macros = [('SCIPY_MKL_H', None)]
        include_dirs = ['/home/builder/master/include']
    >>> 


Whereas my python 3.3 build does not ::

    (mkl)c557-703$ ./numpy_test
    slow blas
    version: 1.8.0

    dot: 0.995588791416958 sec
    (mkl)c557-703$ python3
    Python 3.3.3 (default, Jan  8 2014, 11:50:50) 
    [GCC 4.7.1] on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import numpy as np
    >>> np.__config__.show()
    lapack_opt_info:
        include_dirs = ['/opt/apps/intel/13/composer_xe_2013.2.146/mkl/include/']
        library_dirs = ['/opt/apps/intel/13/composer_xe_2013.2.146/mkl/lib/intel64/']
        libraries = ['mkl_rt', 'pthread']
        define_macros = [('SCIPY_MKL_H', None)]
    lapack_mkl_info:
        include_dirs = ['/opt/apps/intel/13/composer_xe_2013.2.146/mkl/include/']
        library_dirs = ['/opt/apps/intel/13/composer_xe_2013.2.146/mkl/lib/intel64/']
        libraries = ['mkl_rt', 'pthread']
        define_macros = [('SCIPY_MKL_H', None)]
    blas_opt_info:
        include_dirs = ['/opt/apps/intel/13/composer_xe_2013.2.146/mkl/include/']
        library_dirs = ['/opt/apps/intel/13/composer_xe_2013.2.146/mkl/lib/intel64/']
        libraries = ['mkl_rt', 'pthread']
        define_macros = [('SCIPY_MKL_H', None)]
    blas_mkl_info:
        include_dirs = ['/opt/apps/intel/13/composer_xe_2013.2.146/mkl/include/']
        library_dirs = ['/opt/apps/intel/13/composer_xe_2013.2.146/mkl/lib/intel64/']
        libraries = ['mkl_rt', 'pthread']
      define_macros = [('SCIPY_MKL_H', None)]
  mkl_info:
      include_dirs = ['/opt/apps/intel/13/composer_xe_2013.2.146/mkl/include/']
      library_dirs = ['/opt/apps/intel/13/composer_xe_2013.2.146/mkl/lib/intel64/']
      libraries = ['mkl_rt', 'pthread']
      define_macros = [('SCIPY_MKL_H', None)]
  openblas_info:
    NOT AVAILABLE
  >>> 
