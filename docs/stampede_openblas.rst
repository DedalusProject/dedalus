Stampede OpenBLAS libraries
************************************

Rather than only using the build-in MKL libraries (:doc:`stampede_mkl`), 
we may also wish to build and test against OpenBLAS.

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
paths).  With my account settings, this looks like::

     [openblas]
     libraries = openblas
     library_dirs = /home1/00364/tg456434/build/lib
     include_dirs = /home1/00364/tg456434/build/include



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

If you've installed ``nose`` (with ``pip3 install nose``), 
we can further test our numpy build with::

     np.test()
     np.test('full')

We pass ``np.test()`` with no errors (takes roughly 80 seconds); we have one failure on ``np.test('full')`` ::

      ======================================================================
      FAIL: test_allnans (test_nanfunctions.TestNanFunctions_Sum)
      ----------------------------------------------------------------------
      Traceback (most recent call last):
        File "/home1/00364/tg456434/venv/openblas/lib/python3.3/site-packages/numpy/lib/tests/test_nanfunctions.py", line 308, in test_allnans
          assert_(len(w) == 1, 'no warning raised')
        File "/home1/00364/tg456434/venv/openblas/lib/python3.3/site-packages/numpy/testing/utils.py", line 44, in assert_
          raise AssertionError(msg)
      AssertionError: no warning raised

      ----------------------------------------------------------------------
      Ran 5000 tests in 280.081s

      FAILED (KNOWNFAIL=6, SKIP=3, failures=1)
      <nose.result.TextTestResult run=5000 errors=0 failures=1>
      >>> 


*Note: if on ``import numpy as np`` you get an error on loading the
OpenBLAS shared library, see above note about ``$LD_LIBRARY_PATH``.*


