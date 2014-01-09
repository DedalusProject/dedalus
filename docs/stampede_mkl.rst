Stampede MKL libraries
************************************

On Stampede we will likely generally use the Intel MKL libraries for
BLAS; these are loaded through a module command and should be
available by default.  

.. note ::

   We have a ticket open with TACC to start
   resolving problems with numpy builds against MKL.

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
     mkl_libs = mkl_gnu_thread


Then proceed with::

     python3 setup.py config

After executing config, check that numpy has correctly found the
OpenBLAS install.  You should see something like this:

::

    (mkl)login2$ python3 setup.py config
    Running from numpy source directory.
    F2PY Version 2
    blas_opt_info:
    blas_mkl_info:
      FOUND:
        libraries = ['mkl_gnu_thread', 'pthread']
        library_dirs = ['/opt/apps/intel/13/composer_xe_2013.2.146/mkl/lib/intel64/']
        include_dirs = ['/opt/apps/intel/13/composer_xe_2013.2.146/mkl/include']
        define_macros = [('SCIPY_MKL_H', None)]

      FOUND:
          libraries = ['mkl_gnu_thread', 'pthread']
          library_dirs = ['/opt/apps/intel/13/composer_xe_2013.2.146/mkl/lib/intel64/']
          include_dirs = ['/opt/apps/intel/13/composer_xe_2013.2.146/mkl/include']
          define_macros = [('SCIPY_MKL_H', None)]

      non-existing path in 'numpy/lib': 'benchmarks'
      lapack_opt_info:
      openblas_info:
        libraries openblas not found in ['/home1/00364/tg456434/venv/mkl/lib', '/usr/local/lib64', '/usr/local/lib', '/usr/lib64', '/usr/lib']
        NOT AVAILABLE

      lapack_mkl_info:
      mkl_info:
        FOUND:
          libraries = ['mkl_gnu_thread', 'pthread']
          library_dirs = ['/opt/apps/intel/13/composer_xe_2013.2.146/mkl/lib/intel64/']
          include_dirs = ['/opt/apps/intel/13/composer_xe_2013.2.146/mkl/include']
          define_macros = [('SCIPY_MKL_H', None)]

        FOUND:
          libraries = ['mkl_lapack32', 'mkl_lapack64', 'mkl_gnu_thread', 'pthread']
          library_dirs = ['/opt/apps/intel/13/composer_xe_2013.2.146/mkl/lib/intel64/']
          include_dirs = ['/opt/apps/intel/13/composer_xe_2013.2.146/mkl/include']
          define_macros = [('SCIPY_MKL_H', None)]

        FOUND:
          libraries = ['mkl_lapack32', 'mkl_lapack64', 'mkl_gnu_thread', 'pthread']
          library_dirs = ['/opt/apps/intel/13/composer_xe_2013.2.146/mkl/lib/intel64/']
          include_dirs = ['/opt/apps/intel/13/composer_xe_2013.2.146/mkl/include']
          define_macros = [('SCIPY_MKL_H', None)]

      /home1/00364/tg456434/build/lib/python3.3/distutils/dist.py:257: UserWarning: Unknown distribution option: 'define_macros'
        warnings.warn(msg)
      running config
      (mkl)login2$ 


Next do::

     python3 setup.py build
     python3 setup.py install

We're hitting an error on ``install``::

      gcc -pthread -shared build/temp.linux-x86_64-3.3/numpy/linalg/lapack_litemodule.o build/temp.linux-x86_64-3.3/numpy/linalg/lapack_lite/python_xerbla.o -L/opt/apps/intel/13/composer_xe_2013.2.146/mkl/lib/intel64/ -Lbuild/temp.linux-x86_64-3.3 -lmkl_lapack32 -lmkl_lapack64 -lmkl_gnu_thread -lpthread -o build/lib.linux-x86_64-3.3/numpy/linalg/lapack_lite.cpython-33m.so
      /usr/bin/ld: cannot find -lmkl_lapack32
      collect2: error: ld returned 1 exit status
      /usr/bin/ld: cannot find -lmkl_lapack32
      collect2: error: ld returned 1 exit status
      error: Command "gcc -pthread -shared build/temp.linux-x86_64-3.3/numpy/linalg/lapack_litemodule.o build/temp.linux-x86_64-3.3/numpy/linalg/lapack_lite/python_xerbla.o -L/opt/apps/intel/13/composer_xe_2013.2.146/mkl/lib/intel64/ -Lbuild/temp.linux-x86_64-3.3 -lmkl_lapack32 -lmkl_lapack64 -lmkl_gnu_thread -lpthread -o build/lib.linux-x86_64-3.3/numpy/linalg/lapack_lite.cpython-33m.so" failed with exit status 1

Test that things worked by launching ``python3`` and then doing::

     import numpy as np
     np.__config__.show()

If you've installed ``nose`` (with ``pip3 install nose``), 
we can further test our numpy build with::

     np.test()
     np.test('full')
