Final steps
=============

After you've successfully built your full software stack, it's time to
build dedalus2 (likely in your virtualenv).

First, set your ``FFTW_PATH`` and ``MPI_PATH`` environment variables
(see system specific documentation). 
Then change into your root dedalus directory and run::

     python setup.py build_ext --inplace

Dedalus2 is now ready to run
