Install notes for TACC/Stampede
***************************************************************************

On Stampede, we can either install with a ``GCC/mpvapich2/fftw3``
stack with ``OpenBLAS``, or with an ``intel/impi/fftw3 stack`` with
``MKL``.  Mpvaich2 seems to be causing problems for  us, and this
appears to be a known issue with ``mvapich2/1.9``, hence the attempt
to build against ``impi``.  The intel stack should also, in principle,
allow us to explore auto-offloading with the Xenon MIC hardware
accelerators.

.. note::
    The intel stack can probably support both ``MKL`` and ``OpenBLAS``; so
    far the gcc stack only robustly supports ``OpenBLAS``.

Detailed notes appear below.

.. toctree::
   :maxdepth: 1

   stampede_gcc
   stampede_intel
