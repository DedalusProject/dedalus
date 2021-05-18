Full-stack installation shell script
************************************

This all-in-one installation script will build an isolated stack containing a Python installation and the other dependencies needed to run Dedalus.
In most cases, the script can be modified to link with system installations of FFTW, MPI, and linear algebra libraries.

You can get down the installation script using::

    wget https://raw.githubusercontent.com/DedalusProject/dedalus/master/docs/install.sh

and execute it using::

    bash install.sh

The installation script has been tested on a number of Linux distributions and OS X in the past, but is generally no longer supported.
Please consider using the conda-based installation instead.

