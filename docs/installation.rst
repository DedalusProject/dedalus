Installing Dedalus
**************************

Preliminaries
====================

Dedalus relies on Python3 and is distributed via a mercurial distributed
version control system. To install dedalus, first install `mercurial`_.
Then, at the command line and in an appropriate code development
directory, clone the repository. The clone command can be copied from
the ``clone`` button on the main repository page, and is also included
here::

    hg clone https://bitbucket.org/dedalus-project/dedalus2

Example problems are included in a separate repository ``examples2``
and can be cloned using::

    hg clone https://bitbucket.org/dedalus-project/examples2


(note: the ``clone`` button on the wiki pages will clone the dedalus
wiki rather than dedalus itself; they are maintained as seperate
repositories within the bitbucket system).

.. _mercurial: http://mercurial.selenic.com


Python 3 stack
====================
Here are detailed instructions on installing the 
full python3 software stack on a variety of machines.

Supercomputers

.. toctree::
   :maxdepth: 1

   machines/stampede/stampede
   machines/nasa_pleiades/pleiades
   machines/trestles/trestles

Development machines

.. toctree::
   :maxdepth: 1

   machines/mac_os/mac_os


Final steps
====================

After you've successfully built your full software stack, it's time to
build dedalus2 (likely in your virtualenv).

First, set your ``FFTW_PATH`` and ``MPI_PATH`` environment variables
(see system specific documentation). 
Then change into your root dedalus directory and run::

     python3 setup.py build_ext --inplace

Dedalus2 is now ready to run
