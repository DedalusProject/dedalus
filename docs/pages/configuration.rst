..  _configuration:

Configuring Dedalus
*******************

Various aspects of the underlying numerics, logging behavior, and output behavior of Dedalus can be modified through a configuration interface using Python's standard-library `ConfigParser` structure.

A ``dedalus.cfg`` file with the default configuration and descriptions of each option is included in the package.
This file can be copied to any working directory with the command ``python3 -m dedalus get_config``.
The configuration settings can be modified by, in order of increasing precedence:

1. Modifying the package's ``dedalus.cfg`` file (not recommended).
2. Creating a user-based config file at ``~/.dedalus/dedalus.cfg``.
3. Creating a local config file called ``dedalus.cfg`` in the directory where you execute a Dedalus script.
4. Modifying configuration values at the top of a Dedalus script like:

.. code-block:: python

    from dedalus.tools.config import config
    config['logging']['stdout_level'] = 'debug'

The default configuration file in this version of Dedalus is copied below for reference:

.. literalinclude:: ../../dedalus/dedalus.cfg
