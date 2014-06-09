"""
Logging setup.

"""

import logging
from mpi4py import MPI
import os
import sys

from ..tools.config import config
from ..tools.parallel import Sync


# Load config options
stdout_level_str = config['logging']['stdout_level']
file_level_str = config['logging']['file_level']
nonroot_level_str = config['logging']['nonroot_level']
filename = config['logging']['filename']

# Lookup levels
stdout_level = getattr(logging, stdout_level_str.upper())
file_level = getattr(logging, file_level_str.upper())
nonroot_level = getattr(logging, nonroot_level_str.upper())

# Determine base levels
if MPI.COMM_WORLD.rank == 0:
    base_level = min(stdout_level, file_level)
else:
    base_level = max(nonroot_level, min(stdout_level, file_level))

# Base logger
baselogger = logging.getLogger('Dedalus2')
baselogger.setLevel(base_level)

# Formatter
formatter = logging.Formatter('%(asctime)s %(name)s %(rank)s/%(size)s %(levelname)-7s : %(message)s')

# Stream handler
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(stdout_level)
stdout_handler.setFormatter(formatter)
baselogger.addHandler(stdout_handler)

# File handler
if filename.lower() != 'none':

    # Delete previous log file from rank 0
    with Sync():
        if MPI.COMM_WORLD.rank == 0:
            if os.path.exists(filename):
                os.remove(filename)

    file_handler = logging.FileHandler(filename)
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    baselogger.addHandler(file_handler)

# Adapter with MPI information
sizestr = str(MPI.COMM_WORLD.size)
rankstr = str(MPI.COMM_WORLD.rank).rjust(len(sizestr))
mpi_extras = {'rank': rankstr, 'size': sizestr}
logger = logging.LoggerAdapter(baselogger, mpi_extras)

