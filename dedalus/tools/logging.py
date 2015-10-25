"""
Logging setup.

"""

import pathlib
import logging
from mpi4py import MPI
import os
import sys

from ..tools.config import config
from ..tools.parallel import Sync

MPI_RANK = MPI.COMM_WORLD.rank
MPI_SIZE = MPI.COMM_WORLD.size


# Load config options
stdout_level = config['logging']['stdout_level']
file_level = config['logging']['file_level']
nonroot_level = config['logging']['nonroot_level']
filename = config['logging']['filename']

# Root logger config
rootlogger = logging.root
rootlogger.setLevel(0)
if nonroot_level.lower() != 'none':
    if MPI_RANK > 0:
        rootlogger.setLevel(getattr(logging, nonroot_level.upper()))

# Formatter
formatter = logging.Formatter('%(asctime)s %(name)s {}/{} %(levelname)s :: %(message)s'.format(MPI_RANK, MPI_SIZE))

# Stream handler
if stdout_level.lower() != 'none':
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(getattr(logging, stdout_level.upper()))
    stdout_handler.setFormatter(formatter)
    rootlogger.addHandler(stdout_handler)

# File handler
if file_level.lower() != 'none':
    file_path = pathlib.Path('%s_p%i.log' %(filename, MPI_RANK))
    with Sync(MPI.COMM_WORLD):
        if MPI_RANK == 0:
            if not file_path.parent.exists():
                file_path.parent.mkdir(parents=True)
    file_handler = logging.FileHandler(str(file_path), mode='w')
    file_handler.setLevel(getattr(logging, file_level.upper()))
    file_handler.setFormatter(formatter)
    rootlogger.addHandler(file_handler)

