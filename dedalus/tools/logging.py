"""Logging setup."""

import sys
import pathlib
import logging
from mpi4py import MPI
from ..tools.config import config

MPI_RANK = MPI.COMM_WORLD.rank
MPI_SIZE = MPI.COMM_WORLD.size


# Root logger config
rootlogger = logging.root
rootlogger.setLevel(0)
if config['logging']['nonroot_level'].lower() != 'none':
    if MPI_RANK > 0:
        rootlogger.setLevel(getattr(logging, config['logging']['nonroot_level'].upper()))

# Formatter
formatter = logging.Formatter('%(asctime)s %(name)s {}/{} %(levelname)s :: %(message)s'.format(MPI_RANK, MPI_SIZE))

# Stream handler
def add_stdout_handler(level, formatter=formatter):
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(level.upper())
    stdout_handler.setFormatter(formatter)
    rootlogger.addHandler(stdout_handler)
    return stdout_handler

if config['logging']['stdout_level'].lower() != 'none':
    stdout_handler = add_stdout_handler(config['logging']['stdout_level'])

# File handler
def add_file_handler(filename, level, formatter=formatter):
    # Modify path for each process
    file_path = pathlib.Path(f'{filename}_p{MPI_RANK}.log')
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(str(file_path), mode='w')
    file_handler.setLevel(level.upper())
    file_handler.setFormatter(formatter)
    rootlogger.addHandler(file_handler)
    return file_handler

if config['logging']['file_level'].lower() != 'none':
    file_handler = add_file_handler(config['logging']['filename'], config['logging']['file_level'])

