"""
Interface for accessing all submodules.

"""

from .data import distributor
from .data import domain
from .data import field
from .data import operators
from .data import pencil
from .data import system

from .pde import basis
from .pde import nonlinear
from .pde import problems
from .pde import solvers
from .pde import timesteppers

from .tools import array
from .tools import cache
from .tools import config
from .tools import dispatch
from .tools import general
from .tools import logging
from .tools import parallel

