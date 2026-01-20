# Copyright (c) 2013-2022, Keaton J. Burns.
#
# This file is part of Dedalus, which is free software distributed
# under the terms of the GPLv3 license.  A copy of the license should
# have been included in the file 'LICENSE.txt', and is also available
# online at <http://www.gnu.org/licenses/gpl-3.0.html>.

__version__ = "3.0.5"

# Import custom logging to setup rootlogger
from .tools import logging as _logging_setup
import logging
logger = logging.getLogger(__name__.split('.')[-1])

# Warn if threading is not disabled
import os
if os.getenv("OMP_NUM_THREADS") != "1":
    logger.warning('Threading has not been disabled. This may massively degrade Dedalus performance.')
    logger.warning('We strongly suggest setting the "OMP_NUM_THREADS" environment variable to "1".')

# Set numexpr threading to match OMP_NUM_THREADS to supress warning
# Could remove pending https://github.com/pydata/numexpr/issues/344
if os.getenv("NUMEXPR_MAX_THREADS") is None and os.getenv("OMP_NUM_THREADS") is not None:
    os.environ["NUMEXPR_MAX_THREADS"] = os.environ["OMP_NUM_THREADS"]
