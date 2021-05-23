"""
Tools for tracking iteration progress.

"""

import sys
import time
import logging
import numpy as np
import math


def log_progress(iterable, logger, level, **kw):
    """Log iteration progress."""

    if isinstance(level, str):
        try:  # Python 3.4
            level = logging._nameToLevel[level.upper()]
        except AttributeError:  # Python < 3.4
            level = logging.getLevelName(level.upper())
    def write(message):
        logger.log(level, message)
    yield from track(iterable, write, **kw)


def print_progress(iterable, stream=sys.stdout, **kw):
    """Print iteration progress to a stream."""

    def write(message):
        if stream:
            print(message, file=stream, flush=True)
    yield from track(iterable, write, **kw)


def track(iterable, write, desc='Iteration', iter=1, frac=1., dt=np.inf):
    """Track an iterator attaching messages at set cadences."""

    length = len(iterable)
    iter = min(iter, math.ceil(frac*length))

    start_time = time.time()
    last_iter_div = -1
    last_time_div = -1

    for index, item in enumerate(iterable):

        yield item
        elapsed_time = time.time() - start_time
        completed = index + 1

        time_div = elapsed_time // dt
        iter_div = completed // iter
        scheduled = ((time_div > last_time_div) or (iter_div > last_iter_div))
        last = (completed == length)

        if (scheduled or last):
            # Update divisors
            last_time_div = time_div
            last_iter_div = iter_div
            # Compute statistics
            percent = round(100 * completed / length)
            rate = completed / elapsed_time
            projected_time = length / rate
            remaining_time = projected_time - elapsed_time
            # Build and write message
            message = [desc]
            message.append('{:d}/{:d} (~{:d}%)'.format(completed, length, percent))
            message.append('Elapsed: {:s}, Remaining: {:s}, Rate: {:.1e}/s'.format(
                format_time(elapsed_time), format_time(remaining_time), rate))
            message = ' '.join(message)
            write(message)


def format_time(total_sec):
    """Format time strings."""

    total_min, sec = divmod(round(total_sec), 60)
    hr, min = divmod(total_min, 60)

    if hr:
        return '{:d}h {:02d}m {:02d}s'.format(hr, min, sec)
    elif min:
        return '{:d}m {:02d}s'.format(min, sec)
    else:
        return '{:d}s'.format(sec)

