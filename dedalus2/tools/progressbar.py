

import sys
import time
import math


def format_time(total_sec):
    """Format time strings."""

    total_min, sec = divmod(round(total_sec), 60)
    hr, min = divmod(total_min, 60)

    if hr:
        return '{:d}h {:02d}m {:02d}s'.format(hr, min, sec)
    if min:
        return '{:d}m {:02d}s'.format(min, sec)
    return '{:d}s'.format(sec)


def bar(fraction, length=10, done='#', left='-'):
    """Create progress bar string."""

    n_done = math.floor(fraction * length)
    n_left = length - n_done

    return '[' + n_done*done + n_left*left + ']'


class Printer:
    """Wrapper for updating single line in a stream."""

    def __init__(self, stream):
        self.stream = stream
        self.length = 0

    def write(self, message):
        extra = max(0, self.length - len(message))
        self.stream.write('\r' + message + ' '*extra)
        self.stream.flush()
        self.length = len(message)


def progress(iterable, iter=1, dt=1., desc='', stream=sys.stdout, bar=True, erase=True, write=True):
    """Wrap iterable, printing progress bar during iteration."""

    printer = Printer(stream)
    length = len(iterable)
    start_time = time.time()
    last_iter_div = -1
    last_time_div = -1

    completed = 0
    for item in iterable:
        yield item
        completed += 1
        runtime = time.time() - start_time

        time_div = runtime // dt
        iter_div = completed // iter
        scheduled = ((time_div > last_time_div) and (iter_div > last_iter_div))
        if not (write and (scheduled or (completed==length))):
            continue
        last_time_div = time_div
        last_iter_div = iter_div

        # Statistics
        percent = round(100 * completed / length)
        rate = completed / runtime
        endtime = length / rate
        remaining = endtime - runtime

        # Message
        message = []
        if desc:
            message.append(desc)
        if bar:
            message.append(bar(completed/length))
        message.append('{:d}/{:d} = {:d}%'.format(completed, length, percent))
        message.append('(Elapsed: {:s}, Remaining: {:s}, Rate: {:.1e}/s)'.format(
            format_time(runtime), format_time(remaining), rate))
        message = ' '.join(message)
        printer.write(message)

    if write:
        if erase:
            printer.write('')
            printer.write('')
        else:
            stream.write('\n')


if __name__ == '__main__':

    for i in progressbar(range(10000), desc='Test'):
        time.sleep(0.001)

