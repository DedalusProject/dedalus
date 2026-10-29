"""ROCtx range decorator for use with rocprofv3 --marker-trace."""
import ctypes
import functools
import sys
_lib = None
_lib_tried = False


def _get_lib():
    """Load the rocprofiler-sdk ROCtx library once; return None if unavailable."""
    global _lib, _lib_tried
    if not _lib_tried:
        _lib_tried = True
        try:
            lib = ctypes.CDLL("librocprofiler-sdk-roctx.so")
            lib.roctxRangePushA.argtypes = [ctypes.c_char_p]
            lib.roctxRangePushA.restype = ctypes.c_int
            lib.roctxRangePop.restype = ctypes.c_int
            _lib = lib
        except OSError:
            _lib = None  # run without markers (decorator becomes a no-op)
    return _lib


def roctx_range(arg=None, *, sync=False):
    """Wrap a function or method in a ROCtx range.

    Usage:
        @roctx_range                      # range named after the method
        @roctx_range("my_label")          # custom name
        @roctx_range(sync=True)           # synchronize GPU before closing range
        @roctx_range("my_label", sync=True)

    With sync=True the range covers GPU execution (CuPy launches are async),
    not just the CPU-side launch time.
    """
    label = arg if isinstance(arg, str) else None

    def decorate(func):
        name = (label or func.__qualname__).encode()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            lib = _get_lib()
            if lib is None:
                return func(*args, **kwargs)
            depth = lib.roctxRangePushA(name)
            try:
                result = func(*args, **kwargs)
                if sync:
                    cp.cuda.Device().synchronize()
                return result
            finally:
                r = lib.roctxRangePop()
                # print(f"[roctx] {name.decode()}: push={depth} pop={r}", file=sys.stderr)

        return wrapper

    if callable(arg):  # used as bare @roctx_range
        return decorate(arg)
    return decorate


if __name__ == "__main__":
    import cupy as cp
    import cupyx.scipy.fft as cufft

    class Solver:
        def __init__(self, n=512):
            self.data = cp.ones((n, n, n), dtype=cp.float32)

        @roctx_range(sync=True)
        def transform(self, axis=1):
            return cufft.irfft(self.data, axis=axis, n=self.data.shape[axis])

        @roctx_range("custom_name", sync=True)
        def copy(self):
            return self.data.copy()

    s = Solver()
    s.transform()          # warm-up (plan creation, allocation)
    s.transform()
    s.copy()
