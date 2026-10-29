"""
Inverse real FFT of packed coefficients, written straight into the grid array
Entry point: irfft_packed_into(cdata, gdata, axis, kmax, scale=1.0)

Last axis: pack kernel -> C2R plan into gdata.
Other axes: fused pack transpose kernel -> C2R plan -> transpose kernel into gdata.

The transpose kernels have their sizes compiled in, so each (shape, dtype, axis, packed
length, kmax) builds its own kernels, buffers and plan once.
"""
import math
import warnings
from functools import lru_cache

import numpy as np
import cupy as cp
from cupy.cuda import cufft as cufft_backend
from cupyx import jit

warnings.filterwarnings('ignore', message='cupyx.jit.rawkernel is experimental',
                        category=FutureWarning)

THREADS = 256
# Tile (rows, cols), powers of two; measured fastest.
REAL_TILE = (32, 32)
SPECTRUM_TILE = {np.float64: (32, 32), np.float32: (32, 32)}


@lru_cache(maxsize=None)
def _transpose_kernel(real_type, rows, cols):
    """Slow axes: dst[m] (cols x rows) = src[m] (rows x cols) transposed."""
    tile_rows, tile_cols = REAL_TILE
    col_shift, col_mask = tile_cols.bit_length() - 1, tile_cols - 1
    row_shift, row_mask = tile_rows.bit_length() - 1, tile_rows - 1
    pitch = tile_cols + 1                        # odd: no shared-memory bank conflicts
    per_thread = tile_rows * tile_cols // THREADS
    row_tiles = math.ceil(rows / tile_rows)
    col_tiles = math.ceil(cols / tile_cols)
    # The longer side goes on grid x (grid y is limited to 65535); this order is
    # also much faster for very long rows (4.5x for a 524288 x 1024 transpose).
    rows_on_grid_x = rows > cols
    grid = (row_tiles, col_tiles) if rows_on_grid_x else (col_tiles, row_tiles)

    @jit.rawkernel()
    def transpose(src, dst):
        tile = jit.shared_memory(real_type, tile_rows * pitch)
        thread = np.int32(jit.threadIdx.x)       # signed: jit rejects unsigned/signed mixes
        if rows_on_grid_x:
            first_row = np.int64(jit.blockIdx.x) * tile_rows
            first_col = np.int64(jit.blockIdx.y) * tile_cols
        else:
            first_col = np.int64(jit.blockIdx.x) * tile_cols
            first_row = np.int64(jit.blockIdx.y) * tile_rows
        matrix_start = np.int64(jit.blockIdx.z) * rows * cols

        for step in range(per_thread):           # read along columns
            index = thread + step * THREADS
            tile_row = index >> col_shift
            tile_col = index & col_mask
            row = first_row + tile_row
            col = first_col + tile_col
            if row < rows and col < cols:
                tile[tile_row * pitch + tile_col] = src[matrix_start + row * cols + col]
        jit.syncthreads()

        for step in range(per_thread):           # write along rows
            index = thread + step * THREADS
            tile_col = index >> row_shift
            tile_row = index & row_mask
            row = first_row + tile_row
            col = first_col + tile_col
            if row < rows and col < cols:
                dst[matrix_start + col * rows + row] = tile[tile_row * pitch + tile_col]

    return transpose, grid


@lru_cache(maxsize=None)
def _packed_transpose_kernel(real_type, n_modes, cols, packed_len, kmax):
    """Slow axes: packed (packed_len x cols) -> transposed half spectrum (cols x n_modes), interleaved reals."""
    tile_rows, tile_cols = SPECTRUM_TILE[real_type]
    col_shift, col_mask = tile_cols.bit_length() - 1, tile_cols - 1
    row_shift, row_mask = tile_rows.bit_length() - 1, tile_rows - 1
    pitch = tile_cols + 1
    plane = tile_rows * pitch                    # real part, then imaginary part
    per_thread = tile_rows * tile_cols // THREADS
    mode_tiles = math.ceil((kmax + 1) / tile_rows)
    col_tiles = math.ceil(cols / tile_cols)
    modes_on_grid_x = kmax + 1 > cols
    grid = (mode_tiles, col_tiles) if modes_on_grid_x else (col_tiles, mode_tiles)

    @jit.rawkernel()
    def packed_transpose(packed, spectrum, mean_weight, mode_weight):
        tile = jit.shared_memory(real_type, 2 * plane)
        thread = np.int32(jit.threadIdx.x)
        if modes_on_grid_x:
            first_mode = np.int64(jit.blockIdx.x) * tile_rows
            first_col = np.int64(jit.blockIdx.y) * tile_cols
        else:
            first_col = np.int64(jit.blockIdx.x) * tile_cols
            first_mode = np.int64(jit.blockIdx.y) * tile_rows
        matrix = np.int64(jit.blockIdx.z)

        for step in range(per_thread):
            index = thread + step * THREADS
            tile_mode = index >> col_shift
            tile_col = index & col_mask
            position = tile_mode * pitch + tile_col
            mode = first_mode + tile_mode
            col = first_col + tile_col
            if mode <= kmax and col < cols:
                if mode == 0:
                    tile[position] = packed[matrix * packed_len * cols + col] * mean_weight
                    tile[plane + position] = 0
                else:
                    cos_index = (matrix * packed_len + 2 * mode) * cols + col
                    tile[position] = packed[cos_index] * mode_weight
                    tile[plane + position] = packed[cos_index + cols] * mode_weight
        jit.syncthreads()

        for step in range(per_thread):
            index = thread + step * THREADS
            tile_col = index >> row_shift
            tile_mode = index & row_mask
            position = tile_mode * pitch + tile_col
            mode = first_mode + tile_mode
            col = first_col + tile_col
            if mode <= kmax and col < cols:
                out = ((matrix * cols + col) * n_modes + mode) * 2
                spectrum[out] = tile[position]
                spectrum[out + 1] = tile[plane + position]

    return packed_transpose, grid


@jit.rawkernel()
def _pack_kernel(packed, spectrum, mean_weight, mode_weight, packed_len, n_modes, kmax,
                 n_elements):
    """Fast axis: (rows x packed_len) packed -> (rows x n_modes) interleaved spectrum."""
    element = np.int64(jit.blockIdx.x) * np.int64(jit.blockDim.x) + np.int64(jit.threadIdx.x)
    if element < n_elements:
        row = element // n_modes
        mode = element - row * n_modes
        if mode == 0:
            spectrum[2 * element] = packed[row * packed_len] * mean_weight
            spectrum[2 * element + 1] = 0
        elif mode <= kmax:
            cos_index = row * packed_len + 2 * mode
            spectrum[2 * element] = packed[cos_index] * mode_weight
            spectrum[2 * element + 1] = packed[cos_index + 1] * mode_weight


class _PackedC2R:
    def __init__(self, shape, dtype, axis, packed_len, kmax):
        self.real_type = dtype.type
        is_double = dtype == np.float64
        self.n_grid = shape[axis]
        n_modes = self.n_grid // 2 + 1
        n_leading = math.prod(shape[:axis])              # matrices before the axis
        n_trailing = math.prod(shape[axis + 1:])         # contiguous elements after it
        self.n_transforms = n_leading * n_trailing
        self.plan = cufft_backend.Plan1d(
            self.n_grid, cufft_backend.CUFFT_Z2D if is_double else cufft_backend.CUFFT_C2R,
            self.n_transforms)
        spectrum = cp.zeros((self.n_transforms, n_modes),
                            np.complex128 if is_double else np.complex64)
        self.spectrum = spectrum
        self.spectrum_reals = spectrum.reshape(-1).view(self.real_type)
        block = (THREADS, 1, 1)

        if n_trailing == 1:                              # last axis
            n_elements = self.n_transforms * n_modes
            self.pack = _pack_kernel[(math.ceil(n_elements / THREADS), 1, 1), block]
            self.pack_sizes = tuple(np.int64(size) for size in
                                    (packed_len, n_modes, kmax, n_elements))
            self.c2r_output = None
        else:
            kernel, grid = _packed_transpose_kernel(
                self.real_type, n_modes, n_trailing, packed_len, kmax)
            self.pack = kernel[(*grid, n_leading), block]
            self.pack_sizes = ()
            kernel, grid = _transpose_kernel(self.real_type, n_trailing, self.n_grid)
            self.transpose_back = kernel[(*grid, n_leading), block]
            self.c2r_output = cp.empty((self.n_transforms, self.n_grid), dtype)

    def __call__(self, cdata, gdata, scale):
        mean_weight = self.real_type(scale)
        mode_weight = self.real_type(0.5 * scale)        # mode k > 0 stands for +k and -k
        self.pack(cdata.reshape(-1), self.spectrum_reals, mean_weight, mode_weight,
                  *self.pack_sizes)
        if self.c2r_output is None:                      # In-place transform on fast axis
            self.plan.fft(self.spectrum, gdata.reshape(self.n_transforms, self.n_grid),
                          cufft_backend.CUFFT_INVERSE)
        else:
            self.plan.fft(self.spectrum, self.c2r_output, cufft_backend.CUFFT_INVERSE)
            self.transpose_back(self.c2r_output.reshape(-1), gdata.reshape(-1))


@lru_cache(maxsize=None)
def _get_transform(shape, dtype, axis, packed_len, kmax):
    return _PackedC2R(shape, dtype, axis, packed_len, kmax)


def irfft_packed_into(cdata, gdata, axis, kmax, scale=1.0):
    axis %= gdata.ndim
    _get_transform(gdata.shape, gdata.dtype, axis, cdata.shape[axis], kmax)(cdata, gdata, scale)


def clear_cache():
    _get_transform.cache_clear()
