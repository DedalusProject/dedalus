

import numpy as np
from scipy import fftpack

from . import basis


def register_transform(basis, name):
    """Decorator to add transform to basis class dictionary."""
    def wrapper(cls):
        if not hasattr(basis, 'transforms'):
            basis.transforms = {}
        basis.transforms[name] = cls
        return cls
    return wrapper


class Transform:
    pass


class PolynomialTransform(Transform):

    def __init__(self, coeff_shape, dtype, axis, scale):

        self.coeff_shape = coeff_shape
        self.dtype = dtype
        self.axis = axis
        self.scale = scale

        if dtype == np.complex128:
            coeff_shape = list(coeff_shape) + [2]

        self.N0 = N0 = np.prod(coeff_shape[:axis])
        self.N1C = N1C = coeff_shape[axis]
        self.N1G = N1G = int(self.N1C * scale)
        self.N2 = N2 = np.prod(coeff_shape[axis+1:])

        self.gdata_reduced = np.zeros(shape=[N0, N1G, N2], dtype=np.float64)
        self.cdata_reduced = np.zeros(shape=[N0, N1C, N2], dtype=np.float64)

    # def check_arrays(self, cdata, gdata, axis, scale=None):
    #     """
    #     Verify provided arrays sizes and dtypes are correct.
    #     Build compliant arrays if not provided.

    #     """

    #     if cdata is None:
    #         # Build cdata
    #         cshape = list(gdata.shape)
    #         cshape[axis] = self.coeff_size
    #         cdata = fftw.create_array(cshape, self.coeff_dtype)
    #     else:
    #         # Check cdata
    #         if cdata.shape[axis] != self.space.coeff_size:
    #             raise ValueError("cdata does not match coeff_size")
    #         if cdata.dtype != self.domain.dtype:
    #             raise ValueError("cdata does not match coeff_dtype")

    #     if scale:
    #         grid_size = self.space.grid_size(scale)

    #     if gdata is None:
    #         # Build gdata
    #         gshape = list(cdata.shape)
    #         gshape[axis] = grid_size
    #         gdata = fftw.create_array(gshape, self.grid_dtype)
    #     else:
    #         # Check gdata
    #         if scale and (gdata.shape[axis] != grid_size):
    #             raise ValueError("gdata does not match scaled grid_size")
    #         if gdata.dtype != self.domain.dtype:
    #             raise ValueError("gdata does not match grid_dtype")

    #     return cdata, gdata

    @staticmethod
    def resize_reduced(data_in, data_out):
        """Resize data by padding/truncation."""
        size_in = data_in.shape[1]
        size_out = data_out.shape[1]
        if size_in < size_out:
            # Pad with zeros at end of data
            np.copyto(data_out[:, :size_in, :], data_in)
            np.copyto(data_out[:, size_in:, :], 0)
        elif size_in > size_out:
            # Truncate higher order modes at end of data
            np.copyto(data_out, data_in[:, :size_out, :])
        else:
            np.copyto(data_out, data_in)

    def forward(self, gdata, cdata):
        # Make reduced view into input arrays
        self.gdata_reduced.data = gdata
        self.cdata_reduced.data = cdata
        # Transform reduced arrays
        self.forward_reduced()

    def backward(self, cdata, gdata):
        # Make reduced view into input arrays
        self.cdata_reduced.data = cdata
        self.gdata_reduced.data = gdata
        # Transform reduced arrays
        self.backward_reduced()


class ScipyDST(PolynomialTransform):

    def forward_reduced(self):
        # DST-II transform from interior points
        temp = fftpack.dst(self.gdata_reduced, type=2, axis=1)
        # Rescale as sinusoid amplitudes
        temp[:, -1, :] *= 0.5
        temp *= (1 / self.N1G)
        # Resize
        self.resize_reduced(temp, self.cdata_reduced)

    def backward_reduced(self):
        # Resize into gdata for memory efficiency
        self.resize_reduced(self.cdata_reduced, self.gdata_reduced)
        # Rescale from sinusoid amplitudes
        self.gdata_reduced[:, :-1, :] *= 0.5
        # DST-III transform to interior points
        temp = fftpack.dst(self.gdata_reduced, type=3, axis=1)
        np.copyto(self.gdata_reduced, temp)


@register_transform(basis.Sine, 'scipy')
class ScipySineTransform(ScipyDST):

    def forward_reduced(self):
        super().forward_reduced()
        # Shift data, adding zero mode and dropping Nyquist
        start = self.cdata_reduced[:, :-1, :]
        shift = self.cdata_reduced[:, 1:, :]
        np.copyto(shift, start)
        self.cdata_reduced[:, 0, :] = 0

    def backward_reduced(self):
        # Unshift data, adding Nyquist mode and dropping zero
        start = self.cdata_reduced[:, :-1, :]
        shift = self.cdata_reduced[:, 1:, :]
        np.copyto(start, shift)
        self.cdata_reduced[:, -1, :] = 0
        super().backward_reduced()


# @register_transform(basis.ChebyshevU, 'scipy')
# class ScipyChebyshevUTransform(ScipyDST):
#     """
#     f(x) = c_n U_n(x)
#     f(x) sin(θ) = c_n sin(θ) U_n(x)
#                 = c_n sin((n+1)θ)

#     c_n are the U_n coefficients of f(x)
#                 sine coefficients of f(x) sin(θ)
#     """

#     def forward_reduced(self):
#         self.gdata_reduced *= sin_θ
#         super().forward_reduced()

#     def backward_reduce(self):
#         super().backward_reduced()
#         self.gdata_reduced *= (1 / sin_θ)


# cdef class FFTWSineTransform:

#     @CachedMethod
#     def _fftw_dst_setup(self, dtype, gshape, axis):
#         """Build FFTW DST plan and temporary array."""
#         flags = ['FFTW_'+FFTW_RIGOR.upper()]
#         logger.debug("Building FFTW DST plan for (dtype, gshape, axis) = (%s, %s, %s)" %(dtype, gshape, axis))
#         plan = fftw.DiscreteSineTransform(dtype, gshape, axis, flags=flags)
#         temp = fftw.create_array(gshape, dtype)
#         return plan, temp

#     def _forward_fftw(self, gdata, cdata, axis, scale):
#         """Forward transform using FFTW DCT."""
#         cdata, gdata = self.check_arrays(cdata, gdata, axis)
#         plan, temp = self._fftw_dst_setup(gdata.dtype, gdata.shape, axis)
#         plan.forward(gdata, temp)
#         self._forward_dst_scaling(temp, axis)
#         self._resize_coeffs(temp, cdata, axis)
#         return cdata

#     def _backward_fftw(self, cdata, gdata, axis, scale):
#         """Backward transform using FFTW IDCT."""
#         cdata, gdata = self.check_arrays(cdata, gdata, axis, scale)
#         plan, temp = self._fftw_dst_setup(gdata.dtype, gdata.shape, axis)
#         self._resize_coeffs(cdata, temp, axis)
#         self._backward_dst_scaling(temp, axis)
#         plan.backward(temp, gdata)
#         return gdata


@register_transform(basis.Cosine, 'scipy')
class ScipyDCT(PolynomialTransform):

    def forward_reduced(self):
        # DCT-II transform from interior points
        temp = fftpack.dct(self.gdata_reduced, type=2, axis=1)
        # Rescale as sinusoid amplitudes
        temp[:, 0, :] *= 0.5
        temp *= (1 / self.N1G)
        # Resize
        self.resize_reduced(temp, self.cdata_reduced)

    def backward_reduced(self):
        # Resize into gdata for memory efficiency
        self.resize_reduced(self.cdata_reduced, self.gdata_reduced)
        # Rescale from sinusoid amplitudes
        self.gdata_reduced[:, 1:, :] *= 0.5
        # DCT-III transform to interior points
        temp = fftpack.dct(self.gdata_reduced, type=3, axis=1)
        np.copyto(self.gdata_reduced, temp)


@register_transform(basis.ChebyshevT, 'scipy')
class ScipyChebyshevTTransform(ScipyDCT):

    def forward_reduced(self):
        super().forward_reduced()
        # Negate odd modes for natural grid ordering
        self.cdata_reduced[:, 1::2, :] *= -1

    def backward_reduced(self):
        # Negate odd modes for natural grid ordering
        self.cdata_reduced[:, 1::2, :] *= -1
        super().backward_reduced()


# class FFTWCosine:

#     @CachedMethod
#     def _fftw_dct_setup(self, dtype, gshape, axis):
#         """Build FFTW DCT plan and temporary array."""
#         flags = ['FFTW_'+FFTW_RIGOR.upper()]
#         logger.debug("Building FFTW DCT plan for (dtype, gshape, axis) = (%s, %s, %s)" %(dtype, gshape, axis))
#         plan = fftw.DiscreteCosineTransform(dtype, gshape, axis, flags=flags)
#         temp = fftw.create_array(gshape, dtype)
#         return plan, temp

#     def _forward_fftw(self, gdata, cdata, axis, scale):
#         """Forward transform using FFTW DCT."""
#         cdata, gdata = self.check_arrays(cdata, gdata, axis)
#         plan, temp = self._fftw_dct_setup(gdata.dtype, gdata.shape, axis)
#         plan.forward(gdata, temp)
#         self._forward_dct_scaling(temp, axis)
#         self._resize_coeffs(temp, cdata, axis)
#         return cdata

#     def _backward_fftw(self, cdata, gdata, axis, scale):
#         """Backward transform using FFTW IDCT."""
#         cdata, gdata = self.check_arrays(cdata, gdata, axis, scale)
#         plan, temp = self._fftw_dct_setup(gdata.dtype, gdata.shape, axis)
#         self._resize_coeffs(cdata, temp, axis)
#         self._backward_dct_scaling(temp, axis)
#         plan.backward(temp, gdata)
#         return gdata


class ScipyRFFT(PolynomialTransform):

    def forward_reduced(self):
        # RFFT
        temp = fftpack.rfft(self.gdata_reduced, axis=1)
        # Rescale as sinusoid amplitudes
        scaling = np.ones([1, self.N1G, 1]) / self.N1G
        scaling[1:-1] *= 2
        scaling[2::2] *= -1
        temp *= scaling
        # Resize
        self.resize_reduced(temp, self.cdata_reduced)

    def backward_reduced(self):
        # Resize into gdata for memory efficiency
        self.resize_reduced(self.cdata_reduced, self.gdata_reduced)
        # Rescale from sinusoid amplitudes
        scaling = np.ones([1, self.N1G, 1]) / self.N1G
        scaling[1:-1] *= 2
        scaling[2::2] *= -1
        self.gdata_reduced *= 1 / scaling
        # IRFFT
        temp = fftpack.irfft(self.gdata_reduced, axis=1)
        np.copyto(self.gdata_reduced, temp)


@register_transform(basis.Fourier, 'scipy')
class ScipyFourierTransform(ScipyRFFT):

    def forward_reduced(self):
        super().forward_reduced()
        # Shift data, adding zero sine mode and dropping Nyquist cosine
        start = self.cdata_reduced[:, 1:-1, :]
        shift = self.cdata_reduced[:, 2:, :]
        np.copyto(shift, start)
        self.cdata_reduced[:, 1, :] = 0

    def backward_reduced(self):
        # Unshift data, adding Nyquist cosine and dropping zero sine
        start = self.cdata_reduced[:, 1:-1, :]
        shift = self.cdata_reduced[:, 2:, :]
        np.copyto(start, shift)
        self.cdata_reduced[:, -1, :] = 0
        super().backward_reduced()


# class FFTWFourierTransform:

#     @CachedMethod
#     def _fftw_setup(self, dtype, gshape, axis):
#         """Build FFTW plans and temporary arrays."""
#         # Note: regular method used to cache through basis instance

#         logger.debug("Building FFTW FFT plan for (dtype, gshape, axis) = (%s, %s, %s)" %(dtype, gshape, axis))
#         flags = ['FFTW_'+FFTW_RIGOR.upper()]
#         plan = fftw.FourierTransform(dtype, gshape, axis, flags=flags)
#         temp = fftw.create_array(plan.cshape, np.complex128)
#         if dtype == np.float64:
#             resize_coeffs = self._resize_real_coeffs
#         elif dtype == np.complex128:
#             resize_coeffs = self._resize_complex_coeffs

#         return plan, temp, resize_coeffs

#     def _forward_fftw(self, gdata, cdata, axis, meta):
#         """Forward transform using FFTW FFT."""

#         cdata, gdata = self.check_arrays(cdata, gdata, axis)
#         plan, temp, resize_coeffs = self._fftw_setup(gdata.dtype, gdata.shape, axis)
#         # Execute FFTW plan
#         plan.forward(gdata, temp)
#         # Scale FFT output to mode amplitudes
#         temp *= 1 / gdata.shape[axis]
#         # Pad / truncate coefficients
#         resize_coeffs(temp, cdata, axis, gdata.shape[axis])

#         return cdata

#     def _backward_fftw(self, cdata, gdata, axis, meta):
#         """Backward transform using FFTW IFFT."""

#         cdata, gdata = self.check_arrays(cdata, gdata, axis, meta)
#         plan, temp, resize_coeffs = self._fftw_setup(gdata.dtype, gdata.shape, axis)
#         # Pad / truncate coefficients
#         resize_coeffs(cdata, temp, axis, gdata.shape[axis])
#         # Execute FFTW plan
#         plan.backward(temp, gdata)

#         return gdata

