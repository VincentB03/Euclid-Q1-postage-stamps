import os

import numpy as np
from astropy.io import fits

from config import PSF_SIZE


class EuclidPSFModel:
    """PSF grid of one VIS quadrant (adapted from https://github.com/CosmoStat/SHINE).

    ``psf_data`` is the 2-D tile (e.g. 189 x 189) of ``grid_ny x grid_nx``
    stamps of ``stamp_size`` px, sampled evenly over a ``quad_nx x quad_ny``
    quadrant.
    """

    def __init__(self, psf_data: np.ndarray, stamp_size: int = PSF_SIZE, grid_nx: int = 9, grid_ny: int = 9, quad_nx: int = 2048, quad_ny: int = 2066) -> None:
        self.stamps = psf_data.reshape(grid_ny, stamp_size, grid_nx, stamp_size).transpose(0, 2, 1, 3)
        self.stamp_size = stamp_size
        self.grid_nx = grid_nx
        self.grid_ny = grid_ny
        self.grid_x = np.linspace(quad_nx / (2 * grid_nx), quad_nx - quad_nx / (2 * grid_nx), grid_nx)
        self.grid_y = np.linspace(quad_ny / (2 * grid_ny), quad_ny - quad_ny / (2 * grid_ny), grid_ny)

    def interpolate_at(self, x_pix: float, y_pix: float) -> np.ndarray:
        """Bilinear interpolation of the grid at a quadrant pixel position, normalised to sum 1."""
        ix = np.searchsorted(self.grid_x, x_pix) - 1
        ix = int(np.clip(ix, 0, self.grid_nx - 2))
        iy = np.searchsorted(self.grid_y, y_pix) - 1
        iy = int(np.clip(iy, 0, self.grid_ny - 2))

        dx = self.grid_x[ix + 1] - self.grid_x[ix]
        dy = self.grid_y[iy + 1] - self.grid_y[iy]

        wx = (x_pix - self.grid_x[ix]) / dx if dx > 0 else 0.5
        wy = (y_pix - self.grid_y[iy]) / dy if dy > 0 else 0.5
        wx, wy = float(np.clip(wx, 0.0, 1.0)), float(np.clip(wy, 0.0, 1.0))

        stamp = ((1 - wx) * (1 - wy) * self.stamps[iy, ix] + wx * (1 - wy) * self.stamps[iy, ix + 1] +
                 (1 - wx) * wy * self.stamps[iy + 1, ix] + wx * wy * self.stamps[iy + 1, ix + 1])

        total = stamp.sum()
        if total > 0:
            stamp = stamp / total
        return stamp


# ---------------------------------------------------------------------------
# Residual PSF kernels: psf_stamp = psf_ref (*) kernel, with psf_ref a fixed
# isotropic reference PSF; the kernel is recovered by Fourier division.
# ---------------------------------------------------------------------------
DEFAULT_REFERENCE_PSF = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'euclid_vis_isotropic_min_psf.fits'
)


def centered_fft2(image: np.ndarray) -> np.ndarray:
    """FFT over the last two axes of a centred image (no phase ramp)."""
    return np.fft.fft2(np.fft.ifftshift(image, axes=(-2, -1)), axes=(-2, -1))


def centered_ifft2(spectrum: np.ndarray) -> np.ndarray:
    """Inverse of :func:`centered_fft2`, returning a centred real image."""
    return np.fft.fftshift(np.fft.ifft2(spectrum, axes=(-2, -1)).real, axes=(-2, -1))


def load_reference_psf(path: str = DEFAULT_REFERENCE_PSF, normalize: bool = True) -> np.ndarray:
    """First 2-D HDU of ``path`` as float64, normalised to sum 1 if ``normalize``."""
    with fits.open(path) as hdul:
        data = next(h.data for h in hdul if h.data is not None and np.ndim(h.data) == 2)

    psf_ref = np.asarray(data, dtype=np.float64)
    if normalize:
        total = psf_ref.sum()
        if total > 0:
            psf_ref = psf_ref / total
    return psf_ref


def compute_psf_residual(psf_stamp: np.ndarray, psf_ref: np.ndarray,
                         ref_fft: np.ndarray = None, epsilon: float = 0.0) -> np.ndarray:
    """Kernel(s) solving ``psf_stamp = psf_ref (*) kernel``, same shape as ``psf_stamp``.

    ``psf_stamp`` is ``(ny, nx)`` or ``(n, ny, nx)``. ``ref_fft`` caches
    ``centered_fft2(psf_ref)``; ``epsilon > 0`` switches to a Tikhonov-regularised
    division.
    """
    psf_stamp = np.asarray(psf_stamp, dtype=np.float64)
    psf_ref = np.asarray(psf_ref, dtype=np.float64)

    if psf_stamp.shape[-2:] != psf_ref.shape:
        raise ValueError(
            f"PSF stamp shape {psf_stamp.shape[-2:]} and reference shape "
            f"{psf_ref.shape} are incompatible."
        )

    b = centered_fft2(psf_ref) if ref_fft is None else ref_fft
    x = centered_fft2(psf_stamp)

    if epsilon > 0:
        kernel_fft = x * np.conj(b) / (np.abs(b) ** 2 + epsilon)
    else:
        kernel_fft = x / b

    return centered_ifft2(kernel_fft)


def reconvolve_psf(residual_kernel: np.ndarray, psf_ref: np.ndarray,
                   ref_fft: np.ndarray = None) -> np.ndarray:
    """``psf_ref (*) residual_kernel``: inverse of :func:`compute_psf_residual` (``epsilon=0``)."""
    residual_kernel = np.asarray(residual_kernel, dtype=np.float64)
    psf_ref = np.asarray(psf_ref, dtype=np.float64)

    b = centered_fft2(psf_ref) if ref_fft is None else ref_fft
    k = centered_fft2(residual_kernel)
    return centered_ifft2(b * k)
