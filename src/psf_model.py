import os

import numpy as np
from astropy.io import fits

from config import PSF_SIZE

class EuclidPSFModel:
    """
    Code from SHINE repository https://github.com/CosmoStat/SHINE

    Tiled PSF grid for a single Euclid VIS quadrant.

    The PSF FITS extension stores a single 2-D image that tiles
    ``grid_ny x grid_nx`` individual stamps of size
    ``stamp_size x stamp_size``.  This class splits the tile into a
    4-D array and provides nearest-neighbour and bilinear interpolation
    at arbitrary pixel positions.

    Args:
        psf_data: Raw 2-D PSF tile array (e.g. 189 x 189).
        stamp_size: Side length of each individual PSF stamp.
        grid_nx: Number of stamps along the x (column) axis.
        grid_ny: Number of stamps along the y (row) axis.
        quad_nx: Quadrant width in pixels.
        quad_ny: Quadrant height in pixels.
    """

    def __init__(self, psf_data: np.ndarray, stamp_size: int = PSF_SIZE, grid_nx: int = 9, grid_ny: int = 9, quad_nx: int = 2048, quad_ny: int = 2066) -> None:
        self.stamps = psf_data.reshape(grid_ny, stamp_size, grid_nx, stamp_size).transpose(0, 2, 1, 3)
        self.stamp_size = stamp_size
        self.grid_nx = grid_nx
        self.grid_ny = grid_ny
        self.grid_x = np.linspace(quad_nx / (2 * grid_nx), quad_nx - quad_nx / (2 * grid_nx), grid_nx)
        self.grid_y = np.linspace(quad_ny / (2 * grid_ny), quad_ny - quad_ny / (2 * grid_ny), grid_ny)

    def interpolate_at(self, x_pix: float, y_pix: float) -> np.ndarray:
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
# Residual PSF kernels
#
# Each per-object PSF stamp is modelled as ``psf_ref (*) kernel`` where
# ``psf_ref`` is a fixed isotropic reference PSF and ``(*)`` is a 2-D
# convolution.  The *residual kernel* is what the stamp adds on top of that
# reference (anisotropy, breathing, ...) and is recovered by a division in
# Fourier space.
# ---------------------------------------------------------------------------

# Drop the isotropic reference PSF FITS here (added manually).
DEFAULT_REFERENCE_PSF = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'euclid_vis_isotropic_min_psf.fits'
)


def centered_fft2(image: np.ndarray) -> np.ndarray:
    """Forward FFT of a centred PSF (peak in the middle of the array).

    ``ifftshift`` first moves the central pixel to index ``(0, 0)`` so the
    transform carries no linear phase ramp.  Operates on the last two axes,
    so ``image`` may be 2-D ``(ny, nx)`` or a stack ``(n, ny, nx)``.
    """
    return np.fft.fft2(np.fft.ifftshift(image, axes=(-2, -1)), axes=(-2, -1))


def centered_ifft2(spectrum: np.ndarray) -> np.ndarray:
    """Inverse of :func:`centered_fft2`: return a centred real-valued image."""
    return np.fft.fftshift(np.fft.ifft2(spectrum, axes=(-2, -1)).real, axes=(-2, -1))


def load_reference_psf(path: str = DEFAULT_REFERENCE_PSF, normalize: bool = True) -> np.ndarray:
    """Load the isotropic reference PSF used to build residual kernels.

    Args:
        path: Path to the reference PSF FITS file.  The first HDU holding a
            2-D array is used.
        normalize: If True, rescale so the PSF sums to 1.

    Returns:
        The reference PSF as a 2-D float64 array.
    """
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
    """Compute the residual PSF kernel(s) of ``psf_stamp`` relative to ``psf_ref``.

    Solves ``psf_stamp = psf_ref (*) kernel`` for ``kernel`` by dividing the
    two in Fourier space.

    Args:
        psf_stamp: A single PSF stamp ``(ny, nx)`` or a stack ``(n, ny, nx)``.
        psf_ref: The reference PSF, matching the ``(ny, nx)`` of each stamp.
        ref_fft: Optional pre-computed ``centered_fft2(psf_ref)``; pass it to
            avoid recomputing the reference transform on every call/batch.
        epsilon: Tikhonov regularisation on the (power-normalised) division.
            ``0.0`` reproduces the plain Fourier division; a small positive
            value damps high-frequency noise amplification.

    Returns:
        The residual kernel(s), a real array centred (peak in the middle)
        with the same shape as ``psf_stamp``.
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
    """Rebuild PSF stamp(s) from residual kernel(s): ``psf_ref (*) residual_kernel``.

    Inverse of :func:`compute_psf_residual` with ``epsilon=0``.  Use it to
    check that a residual kernel round-trips back to the original stamp.

    Args:
        residual_kernel: A kernel ``(ny, nx)`` or a stack ``(n, ny, nx)``.
        psf_ref: The reference PSF, matching the ``(ny, nx)`` of each kernel.
        ref_fft: Optional pre-computed ``centered_fft2(psf_ref)``.

    Returns:
        The reconstructed PSF stamp(s), centred, same shape as
        ``residual_kernel``.
    """
    residual_kernel = np.asarray(residual_kernel, dtype=np.float64)
    psf_ref = np.asarray(psf_ref, dtype=np.float64)

    b = centered_fft2(psf_ref) if ref_fft is None else ref_fft
    k = centered_fft2(residual_kernel)
    return centered_ifft2(b * k)