import numpy as np
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