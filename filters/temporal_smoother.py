import numpy as np
import scipy.ndimage
import scipy.signal


class TemporalSmoother:
    """Applies exponential, Gaussian, or median smoothing to time-series data."""

    def __init__(self, window: int = 5, method: str = "exponential", alpha: float = 0.3):
        if method not in ("exponential", "gaussian", "median"):
            raise ValueError(f"Unknown method '{method}'. Use 'exponential', 'gaussian', or 'median'.")
        self.window = window
        self.method = method
        self.alpha = alpha

    def smooth(self, series: np.ndarray) -> np.ndarray:
        """Smooth a 1-D array according to the configured method."""
        series = np.asarray(series, dtype=float)
        if self.method == "exponential":
            out = np.empty_like(series)
            out[0] = series[0]
            for i in range(1, len(series)):
                out[i] = self.alpha * series[i] + (1.0 - self.alpha) * out[i - 1]
            return out
        elif self.method == "gaussian":
            sigma = self.window / 4.0
            return scipy.ndimage.gaussian_filter1d(series, sigma=sigma)
        else:  # median
            kernel_size = self.window if self.window % 2 == 1 else self.window + 1
            return scipy.signal.medfilt(series, kernel_size=kernel_size).astype(float)

    def smooth_2d(self, matrix: np.ndarray) -> np.ndarray:
        """Apply smooth() independently to each row of a 2-D array."""
        matrix = np.asarray(matrix, dtype=float)
        return np.vstack([self.smooth(row) for row in matrix])
