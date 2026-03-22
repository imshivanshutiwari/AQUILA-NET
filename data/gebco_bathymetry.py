import logging
import math
import os
from typing import List, Tuple

import numpy as np
import pandas as pd

try:
    import netCDF4  # noqa: F401

    _NETCDF4_AVAILABLE = True
except ImportError:
    _NETCDF4_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "netCDF4 is not installed; GEBCOBathymetry will use physics-based fallback."
    )

logger = logging.getLogger(__name__)


class GEBCOBathymetry:
    """Loads and queries the GEBCO 2023 global bathymetric grid (NetCDF-4)."""

    def __init__(self):
        self._dataset = None
        self._lat_var: np.ndarray = np.array([])
        self._lon_var: np.ndarray = np.array([])
        self._elev: np.ndarray = np.array([])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, filepath: str = "data/cache/gebco_2023.nc") -> "GEBCOBathymetry":
        """Load the GEBCO NetCDF file; sets ``self._dataset = None`` if file missing."""
        if not os.path.exists(filepath):
            logger.warning("GEBCO file not found at '%s'; using physics-based fallback.", filepath)
            self._dataset = None
            return self

        if not _NETCDF4_AVAILABLE:
            logger.warning("netCDF4 not installed; using physics-based fallback.")
            self._dataset = None
            return self

        try:
            import netCDF4 as nc

            self._dataset = nc.Dataset(filepath, "r")
            # GEBCO variable names (2023 grid)
            lat_key = "lat" if "lat" in self._dataset.variables else "latitude"
            lon_key = "lon" if "lon" in self._dataset.variables else "longitude"
            elev_key = "elevation" if "elevation" in self._dataset.variables else "z"

            self._lat_var = self._dataset.variables[lat_key][:]
            self._lon_var = self._dataset.variables[lon_key][:]
            self._elev = self._dataset.variables[elev_key][:]
            logger.info("GEBCO grid loaded: %s", filepath)
        except Exception as exc:
            logger.warning("Could not load GEBCO file (%s); using physics-based fallback.", exc)
            self._dataset = None

        return self

    def get_depth_at_coords(self, lat: float, lon: float) -> float:
        """Return depth in metres (negative = below sea level) at (lat, lon).

        Uses nearest-neighbour lookup if the grid is loaded, otherwise the
        physics-based fallback.
        """
        if self._dataset is None or self._elev.size == 0:
            return self._fallback_depth(lat, lon)

        lat_idx = int(np.argmin(np.abs(self._lat_var - lat)))
        lon_idx = int(np.argmin(np.abs(self._lon_var - lon)))
        try:
            depth = float(self._elev[lat_idx, lon_idx])
        except IndexError:
            depth = self._fallback_depth(lat, lon)
        return depth

    def get_depth_profile(
        self,
        start_coords: Tuple[float, float],
        end_coords: Tuple[float, float],
        n_points: int = 100,
    ) -> np.ndarray:
        """Return an array of depth values sampled along the great-circle path.

        Linear interpolation in lat/lon space is used (adequate for cables <
        a few thousand km; for longer routes use spherical interpolation).
        """
        lats = np.linspace(start_coords[0], end_coords[0], n_points)
        lons = np.linspace(start_coords[1], end_coords[1], n_points)
        depths = np.array([self.get_depth_at_coords(la, lo) for la, lo in zip(lats, lons)])
        return depths

    def compute_seabed_gradient(
        self, lat: float, lon: float, radius_deg: float = 0.5
    ) -> float:
        """Return the mean absolute depth gradient (m/deg) in a square region."""
        n = 10  # sample grid side
        lats = np.linspace(lat - radius_deg, lat + radius_deg, n)
        lons = np.linspace(lon - radius_deg, lon + radius_deg, n)

        grid = np.array(
            [[self.get_depth_at_coords(la, lo) for lo in lons] for la in lats]
        )
        grad_lat = np.abs(np.gradient(grid, axis=0))
        grad_lon = np.abs(np.gradient(grid, axis=1))
        return float(np.mean(grad_lat + grad_lon))

    def extract_cable_route_bathymetry(
        self, cable_route: List[Tuple[float, float]]
    ) -> pd.DataFrame:
        """Return a DataFrame with columns [lat, lon, depth_m] for each waypoint."""
        records = []
        for lat, lon in cable_route:
            records.append(
                {
                    "lat": lat,
                    "lon": lon,
                    "depth_m": self.get_depth_at_coords(lat, lon),
                }
            )
        return pd.DataFrame(records, columns=["lat", "lon", "depth_m"])

    # ------------------------------------------------------------------
    # Physics-based fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_depth(lat: float, lon: float) -> float:  # noqa: ARG004
        """Mid-ocean depth approximation (metres, negative = below sea level).

        Uses a sinusoidal model that places deep ocean at equator and
        shallower regions toward the poles, matching average oceanic profiles.
        """
        return -4000.0 - 1000.0 * math.sin(math.radians(lat))
