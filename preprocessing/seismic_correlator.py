import math

import numpy as np


class SeismicCorrelator:
    """Correlates cable nodes with seismic event catalogues."""

    def compute_correlation(self, cable_nodes: list, seismic_catalog) -> np.ndarray:
        """Compute per-node seismic features.

        Args:
            cable_nodes: list of dicts with keys 'lat' and 'lon'.
            seismic_catalog: iterable of dicts with keys 'lat', 'lon',
                             'magnitude'.

        Returns:
            Array of shape (n_nodes, 3) where columns are:
                [n_events_within_500km, max_magnitude_within_500km, pgv_estimate]
        """
        events = list(seismic_catalog)
        n_nodes = len(cable_nodes)
        result = np.zeros((n_nodes, 3), dtype=float)

        for i, node in enumerate(cable_nodes):
            lat_n = float(node["lat"])
            lon_n = float(node["lon"])

            n_events = 0
            max_mag = 0.0
            pgv = 0.0

            for event in events:
                lat_e = float(event["lat"])
                lon_e = float(event["lon"])
                mag = float(event.get("magnitude", 0.0))
                dist = self.haversine(lat_n, lon_n, lat_e, lon_e)

                if dist <= 500.0:
                    n_events += 1
                    if mag > max_mag:
                        max_mag = mag

                dist_safe = max(dist, 1.0)
                pgv += mag / (dist_safe ** 2)

            result[i, 0] = float(n_events)
            result[i, 1] = max_mag
            result[i, 2] = pgv

        return result

    @staticmethod
    def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Great-circle distance between two points in kilometres.

        Args:
            lat1, lon1: coordinates of the first point in decimal degrees.
            lat2, lon2: coordinates of the second point in decimal degrees.

        Returns:
            Distance in km.
        """
        R = 6371.0
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = (math.sin(dphi / 2.0) ** 2
             + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2)
        return R * 2.0 * math.asin(math.sqrt(a))
