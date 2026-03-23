import torch


class EdgeFeatureBuilder:
    """Builds normalised edge feature tensors for cable graph edges."""

    def build(
        self,
        cable_dict: dict,
        landing_point_a: dict,
        landing_point_b: dict,
    ) -> torch.Tensor:
        """Construct a 6-dimensional edge feature tensor.

        Args:
            cable_dict: must contain 'length_km', 'rfs' (year), 'type',
                        and optionally 'depth_mean_m'.
            landing_point_a: dict with 'lat' and 'lon'.
            landing_point_b: dict with 'lat' and 'lon'.

        Returns:
            Tensor of shape (6,):
            [length_km/10000, age_years/50, cable_type_encoded,
             lat_diff, lon_diff, depth_mean/10000]
        """
        length_km = float(cable_dict.get("length_km", 0.0))
        age_years = self.compute_age(cable_dict.get("rfs", None))
        cable_type_enc = self.encode_cable_type(cable_dict.get("type", ""))
        depth_mean = float(cable_dict.get("depth_mean_m", 0.0))

        lat_a = float(landing_point_a.get("lat", 0.0))
        lon_a = float(landing_point_a.get("lon", 0.0))
        lat_b = float(landing_point_b.get("lat", 0.0))
        lon_b = float(landing_point_b.get("lon", 0.0))

        lat_diff = lat_b - lat_a
        lon_diff = lon_b - lon_a

        return torch.tensor(
            [
                length_km / 10000.0,
                age_years / 50.0,
                cable_type_enc,
                lat_diff,
                lon_diff,
                depth_mean / 10000.0,
            ],
            dtype=torch.float,
        )

    @staticmethod
    def encode_cable_type(cable_type: str) -> float:
        """Map cable type string to a numeric encoding.

        Args:
            cable_type: one of 'fiber', 'coaxial', 'hybrid', 'power'.

        Returns:
            0.0  → fiber
            0.25 → coaxial
            0.5  → hybrid
            0.75 → power
            1.0  → unknown / other
        """
        mapping = {
            "fiber": 0.0,
            "coaxial": 0.25,
            "hybrid": 0.5,
            "power": 0.75,
        }
        return mapping.get(str(cable_type).lower(), 1.0)

    @staticmethod
    def compute_age(rfs_year) -> float:
        """Compute cable age in years relative to 2024.

        Args:
            rfs_year: ready-for-service year (int, str, or None).

        Returns:
            2024 - rfs_year if valid, else 10.0.
        """
        try:
            return float(2024 - int(rfs_year))
        except (ValueError, TypeError):
            return 10.0
